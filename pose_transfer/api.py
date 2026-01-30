"""
Pose Transfer API Module (v3 - 중앙 집중식 출력 제어)

실행 모드:
1. Standalone Mode: pose_extractor 단독 실행 → io/ 폴더 사용
2. Module Mode (외부 경로 주입): 외부에서 경로 직접 전달
3. Pozibility Mode: modules/pose_extractor로 통합 시 → data/ 폴더 사용

출력 제어:
- 핵심 결과물: trans_kp.json, trans_sk.jpg (항상 생성)
- 디버그 결과물: output.debug 설정으로 제어
"""
import sys
import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

# OpenMP 중복 로딩 에러 방지 (onnxruntime/rtmlib 충돌 회피)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .pipeline import PipelineConfig, PoseTransferPipeline
from .utils.io import save_json, save_image, load_image
from .logic.debug_generator import generate_debug_text
from .utils.face_transfer_visualizer import generate_face_transfer_image


# ====================================================
# [Output Config] 출력 설정 구조체
# ====================================================
@dataclass
class OutputConfig:
    """출력 설정 - 핵심 vs 디버그 분리"""
    # 핵심 결과물 (항상 생성 권장)
    save_json: bool = True
    save_skeleton_image: bool = True
    
    # 디버그 마스터 스위치
    debug_enabled: bool = False
    
    # 개별 디버그 출력
    save_overlay: bool = False        # *_rend.jpg
    save_bbox: bool = False           # *_debug_bbox.jpg
    save_debug_text: bool = False     # *_debug.txt
    save_depth: bool = False          # *_depth.png
    save_face_viz: bool = False       # face_transfer_debug.jpg
    
    @classmethod
    def from_yaml(cls, yaml_config: dict) -> 'OutputConfig':
        """YAML config에서 OutputConfig 생성"""
        output_cfg = yaml_config.get('output', {})
        debug_cfg = output_cfg.get('debug', {})
        
        # 디버그 마스터 스위치 확인
        debug_enabled = debug_cfg.get('enabled', False)
        
        # Legacy 호환: 기존 save_debug_image, save_debug_txt 지원
        legacy_debug_image = output_cfg.get('save_debug_image', False)
        legacy_debug_txt = output_cfg.get('save_debug_txt', False)
        
        return cls(
            save_json=output_cfg.get('save_json', True),
            save_skeleton_image=output_cfg.get('save_skeleton_image', True),
            debug_enabled=debug_enabled or legacy_debug_image or legacy_debug_txt,
            save_overlay=debug_cfg.get('save_overlay', legacy_debug_image),
            save_bbox=debug_cfg.get('save_bbox', legacy_debug_image),
            save_debug_text=debug_cfg.get('save_text', legacy_debug_txt),
            save_depth=debug_cfg.get('save_depth', False),
            save_face_viz=debug_cfg.get('save_face_viz', False),
        )
    
    def should_save_debug(self, item: str) -> bool:
        """특정 디버그 항목을 저장해야 하는지 확인"""
        if not self.debug_enabled:
            return False
        return getattr(self, f'save_{item}', False)


# ====================================================
# [Path Resolution] 경로 결정 로직
# ====================================================
def resolve_data_paths() -> Tuple[Path, Path, Path]:
    """
    실행 환경에 따라 input/output 경로 결정
    
    Returns:
        (src_dir, ref_dir, output_dir)
    """
    pose_transfer_dir = Path(__file__).resolve().parent
    module_root = pose_transfer_dir.parent
    potential_pozibility_root = module_root.parent.parent
    pozibility_data = potential_pozibility_root / "data"
    
    # [Case 1] Pozibility 통합 모드
    if pozibility_data.exists() and (pozibility_data / "inputs").exists():
        print("ℹ️  [API] Pozibility Mode: data/ 폴더를 사용합니다.")
        return (
            pozibility_data / "inputs" / "src",
            pozibility_data / "inputs" / "ref",
            pozibility_data / "preprocess_outputs"
        )
    
    # [Case 2] Standalone 모드
    io_dir = module_root / "io"
    if io_dir.exists():
        print("ℹ️  [API] Standalone Mode: io/ 폴더를 사용합니다.")
        return (
            io_dir / "inputs" / "src",
            io_dir / "inputs" / "ref",
            io_dir / "outputs"
        )
    
    # [Case 3] 폴더 없음 - 기본 io/ 생성
    print("ℹ️  [API] io/ 폴더가 없어 생성합니다.")
    io_dir.mkdir(parents=True, exist_ok=True)
    (io_dir / "inputs" / "src").mkdir(parents=True, exist_ok=True)
    (io_dir / "inputs" / "ref").mkdir(parents=True, exist_ok=True)
    (io_dir / "outputs").mkdir(parents=True, exist_ok=True)
    
    return (
        io_dir / "inputs" / "src",
        io_dir / "inputs" / "ref",
        io_dir / "outputs"
    )


def find_first_image(directory: Path) -> Optional[Path]:
    """디렉토리에서 첫 번째 이미지 파일 찾기"""
    if not directory.exists():
        return None
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    
    return files[0] if files else None


# ====================================================
# [API] 외부에서 호출하는 핵심 함수
# ====================================================
def execute_pose_transfer(
    source_path: Union[str, Path] = None,
    reference_path: Union[str, Path] = None,
    output_root: Union[str, Path] = None,
    config_path: str = None,
    explicit_config: Optional[dict] = None
) -> Dict[str, str]:
    """
    포즈 전이 실행
    
    Args:
        source_path: Source 이미지 경로 (None이면 자동 탐색)
        reference_path: Reference 이미지 경로 (None이면 자동 탐색)
        output_root: 출력 디렉토리 (None이면 자동 결정)
        config_path: 설정 파일 경로
        explicit_config: 명시적 설정 딕셔너리
    
    Returns:
        결과 파일 경로 딕셔너리
    """
    
    # 1. 설정 파일 로드
    if config_path is None:
        base_dir = Path(__file__).resolve().parent
        config_path = base_dir / "config" / "default.yaml"
    
    config_p = Path(config_path)
    yaml_config = explicit_config or {}
    
    if not yaml_config and config_p.exists():
        with open(config_p, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

    # 2. 경로 결정
    final_src_path = None
    final_ref_path = None
    final_output_dir = None
    
    if source_path is not None and reference_path is not None:
        print("ℹ️  [API] Module Mode: 외부 경로를 사용합니다.")
        final_src_path = Path(source_path)
        final_ref_path = Path(reference_path)
        
        if output_root is not None:
            final_output_dir = Path(output_root)
        else:
            _, _, auto_output = resolve_data_paths()
            final_output_dir = auto_output
    else:
        src_dir, ref_dir, auto_output_dir = resolve_data_paths()
        final_output_dir = Path(output_root) if output_root is not None else auto_output_dir
        
        final_src_path = find_first_image(src_dir)
        final_ref_path = find_first_image(ref_dir)
        
        if final_src_path is None:
            raise FileNotFoundError(
                f"❌ Source 이미지를 찾을 수 없습니다.\n"
                f"👉 경로: {src_dir}\n"
                f"💡 해당 폴더에 이미지 파일을 넣어주세요."
            )
        
        if final_ref_path is None:
            raise FileNotFoundError(
                f"❌ Reference 이미지를 찾을 수 없습니다.\n"
                f"👉 경로: {ref_dir}\n"
                f"💡 해당 폴더에 이미지 파일을 넣어주세요."
            )
        
        print(f"    👉 Source: {final_src_path.name}")
        print(f"    👉 Reference: {final_ref_path.name}")
        print(f"    👉 Output: {final_output_dir}")

    # 3. 파일 존재 확인
    if not final_src_path.exists():
        raise FileNotFoundError(f"Source file not found: {final_src_path}")
    if not final_ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {final_ref_path}")

    # 4. 출력 설정 로드
    output_config = OutputConfig.from_yaml(yaml_config)
    
    # 5. 파이프라인 설정
    if config_p.exists():
        pipeline_config = PipelineConfig.from_yaml(str(config_p))
    else:
        pipeline_config = PipelineConfig()

    # 작업 ID 생성
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    job_id = f"{date_str}_{time_str}_{final_src_path.stem}_to_{final_ref_path.stem}"
    
    # 출력 디렉토리 생성
    out_dirs = _setup_directories(str(final_output_dir), job_id)

    print(f"\n🚀 [Start Job] {job_id}")
    
    # use_face 설정 가져오기
    face_rendering_config = yaml_config.get('face_rendering', {})
    use_face_setting = face_rendering_config.get('enabled', True)

    try:
        pipeline = PoseTransferPipeline(pipeline_config, yaml_config=yaml_config)
        
        # ============================================================
        # [STEP 1] Source/Reference 분석 및 저장
        # ============================================================
        print("📊 Analyzing Inputs...")
        print(f"   ▶ SRC analyze start: {final_src_path.name}")
        _save_analysis(
            pipeline, final_src_path, out_dirs["src"], "src",
            output_config, use_face=use_face_setting
        )
        print(f"   ✅ SRC analyze done: {final_src_path.name}")
        
        print(f"   ▶ REF analyze start: {final_ref_path.name}")
        _save_analysis(
            pipeline, final_ref_path, out_dirs["ref"], "ref",
            output_config, use_face=use_face_setting
        )
        print(f"   ✅ REF analyze done: {final_ref_path.name}")
        
        # ============================================================
        # [STEP 2] 디버그 정보 생성 (Cross-Filter 분석)
        # ============================================================
        if output_config.should_save_debug('debug_text'):
            print("📝 Generating Debug Info...")
            _save_debug_texts(pipeline, final_src_path, final_ref_path, out_dirs, yaml_config)
        
        # ============================================================
        # [STEP 3] Transfer 실행
        # ============================================================
        print("✨ Running Transfer...")
        result = pipeline.transfer(final_src_path, final_ref_path)
        
        res_paths = {}
        
        # ============================================================
        # [STEP 4] 핵심 결과물 저장 (항상)
        # ============================================================
        # 4-1. 배경 이미지
        path_bg = out_dirs["trans"] / "trans_bg.jpg"
        final_bg = result.modified_source_image if result.modified_source_image is not None else load_image(final_src_path)
        save_image(final_bg, str(path_bg))
        res_paths['background'] = str(path_bg)
        
        # 4-2. JSON (핵심)
        if output_config.save_json:
            path_json = out_dirs["trans"] / "trans_kp.json"
            save_json(result.to_json(), str(path_json))
            res_paths['json'] = str(path_json)
        
        # 4-3. Skeleton 이미지 (핵심)
        if output_config.save_skeleton_image:
            path_skel = out_dirs["trans"] / "trans_sk.jpg"
            save_image(result.skeleton_image, str(path_skel))
            res_paths['skeleton'] = str(path_skel)
        
        # ============================================================
        # [STEP 5] 디버그 결과물 저장 (옵션)
        # ============================================================
        # 5-1. Overlay 이미지
        if output_config.should_save_debug('overlay'):
            path_overlay = out_dirs["trans"] / "trans_rend.jpg"
            overlay = pipeline.renderer.render(final_bg, result.transferred_keypoints, result.transferred_scores)
            save_image(overlay, str(path_overlay))
            res_paths['overlay'] = str(path_overlay)
        
        # 5-2. BBox 디버그 이미지
        if output_config.should_save_debug('bbox'):
            if result.src_debug_image is not None:
                save_image(result.src_debug_image, str(out_dirs["src"] / "src_debug_bbox.jpg"))
            if result.ref_debug_image is not None:
                save_image(result.ref_debug_image, str(out_dirs["ref"] / "ref_debug_bbox.jpg"))
        
        # 5-3. Depth 시각화
        if output_config.should_save_debug('depth'):
            _save_depth_outputs(pipeline, result, final_src_path, final_ref_path, out_dirs)
        
        # 5-4. Transfer 디버그 텍스트
        if output_config.should_save_debug('debug_text'):
            _save_transfer_debug(result, final_src_path, final_ref_path, out_dirs["trans"])
        
        # 5-5. Face Transfer 시각화
        if output_config.should_save_debug('face_viz'):
            _save_face_visualization(result, out_dirs["trans"])

        res_paths['job_dir'] = str(out_dirs['root'])
        
        print(f"✅ Finished Job: {out_dirs['root']}")
        return res_paths

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Pose transfer failed: {e}")


# ====================================================
# [Internal] 출력 헬퍼 함수들
# ====================================================
def _setup_directories(output_root: str, job_id: str):
    """출력 디렉토리 구조 생성"""
    base_dir = Path(output_root) / job_id
    dirs = {
        "root": base_dir,
        "src": base_dir / "src",
        "ref": base_dir / "ref",
        "trans": base_dir / "trans"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_analysis(
    pipeline, 
    image_path: Path, 
    output_dir: Path, 
    prefix: str, 
    output_config: OutputConfig,
    use_face: bool = True
):
    """
    포즈 추출 및 저장 (중앙 집중식)
    
    핵심: _kp.json, _sk.jpg
    디버그: _rend.jpg
    """
    json_data, skel_img, overlay_img = pipeline.extract_and_render(image_path, use_face=use_face)
    
    # 핵심 결과물
    if output_config.save_json:
        save_json(json_data, str(output_dir / f"{prefix}_kp.json"))
    if output_config.save_skeleton_image:
        save_image(skel_img, str(output_dir / f"{prefix}_sk.jpg"))
    
    # 디버그 결과물
    if output_config.should_save_debug('overlay'):
        save_image(overlay_img, str(output_dir / f"{prefix}_rend.jpg"))


def _save_debug_texts(
    pipeline,
    src_path: Path,
    ref_path: Path,
    out_dirs: dict,
    yaml_config: dict
):
    """Cross-Filter 디버그 텍스트 저장"""
    try:
        # Source 디버그
        src_debug_text = generate_debug_text(
            src_path,
            dw_extractor=pipeline.extractor,
            body_extractor=None,
            config=yaml_config
        )
        if src_debug_text:
            src_debug_path = out_dirs["src"] / f"{src_path.stem}_debug.txt"
            src_debug_path.write_text(src_debug_text, encoding='utf-8')
            print(f"   ✅ Source debug: {src_debug_path.name}")
        
        # Reference 디버그
        ref_debug_text = generate_debug_text(
            ref_path,
            dw_extractor=pipeline.extractor,
            body_extractor=None,
            config=yaml_config
        )
        if ref_debug_text:
            ref_debug_path = out_dirs["ref"] / f"{ref_path.stem}_debug.txt"
            ref_debug_path.write_text(ref_debug_text, encoding='utf-8')
            print(f"   ✅ Reference debug: {ref_debug_path.name}")
            
    except Exception as e:
        print(f"   ⚠️ Debug 생성 중 오류 (무시): {e}")


def _save_depth_outputs(pipeline, result, src_path: Path, ref_path: Path, out_dirs: dict):
    """Depth 시각화 저장"""
    try:
        if result.processing_info and 'depth_maps' in result.processing_info:
            depth_maps = result.processing_info['depth_maps']
            if 'src' in depth_maps:
                _save_depth_visual(depth_maps['src'], out_dirs["src"] / f"{src_path.stem}_depth.png")
            if 'ref' in depth_maps:
                _save_depth_visual(depth_maps['ref'], out_dirs["ref"] / f"{ref_path.stem}_depth.png")
        elif getattr(pipeline, 'depth_extractor', None) is not None:
            src_img = load_image(src_path)
            ref_img = load_image(ref_path)
            src_depth_map = pipeline.depth_extractor.estimate(src_img)
            ref_depth_map = pipeline.depth_extractor.estimate(ref_img)
            _save_depth_visual(src_depth_map, out_dirs["src"] / f"{src_path.stem}_depth.png")
            _save_depth_visual(ref_depth_map, out_dirs["ref"] / f"{ref_path.stem}_depth.png")
            print("   ✅ Depth visuals saved (fallback)")
    except Exception as e:
        print(f"   ⚠️ Depth visualization failed: {e}")


def _save_depth_visual(depth_map: np.ndarray, output_path: Path) -> None:
    """Depth map을 시각화 이미지로 저장"""
    depth = depth_map.astype(np.float32)
    d_min = float(np.percentile(depth, 1))
    d_max = float(np.percentile(depth, 99))
    depth = (depth - d_min) / (d_max - d_min + 1e-6)
    depth = np.clip(depth, 0.0, 1.0)
    depth_u8 = (depth * 255).astype(np.uint8)
    save_image(depth_u8, str(output_path))


def _save_transfer_debug(result, src_path: Path, ref_path: Path, trans_dir: Path):
    """Transfer 디버그 텍스트 저장"""
    try:
        trans_debug_path = trans_dir / "trans_debug.txt"
        trans_debug_content = _generate_transfer_debug_info(
            result=result,
            src_path=src_path,
            ref_path=ref_path
        )
        trans_debug_path.write_text(trans_debug_content, encoding='utf-8')
        print(f"   ✅ Transfer debug: {trans_debug_path.name}")
    except Exception as e:
        print(f"   ⚠️ Transfer Debug 생성 중 오류: {e}")


def _save_face_visualization(result, trans_dir: Path):
    """Face Transfer 시각화 저장"""
    try:
        if result.processing_info and 'face_transfer_debug' in result.processing_info:
            face_viz_image = generate_face_transfer_image(
                debug_info=result.processing_info['face_transfer_debug'],
                src_kpts=result.source_keypoints,
                src_scores=result.source_scores,
                ref_kpts=result.reference_keypoints,
                ref_scores=result.reference_scores,
                trans_kpts=result.transferred_keypoints,
                trans_scores=result.transferred_scores
            )
            if face_viz_image is not None:
                vis_path = trans_dir / "face_transfer_debug.jpg"
                save_image(face_viz_image, str(vis_path))
                print(f"   ✅ Face transfer visualization: {vis_path.name}")
    except Exception as e:
        print(f"   ⚠️ Face visualization 생성 중 오류: {e}")


def _generate_transfer_debug_info(result, src_path: Path, ref_path: Path) -> str:
    """Transfer 결과에 대한 디버그 정보 생성"""
    lines = []
    lines.append("=" * 80)
    lines.append("Pose Transfer Debug Information")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Source: {src_path.name}")
    lines.append(f"Reference: {ref_path.name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 이미지 크기 정보
    lines.append("=" * 80)
    lines.append("[1] Image Information")
    lines.append("=" * 80)
    lines.append(f"Final Canvas Size: {result.image_size[1]}x{result.image_size[0]} (W×H)")
    lines.append("")
    
    # 정렬 정보
    if result.alignment_info:
        align = result.alignment_info
        lines.append("=" * 80)
        lines.append("[2] Alignment Information")
        lines.append("=" * 80)
        lines.append(f"Alignment Method: {align.alignment_method}")
        lines.append(f"Transfer Lower Body: {align.should_transfer_lower}")
        lines.append(f"Align by Feet: {align.align_by_feet}")
        lines.append(f"Face Scale Ratio: {align.face_scale_ratio:.4f}")
        lines.append("")
    
    # 키포인트 통계
    lines.append("=" * 80)
    lines.append("[3] Keypoint Statistics")
    lines.append("=" * 80)
    
    trans_valid = np.sum(result.transferred_scores > 0)
    trans_total = len(result.transferred_scores)
    lines.append(f"Transferred: {trans_valid}/{trans_total} ({trans_valid/trans_total*100:.1f}%)")
    
    src_valid = np.sum(result.source_scores > 0)
    src_total = len(result.source_scores)
    lines.append(f"Source: {src_valid}/{src_total} ({src_valid/src_total*100:.1f}%)")
    
    ref_valid = np.sum(result.reference_scores > 0)
    ref_total = len(result.reference_scores)
    lines.append(f"Reference: {ref_valid}/{ref_total} ({ref_valid/ref_total*100:.1f}%)")
    lines.append("")

    # Transfer log 상세
    if result.processing_info and 'transfer_log' in result.processing_info:
        log = result.processing_info['transfer_log']
        
        # Depth info
        if 'depth' in log:
            depth_info = log['depth']
            lines.append("=" * 80)
            lines.append("[4] Depth Information")
            lines.append("=" * 80)
            lines.append(f"enabled: {depth_info.get('enabled', False)}")
            lines.append(f"z_scale: {depth_info.get('z_scale', 'N/A')}")
            lines.append(f"src_depth_min: {depth_info.get('src_depth_min', 'N/A')}")
            lines.append(f"src_depth_max: {depth_info.get('src_depth_max', 'N/A')}")
            lines.append(f"src_depth_mean: {depth_info.get('src_depth_mean', 'N/A')}")
            lines.append(f"ref_depth_min: {depth_info.get('ref_depth_min', 'N/A')}")
            lines.append(f"ref_depth_max: {depth_info.get('ref_depth_max', 'N/A')}")
            lines.append(f"ref_depth_mean: {depth_info.get('ref_depth_mean', 'N/A')}")
            lines.append("")

        # Bone lengths (Source 기준)
        if result.source_bone_lengths:
            lines.append("=" * 80)
            lines.append("[5] Source Bone Lengths (Reference for Transfer)")
            lines.append("=" * 80)
            for bone_name, length in sorted(result.source_bone_lengths.items()):
                lines.append(f"{bone_name:20s}: {length:8.2f}")
            lines.append("")

        # Transfer Processing Log
        lines.append("=" * 80)
        lines.append("[6] Transfer Processing Log")
        lines.append("=" * 80)
        if 'body_transferred' in log:
            lines.append(f"Body (17 keypoints): {log['body_transferred']}")
        if 'face_transferred' in log:
            lines.append(f"Face (68 keypoints): {log['face_transferred']}")
        if 'hand_transferred' in log:
            lines.append(f"Hands (42 keypoints): {log['hand_transferred']}")
        if 'foot_transferred' in log:
            lines.append(f"Feet (6 keypoints): {log['foot_transferred']}")
        lines.append("")

        # ============================================================
        # [7] Scale Arbitration Logic (NEW - engine.py v36)
        # ============================================================
        if 'scale_arbitration' in log:
            # scale_arbitration은 이미 포맷된 문자열
            lines.append(log['scale_arbitration'])
            lines.append("")

        # Hand Transfer Details
        if 'hand_debug' in log:
            lines.append("=" * 80)
            lines.append("[8] Hand Transfer Details")
            lines.append("=" * 80)
            lines.append(f"Hand Scale Ratio (ref→src): {log.get('hand_scale_ratio', 'N/A')}")
            for hand in log['hand_debug']:
                side = hand.get('side', 'Unknown')
                lines.append("")
                lines.append(f"[{side}]")
                lines.append(f"  trans_wrist_score: {hand.get('trans_wrist_score', 'N/A')}")
                if 'src_hand_count' in hand or 'ref_hand_count' in hand:
                    lines.append(f"  src_hand_count: {hand.get('src_hand_count', 'N/A')}/21")
                    lines.append(f"  ref_hand_count: {hand.get('ref_hand_count', 'N/A')}/21")
                lines.append(f"  status: {hand.get('status', 'N/A')}")
                if hand.get('status') == 'ok':
                    lines.append(f"  strategy: {hand.get('strategy', 'N/A')}")
                    lines.append(f"  scale: {hand.get('scale', 'N/A')}")
                    lines.append(f"  scale_source: {hand.get('scale_source', 'N/A')}")
                    lines.append(f"  transferred: {hand.get('transferred', 'N/A')}/21")
                else:
                    lines.append(f"  reason: {hand.get('reason', 'N/A')}")
            lines.append("")

        # Foot Transfer Details
        if 'foot_debug' in log:
            lines.append("=" * 80)
            lines.append("[9] Foot Transfer Details")
            lines.append("=" * 80)
            for foot in log['foot_debug']:
                side = foot.get('side', 'Unknown')
                lines.append("")
                lines.append(f"[{side}]")
                lines.append(f"  parent: {foot.get('parent', 'N/A')} ({foot.get('parent_idx', 'N/A')})")
                lines.append(f"  src_base: {foot.get('src_base', 'N/A')}")
                lines.append(f"  scale_factor: {foot.get('scale_factor', 'N/A')}")
            lines.append("")

        # Upper Ratio Tuning
        if 'upper_ratio_tuning' in log:
            lines.append("=" * 80)
            lines.append("[10] Upper Limb Ratio Tuning")
            lines.append("=" * 80)
            tuning = log['upper_ratio_tuning']
            lines.append(f"src_torso: {tuning.get('src_torso', 'N/A')}")
            lines.append(f"trans_torso: {tuning.get('trans_torso', 'N/A')}")
            lines.append(f"ratio_source: {tuning.get('ratio_source', 'N/A')}")
            lines.append("")

        # Lower Ratio Tuning
        if 'lower_ratio_tuning' in log:
            lines.append("=" * 80)
            lines.append("[11] Lower Limb Ratio Tuning")
            lines.append("=" * 80)
            tuning = log['lower_ratio_tuning']
            lines.append(f"src_torso: {tuning.get('src_torso', 'N/A')}")
            lines.append(f"trans_torso: {tuning.get('trans_torso', 'N/A')}")
            lines.append(f"ratio_source: {tuning.get('ratio_source', 'N/A')}")
            lines.append("")

    # Face Transfer Details
    if result.processing_info and 'face_transfer_debug' in result.processing_info:
        face_debug = result.processing_info['face_transfer_debug']
        lines.append("=" * 80)
        lines.append("[12] Face Transfer Details")
        lines.append("=" * 80)
        lines.append("")
        
        if 'src_neck_nose_y' in face_debug:
            lines.append(f"🔹 Source (Src):")
            lines.append(f"   Neck→Nose Y축 거리: {face_debug['src_neck_nose_y']:.1f}px")
            lines.append("")
        
        if 'ref_neck_nose_y' in face_debug:
            lines.append(f"🔹 Reference (Ref):")
            lines.append(f"   Neck→Nose Y축 거리: {face_debug['ref_neck_nose_y']:.1f}px")
            lines.append("")
        
        if 'trans_neck_nose_y' in face_debug:
            lines.append(f"🔹 Transfer (Trans):")
            lines.append(f"   Neck→Nose Y축 거리: {face_debug['trans_neck_nose_y']:.1f}px")
            lines.append("")
    
    lines.append("=" * 80)
    lines.append("[Note]")
    lines.append("=" * 80)
    lines.append("이 파일은 Pose Transfer 결과 분석을 위한 디버그 정보입니다.")
    lines.append("")
    
    return "\n".join(lines)


# ====================================================
# [Export] 외부 공개 API
# ====================================================
__all__ = [
    'execute_pose_transfer',
    'resolve_data_paths',
    'find_first_image',
    'OutputConfig',
    'PipelineConfig',
    'PoseTransferPipeline'
]
