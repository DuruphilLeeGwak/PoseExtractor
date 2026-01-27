"""
Pose Transfer API Module (v2 - Pozibility 통합 지원)

실행 모드:
1. Standalone Mode: pose_extractor 단독 실행 → io/ 폴더 사용
2. Module Mode (외부 경로 주입): 외부에서 경로 직접 전달
3. Pozibility Mode: modules/pose_extractor로 통합 시 → data/ 폴더 사용
"""
import sys
import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, Tuple
import numpy as np

# OpenMP 중복 로딩 에러 방지 (onnxruntime/rtmlib 충돌 회피)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .pipeline import PipelineConfig, PoseTransferPipeline
from .utils.io import save_json, save_image, load_image
from .logic.debug_generator import generate_debug_info


# ====================================================
# [Path Resolution] 경로 결정 로직
# ====================================================
def resolve_data_paths() -> Tuple[Path, Path, Path]:
    """
    실행 환경에 따라 input/output 경로 결정
    
    Returns:
        (src_dir, ref_dir, output_dir)
    """
    # pose_transfer/ 폴더 위치
    pose_transfer_dir = Path(__file__).resolve().parent
    
    # pose_extractor/ 또는 모듈 루트
    module_root = pose_transfer_dir.parent
    
    # Pozibility 프로젝트 루트 (modules/pose_extractor의 상위의 상위)
    potential_pozibility_root = module_root.parent.parent
    pozibility_data = potential_pozibility_root / "data"
    
    # [Case 1] Pozibility 통합 모드: data/ 폴더가 존재하면 사용
    if pozibility_data.exists() and (pozibility_data / "inputs").exists():
        print("ℹ️  [API] Pozibility Mode: data/ 폴더를 사용합니다.")
        return (
            pozibility_data / "inputs" / "src",
            pozibility_data / "inputs" / "ref",
            pozibility_data / "preprocess_outputs"
        )
    
    # [Case 2] Standalone 모드: io/ 폴더 사용
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
    
    # [Case A] Module Mode: 외부에서 경로를 주입받음
    if source_path is not None and reference_path is not None:
        print("ℹ️  [API] Module Mode: 외부 경로를 사용합니다.")
        final_src_path = Path(source_path)
        final_ref_path = Path(reference_path)
        
        if output_root is not None:
            final_output_dir = Path(output_root)
        else:
            # 외부 경로 주입인데 output_root가 없으면 자동 결정
            _, _, auto_output = resolve_data_paths()
            final_output_dir = auto_output
    
    # [Case B] Auto Mode: 경로 자동 탐색
    else:
        src_dir, ref_dir, auto_output_dir = resolve_data_paths()
        
        # output_root 결정
        final_output_dir = Path(output_root) if output_root is not None else auto_output_dir
        
        # 이미지 파일 찾기
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

    # 4. 파이프라인 설정
    if config_p.exists():
        pipeline_config = PipelineConfig.from_yaml(str(config_p))
    else:
        pipeline_config = PipelineConfig()

    # 출력 옵션 로드
    output_cfg = yaml_config.get('output', {})
    do_save_json = output_cfg.get('save_json', True)
    do_save_skel = output_cfg.get('save_skeleton_image', True)
    do_save_debug = output_cfg.get('save_debug_image', False)
    do_save_debug_txt = output_cfg.get('save_debug_txt', True)  # 디버그 정보 생성 옵션

    # 작업 ID 생성
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    job_id = f"{date_str}_{time_str}_{final_src_path.stem}_to_{final_ref_path.stem}"
    
    # 출력 디렉토리 생성
    out_dirs = _setup_directories(str(final_output_dir), job_id)

    print(f"\n🚀 [Start Job] {job_id}")

    try:
        pipeline = PoseTransferPipeline(pipeline_config, yaml_config=yaml_config)
        
        # use_face 설정 가져오기 (face_rendering.enabled)
        face_rendering_config = yaml_config.get('face_rendering', {})
        use_face_setting = face_rendering_config.get('enabled', True)
        
        print("📊 Analyzing Inputs...")
        print(f"   ▶ SRC analyze start: {final_src_path.name}")
        _save_analysis(pipeline, final_src_path, out_dirs["src"], "src", do_save_json, do_save_skel, do_save_debug, use_face=use_face_setting)
        print(f"   ✅ SRC analyze done: {final_src_path.name}")
        print(f"   ▶ REF analyze start: {final_ref_path.name}")
        _save_analysis(pipeline, final_ref_path, out_dirs["ref"], "ref", do_save_json, do_save_skel, do_save_debug, use_face=use_face_setting)
        print(f"   ✅ REF analyze done: {final_ref_path.name}")
        
        # 디버그 정보 생성 (src와 ref에 대해)
        if do_save_debug_txt:
            print("📝 Generating Debug Info...")
            try:
                src_debug_path = generate_debug_info(
                    final_src_path, 
                    out_dirs["src"],
                    dw_extractor=pipeline.extractor,
                    body_extractor=None,  # 내부에서 생성
                    config=yaml_config
                )
                if src_debug_path:
                    print(f"   ✅ Source debug: {Path(src_debug_path).name}")
                
                ref_debug_path = generate_debug_info(
                    final_ref_path,
                    out_dirs["ref"],
                    dw_extractor=pipeline.extractor,
                    body_extractor=None,
                    config=yaml_config
                )
                if ref_debug_path:
                    print(f"   ✅ Reference debug: {Path(ref_debug_path).name}")
            except Exception as e:
                print(f"   ⚠️ Debug 생성 중 오류 (무시): {e}")
        
        print("✨ Running Transfer...")
        result = pipeline.transfer(final_src_path, final_ref_path)
        
        res_paths = {}
        
        # 1. 배경 저장
        path_bg = out_dirs["trans"] / "trans_bg.jpg"
        final_bg = result.modified_source_image if result.modified_source_image is not None else load_image(final_src_path)
        save_image(final_bg, str(path_bg))
        res_paths['background'] = str(path_bg)

        # 2. JSON 저장
        if do_save_json:
            path_json = out_dirs["trans"] / "trans_kp.json"
            save_json(result.to_json(), str(path_json))
            res_paths['json'] = str(path_json)
        
        # 3. Skeleton 저장
        if do_save_skel:
            path_skel = out_dirs["trans"] / "trans_sk.jpg"
            save_image(result.skeleton_image, str(path_skel))
            res_paths['skeleton'] = str(path_skel)
        
        # 4. Overlay 저장
        if do_save_debug:
            path_overlay = out_dirs["trans"] / "trans_rend.jpg"
            overlay = pipeline.renderer.render(final_bg, result.transferred_keypoints, result.transferred_scores)
            save_image(overlay, str(path_overlay))
            res_paths['overlay'] = str(path_overlay)
        
        # Debug BBox 저장
        if result.src_debug_image is not None:
            save_image(result.src_debug_image, str(out_dirs["src"] / "src_debug_bbox.jpg"))
        if result.ref_debug_image is not None:
            save_image(result.ref_debug_image, str(out_dirs["ref"] / "ref_debug_bbox.jpg"))

        # Depth visualization 저장 (optional)
        if result.processing_info and 'depth_maps' in result.processing_info:
            depth_maps = result.processing_info['depth_maps']
            if 'src' in depth_maps:
                _save_depth_visual(depth_maps['src'], out_dirs["src"] / f"{final_src_path.stem}_depth.png")
            if 'ref' in depth_maps:
                _save_depth_visual(depth_maps['ref'], out_dirs["ref"] / f"{final_ref_path.stem}_depth.png")
        elif getattr(pipeline, 'depth_extractor', None) is not None:
            # Fallback: generate depth maps directly if not provided
            try:
                src_img = load_image(final_src_path)
                ref_img = load_image(final_ref_path)
                src_depth_map = pipeline.depth_extractor.estimate(src_img)
                ref_depth_map = pipeline.depth_extractor.estimate(ref_img)
                _save_depth_visual(src_depth_map, out_dirs["src"] / f"{final_src_path.stem}_depth.png")
                _save_depth_visual(ref_depth_map, out_dirs["ref"] / f"{final_ref_path.stem}_depth.png")
                print("   ✅ Depth visuals saved (fallback)")
            except Exception as e:
                print(f"   ⚠️ Depth visualization fallback failed: {e}")

        # [v4.1] Transfer Debug 정보 생성
        if do_save_debug_txt:
            try:
                trans_debug_path = out_dirs["trans"] / "trans_debug.txt"
                trans_debug_content = _generate_transfer_debug_info(
                    result=result,
                    src_path=final_src_path,
                    ref_path=final_ref_path
                )
                trans_debug_path.write_text(trans_debug_content, encoding='utf-8')
                print(f"   ✅ Transfer debug: {trans_debug_path.name}")
                
                # Face Transfer 시각화 생성
                if result.processing_info and 'face_transfer_debug' in result.processing_info:
                    from .utils.face_transfer_visualizer import create_face_transfer_visualization
                    
                    vis_path = out_dirs["trans"] / "face_transfer_debug.jpg"
                    success = create_face_transfer_visualization(
                        debug_info=result.processing_info['face_transfer_debug'],
                        src_kpts=result.source_keypoints,
                        src_scores=result.source_scores,
                        ref_kpts=result.reference_keypoints,
                        ref_scores=result.reference_scores,
                        trans_kpts=result.transferred_keypoints,
                        trans_scores=result.transferred_scores,
                        output_path=vis_path
                    )
                    if success:
                        print(f"   ✅ Face transfer visualization: {vis_path.name}")
            except Exception as e:
                print(f"   ⚠️ Transfer Debug 생성 중 오류 (무시): {e}")

        res_paths['job_dir'] = str(out_dirs['root'])
        
        print(f"✅ Finished Job: {out_dirs['root']}")
        return res_paths

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Pose transfer failed: {e}")


# ====================================================
# [Internal] 내부 헬퍼 함수들
# ====================================================
def _setup_directories(output_root: str, job_id: str):
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


def _save_analysis(pipeline, image_path: Path, output_dir: Path, prefix: str, save_json_flag, save_skel_flag, save_debug_flag, use_face=None):
    """포즈 추출 및 저장
    
    Args:
        use_face: Face landmarks 표시 여부 (None이면 config 설정 따름)
    """
    json_data, skel_img, overlay_img = pipeline.extract_and_render(image_path, use_face=use_face)
    
    if save_json_flag:
        save_json(json_data, str(output_dir / f"{prefix}_kp.json"))
    if save_skel_flag:
        save_image(skel_img, str(output_dir / f"{prefix}_sk.jpg"))
    if save_debug_flag:
        save_image(overlay_img, str(output_dir / f"{prefix}_rend.jpg"))


def _save_depth_visual(depth_map: np.ndarray, output_path: Path) -> None:
    depth = depth_map.astype(np.float32)
    d_min = float(np.percentile(depth, 1))
    d_max = float(np.percentile(depth, 99))
    depth = (depth - d_min) / (d_max - d_min + 1e-6)
    depth = np.clip(depth, 0.0, 1.0)
    depth_u8 = (depth * 255).astype(np.uint8)
    save_image(depth_u8, str(output_path))


def _generate_transfer_debug_info(result, src_path: Path, ref_path: Path) -> str:
    """
    Transfer 결과에 대한 디버그 정보 생성
    
    Args:
        result: PipelineResult 객체
        src_path: Source 이미지 경로
        ref_path: Reference 이미지 경로
    
    Returns:
        디버그 정보 문자열
    """
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
    
    import numpy as np
    
    # Transferred keypoints
    trans_valid = np.sum(result.transferred_scores > 0)
    trans_total = len(result.transferred_scores)
    lines.append(f"Transferred: {trans_valid}/{trans_total} ({trans_valid/trans_total*100:.1f}%)")
    
    # Source keypoints
    src_valid = np.sum(result.source_scores > 0)
    src_total = len(result.source_scores)
    lines.append(f"Source: {src_valid}/{src_total} ({src_valid/src_total*100:.1f}%)")
    
    # Reference keypoints
    ref_valid = np.sum(result.reference_scores > 0)
    ref_total = len(result.reference_scores)
    lines.append(f"Reference: {ref_valid}/{ref_total} ({ref_valid/ref_total*100:.1f}%)")
    lines.append("")

    # Depth info
    if result.processing_info and 'transfer_log' in result.processing_info:
        log = result.processing_info['transfer_log']
        if 'depth' in log:
            depth_info = log['depth']
            lines.append("=" * 80)
            lines.append("[3.5] Depth Information")
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
        lines.append("[4] Source Bone Lengths (Reference for Transfer)")
        lines.append("=" * 80)
        for bone_name, length in sorted(result.source_bone_lengths.items()):
            lines.append(f"{bone_name:20s}: {length:8.2f}")
        lines.append("")
    
    # Transfer log
    if result.processing_info and 'transfer_log' in result.processing_info:
        log = result.processing_info['transfer_log']
        lines.append("=" * 80)
        lines.append("[5] Transfer Processing Log")
        lines.append("=" * 80)
        
        # 전이된 부위별 통계
        if 'body_transferred' in log:
            lines.append(f"Body (17 keypoints): {log['body_transferred']}")
        if 'face_transferred' in log:
            lines.append(f"Face (68 keypoints): {log['face_transferred']}")
        if 'hand_transferred' in log:
            lines.append(f"Hands (42 keypoints): {log['hand_transferred']}")
        if 'foot_transferred' in log:
            lines.append(f"Feet (6 keypoints): {log['foot_transferred']}")
    
    # Hand Transfer Debug Info
    if result.processing_info and 'transfer_log' in result.processing_info:
        log = result.processing_info['transfer_log']
        if 'hand_debug' in log:
            lines.append("")
            lines.append("=" * 80)
            lines.append("[6] Hand Transfer Details")
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
                    if 'src_hand_base' in hand:
                        lines.append(f"  src_hand_base: {hand.get('src_hand_base', 'N/A')}")
                    if 'ref_torso_len' in hand:
                        lines.append(f"  ref_torso_len: {hand.get('ref_torso_len', 'N/A')}")
                    if 'ref_hand_ratio' in hand:
                        lines.append(f"  ref_hand_ratio: {hand.get('ref_hand_ratio', 'N/A')}")
                    if 'ref_hand_ratio_max' in hand:
                        lines.append(f"  ref_hand_ratio_max: {hand.get('ref_hand_ratio_max', 'N/A')}")
                    if 'target_hand_size' in hand:
                        lines.append(f"  target_hand_size: {hand.get('target_hand_size', 'N/A')}")
                    if 'src_hand_mean_dist' in hand and 'ref_hand_mean_dist' in hand:
                        lines.append(f"  src_hand_mean_dist: {hand.get('src_hand_mean_dist', 'N/A')}")
                        lines.append(f"  ref_hand_mean_dist: {hand.get('ref_hand_mean_dist', 'N/A')}")
                        lines.append(f"  pairwise_count: {hand.get('pairwise_count', 'N/A')}")
                else:
                    lines.append(f"  reason: {hand.get('reason', 'N/A')}")

    # Foot Transfer Debug Info
    if result.processing_info and 'transfer_log' in result.processing_info:
        log = result.processing_info['transfer_log']
        if 'foot_debug' in log:
            lines.append("")
            lines.append("=" * 80)
            lines.append("[7] Foot Transfer Details")
            lines.append("=" * 80)
            for foot in log['foot_debug']:
                side = foot.get('side', 'Unknown')
                lines.append("")
                lines.append(f"[{side}]")
                lines.append(f"  parent: {foot.get('parent', 'N/A')} ({foot.get('parent_idx', 'N/A')})")
                lines.append(f"  src_base: {foot.get('src_base', 'N/A')}")
                lines.append(f"  ref_torso_len: {foot.get('ref_torso_len', 'N/A')}")
                lines.append(f"  ref_ratio: {foot.get('ref_ratio', 'N/A')}")
                lines.append(f"  ref_ratio_max: {foot.get('ref_ratio_max', 'N/A')}")
                lines.append(f"  target_size: {foot.get('target_size', 'N/A')}")
                lines.append(f"  scale_factor: {foot.get('scale_factor', 'N/A')}")
                lines.append(f"  ref_mean_used: {foot.get('ref_mean_used', 'N/A')}")
                lines.append(f"  src_left_mean: {foot.get('src_left_mean', 'N/A')}")
                lines.append(f"  src_right_mean: {foot.get('src_right_mean', 'N/A')}")
                lines.append(f"  ref_left_mean: {foot.get('ref_left_mean', 'N/A')}")
                lines.append(f"  ref_right_mean: {foot.get('ref_right_mean', 'N/A')}")
                for child in foot.get('children', []):
                    lines.append(f"  - {child.get('name', 'N/A')} (idx={child.get('idx', 'N/A')}): length={child.get('length', 'N/A')} source={child.get('source', 'N/A')} ref_vec={child.get('ref_vec', 'N/A')} ref_len={child.get('ref_len', 'N/A')}")

        if 'upper_ratio_tuning' in log:
            lines.append("")
            lines.append("=" * 80)
            lines.append("[8] Upper Limb Ratio Tuning")
            lines.append("=" * 80)
            tuning = log['upper_ratio_tuning']
            lines.append(f"src_torso: {tuning.get('src_torso', 'N/A')}")
            lines.append(f"trans_torso: {tuning.get('trans_torso', 'N/A')}")
            ratios_2d = tuning.get('ratios_2d', {})
            ratios_3d = tuning.get('ratios_3d', {})
            ratios_used = tuning.get('ratios_used', {})
            deltas = tuning.get('deltas_vs_2d', {})
            lines.append(f"ratio_source: {tuning.get('ratio_source', 'N/A')}")
            for k in sorted(set(list(ratios_used.keys()) + list(ratios_2d.keys()) + list(ratios_3d.keys()))):
                lines.append(
                    f"  {k}: 2d={ratios_2d.get(k, 'N/A')} 3d={ratios_3d.get(k, 'N/A')} used={ratios_used.get(k, 'N/A')} delta={deltas.get(k, 'N/A')}"
                )

        if 'lower_ratio_tuning' in log:
            lines.append("")
            lines.append("=" * 80)
            lines.append("[9] Lower Limb Ratio Tuning")
            lines.append("=" * 80)
            tuning = log['lower_ratio_tuning']
            lines.append(f"src_torso: {tuning.get('src_torso', 'N/A')}")
            lines.append(f"trans_torso: {tuning.get('trans_torso', 'N/A')}")
            ratios_2d = tuning.get('ratios_2d', {})
            ratios_3d = tuning.get('ratios_3d', {})
            ratios_used = tuning.get('ratios_used', {})
            deltas = tuning.get('deltas_vs_2d', {})
            lines.append(f"ratio_source: {tuning.get('ratio_source', 'N/A')}")
            for k in sorted(set(list(ratios_used.keys()) + list(ratios_2d.keys()) + list(ratios_3d.keys()))):
                lines.append(
                    f"  {k}: 2d={ratios_2d.get(k, 'N/A')} 3d={ratios_3d.get(k, 'N/A')} used={ratios_used.get(k, 'N/A')} delta={deltas.get(k, 'N/A')}"
                )

    # Face Transfer Debug Info
    if result.processing_info and 'face_transfer_debug' in result.processing_info:
        face_debug = result.processing_info['face_transfer_debug']
        lines.append("")
        lines.append("=" * 80)
        lines.append("[10] Face Transfer Details")
        lines.append("=" * 80)
        lines.append("")
        lines.append("📐 Distance Definitions:")
        lines.append("   Y축 거리: 어깨 중심점(Neck)에서 각 키포인트까지의 수직(위→아래) 거리")
        lines.append("   X축 거리: 각 키포인트 간의 수평(좌→우) 거리")
        lines.append("   - 어깨 중심점(Neck): (Left Shoulder + Right Shoulder) / 2")
        lines.append("   - 수직거리 측정: |키포인트_y - Neck_y|")
        lines.append("")
        
        if 'src_neck_nose_y' in face_debug:
            lines.append(f"🔹 Source (Src):")
            lines.append(f"   Neck→Nose Y축 거리: {face_debug['src_neck_nose_y']:.1f}px")
            lines.append(f"   └─ 어깨 중심에서 코까지의 수직 거리 (Src의 목 길이)")
            lines.append("")
        
        if 'ref_neck_nose_y' in face_debug and 'ref_neck_nose_x_ratio' in face_debug:
            lines.append(f"🔹 Reference (Ref):")
            lines.append(f"   Neck→Nose Y축 거리: {face_debug['ref_neck_nose_y']:.1f}px")
            lines.append(f"   Neck→Nose X축 비율: {face_debug['ref_neck_nose_x_ratio']:.3f}")
            lines.append(f"   └─ Ref의 머리 각도 정보 (좌우 기울기)")
            lines.append("")
        
        if 'trans_neck_nose_y' in face_debug and 'trans_neck_nose_x' in face_debug:
            lines.append(f"🔹 Transfer (Trans):")
            lines.append(f"   Neck→Nose Y축 거리: {face_debug['trans_neck_nose_y']:.1f}px")
            lines.append(f"   └─ Src 비율 유지 ({face_debug.get('src_neck_nose_y', 0):.1f}px × {face_debug.get('global_scale', 1.0):.3f} = {face_debug['trans_neck_nose_y']:.1f}px)")
            lines.append(f"   Neck→Nose X축 거리: {face_debug['trans_neck_nose_x']:.1f}px")
            lines.append(f"   └─ Ref 각도 반영 (좌우 회전)")
            lines.append("")
            lines.append(f"   ✅ 어깨→코 수직거리 유지: {face_debug.get('src_neck_nose_y', 0):.1f} → {face_debug['trans_neck_nose_y']:.1f}px")
            lines.append("")
        
        if 'left_eye_y' in face_debug and 'left_eye_x' in face_debug:
            lines.append(f"🔹 Left Eye (왼쪽 눈):")
            lines.append(f"   Nose→Left Eye Y축: {face_debug['left_eye_y']:.1f}px")
            lines.append(f"   Nose→Left Eye X축: {face_debug['left_eye_x']:.1f}px")
            lines.append(f"   └─ Ref 얼굴 구조 적용 (Ref 비율 × {face_debug.get('global_scale', 1.0):.3f})")
            lines.append("")
        
        if 'right_eye_y' in face_debug and 'right_eye_x' in face_debug:
            lines.append(f"🔹 Right Eye (오른쪽 눈):")
            lines.append(f"   Nose→Right Eye Y축: {face_debug['right_eye_y']:.1f}px")
            lines.append(f"   Nose→Right Eye X축: {face_debug['right_eye_x']:.1f}px")
            lines.append(f"   └─ Ref 얼굴 구조 적용 (Ref 비율 × {face_debug.get('global_scale', 1.0):.3f})")
            lines.append("")
        
        lines.append("📊 Summary:")
        lines.append("   - 어깨→코 수직거리: Src 신체 비율 유지")
        lines.append("   - 코→눈 거리: Ref 얼굴 구조 적용")
        lines.append("   - 좌우 회전: Ref 머리 각도 반영")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("[Note]")
    lines.append("=" * 80)
    lines.append("이 파일은 Pose Transfer 결과 분석을 위한 디버그 정보입니다.")
    lines.append("추가 정보가 필요한 경우 내용을 확장할 수 있습니다.")
    lines.append("")
    
    return "\n".join(lines)


# ====================================================
# [Export] 외부 공개 API
# ====================================================
__all__ = [
    'execute_pose_transfer',
    'resolve_data_paths',
    'find_first_image',
    'PipelineConfig',
    'PoseTransferPipeline'
]