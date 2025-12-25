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

from .pipeline import PipelineConfig, PoseTransferPipeline
from .utils.io import save_json, save_image, load_image


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

    # 작업 ID 생성
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    job_id = f"{date_str}_{time_str}_{final_src_path.stem}_to_{final_ref_path.stem}"
    
    # 출력 디렉토리 생성
    out_dirs = _setup_directories(str(final_output_dir), job_id)

    print(f"\n🚀 [Start Job] {job_id}")

    try:
        pipeline = PoseTransferPipeline(pipeline_config, yaml_config=yaml_config)
        
        print("📊 Analyzing Inputs...")
        _save_analysis(pipeline, final_src_path, out_dirs["src"], "src", do_save_json, do_save_skel, do_save_debug)
        _save_analysis(pipeline, final_ref_path, out_dirs["ref"], "ref", do_save_json, do_save_skel, do_save_debug)
        
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


def _save_analysis(pipeline, image_path: Path, output_dir: Path, prefix: str, save_json_flag, save_skel_flag, save_debug_flag):
    json_data, skel_img, overlay_img = pipeline.extract_and_render(image_path)
    
    if save_json_flag:
        save_json(json_data, str(output_dir / f"{prefix}_kp.json"))
    if save_skel_flag:
        save_image(skel_img, str(output_dir / f"{prefix}_sk.jpg"))
    if save_debug_flag:
        save_image(overlay_img, str(output_dir / f"{prefix}_rend.jpg"))


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