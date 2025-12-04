"""
Pose Transfer API Module (Updated)
- Naming Convention Changed: src, ref, trans / kp, sk, rend
- Output Format: JPG
"""
import sys
import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, Tuple

# 패키지 내부 임포트
from .pipeline import PipelineConfig, PoseTransferPipeline
from .utils.io import save_json, save_image, load_image

# ====================================================
# [Helper] 경로 결정 로직
# ====================================================
def resolve_input_paths(cli_args, yaml_config) -> Tuple[Path, Path]:
    if cli_args.source and cli_args.reference:
        print("ℹ️  [Input] Using CLI arguments.")
        return Path(cli_args.source), Path(cli_args.reference)

    input_cfg = yaml_config.get('input_mode', {})
    mode = input_cfg.get('type', 'internal')

    if mode == 'external':
        src = input_cfg.get('external', {}).get('source_path', '')
        ref = input_cfg.get('external', {}).get('reference_path', '')
        print(f"ℹ️  [Input] Using YAML External Mode.")
        return Path(src), Path(ref)
    else:
        # Internal Mode
        internal_cfg = input_cfg.get('internal', {})
        root = internal_cfg.get('root_dir', 'inputs')
        src_dir_name = internal_cfg.get('src_dir', 'src')
        ref_dir_name = internal_cfg.get('ref_dir', 'ref')
        
        project_root = Path(__file__).parent.parent
        inputs_root = project_root / root
        
        src_dir_path = inputs_root / src_dir_name
        ref_dir_path = inputs_root / ref_dir_name
        
        print(f"ℹ️  [Input] Internal Mode: Scanning folders...")

        def find_first_image(directory: Path, label: str) -> Path:
            if not directory.exists():
                raise FileNotFoundError(f"❌ '{label}' directory not found: {directory}")
            
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            
            if not files:
                raise FileNotFoundError(f"❌ No images found in '{label}' directory: {directory}")
            
            return files[0]

        src_p = find_first_image(src_dir_path, "src")
        ref_p = find_first_image(ref_dir_path, "ref")
        
        print(f"    👉 Source: {src_p.name}")
        print(f"    👉 Reference: {ref_p.name}")

        return src_p, ref_p

# ====================================================
# [API] 외부에서 호출하는 핵심 함수
# ====================================================
def execute_pose_transfer(
    source_path: Union[str, Path],
    reference_path: Union[str, Path],
    output_root: str = "outputs",
    config_path: str = "pose_transfer/config/default.yaml",
    explicit_config: Optional[dict] = None
) -> Dict[str, str]:
    
    src_p = Path(source_path)
    ref_p = Path(reference_path)
    
    if not src_p.exists():
        raise FileNotFoundError(f"Source file not found: {src_p}")
    if not ref_p.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_p}")

    yaml_config = explicit_config or {}
    if not yaml_config and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            
    if Path(config_path).exists():
        pipeline_config = PipelineConfig.from_yaml(str(config_path))
    else:
        pipeline_config = PipelineConfig()

    system_cfg = yaml_config.get('system', {})
    enable_archiving = system_cfg.get('enable_archiving', False)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    job_id = f"{date_str}_{time_str}_{src_p.stem}_to_{ref_p.stem}"
    
    # [수정] 디렉토리 설정 로직 호출
    out_dirs = _setup_directories(output_root, job_id)

    print(f"\n🚀 [Start Job] {job_id}")

    try:
        pipeline = PoseTransferPipeline(pipeline_config, yaml_config=yaml_config)
        
        print("📊 Analyzing Inputs...")
        # [수정] prefix를 'source' -> 'src', 'reference' -> 'ref'로 변경
        _save_analysis(pipeline, src_p, out_dirs["src"], "src")
        _save_analysis(pipeline, ref_p, out_dirs["ref"], "ref")
        
        print("✨ Running Transfer...")
        result = pipeline.transfer(src_p, ref_p)
        
        res_paths = {}
        
        # [수정] 파일명 변경: transferred -> trans, keypoints -> kp
        path_json = out_dirs["trans"] / "trans_kp.json"
        save_json(result.to_json(), str(path_json))
        res_paths['json'] = str(path_json)
        
        # [수정] 파일명 변경: skeleton -> sk, 포맷 -> jpg
        path_skel = out_dirs["trans"] / "trans_sk.jpg"
        save_image(result.skeleton_image, str(path_skel))
        res_paths['skeleton'] = str(path_skel)
        
        # [수정] 파일명 변경: overlay -> rend, 포맷 -> jpg
        path_overlay = out_dirs["trans"] / "trans_rend.jpg"
        src_img = load_image(src_p)
        overlay = pipeline.renderer.render(src_img, result.transferred_keypoints, result.transferred_scores)
        save_image(overlay, str(path_overlay))
        res_paths['overlay'] = str(path_overlay)
        
        res_paths['job_dir'] = str(out_dirs['root'])
        
        print(f"✅ Finished: {path_skel}")
        
        _cleanup_inputs(src_p, ref_p, enable_archiving)
        
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
    # [수정] 폴더명 변경: 01_source -> src, 02_reference -> ref, 03_result -> trans
    dirs = {
        "root": base_dir,
        "src": base_dir / "src",
        "ref": base_dir / "ref",
        "trans": base_dir / "trans"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def _save_analysis(pipeline, image_path: Path, output_dir: Path, prefix: str):
    json_data, skel_img, overlay_img = pipeline.extract_and_render(image_path)
    
    # [수정] 파일명 및 확장자 변경
    save_json(json_data, str(output_dir / f"{prefix}_kp.json"))
    save_image(skel_img, str(output_dir / f"{prefix}_sk.jpg"))   # png -> jpg
    save_image(overlay_img, str(output_dir / f"{prefix}_rend.jpg")) # png -> jpg

def _cleanup_inputs(src_path: Path, ref_path: Path, enable_archiving: bool, archive_root: str = "archive"):
    if enable_archiving:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path(archive_root)
        (archive_dir / "src").mkdir(parents=True, exist_ok=True) # source -> src
        (archive_dir / "ref").mkdir(parents=True, exist_ok=True) # reference -> ref
        
        dest_src = archive_dir / "src" / f"{timestamp}_{src_path.name}"
        dest_ref = archive_dir / "ref" / f"{timestamp}_{ref_path.name}"
        
        shutil.move(str(src_path), str(dest_src))
        shutil.move(str(ref_path), str(dest_ref))
        print(f"📦 Archived inputs to {archive_dir}")
    else:
        try:
            if src_path.exists(): os.remove(str(src_path))
            if ref_path.exists(): os.remove(str(ref_path))
            print("🗑️  Cleaned up input files (Volatile)")
        except Exception as e:
            print(f"⚠️ Failed to delete inputs: {e}")