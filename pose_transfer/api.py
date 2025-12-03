"""
Pose Transfer API Module
(핵심 로직이 모여있는 곳)
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
        print(f"ℹ️  [Input] Using YAML External Mode: {src}, {ref}")
        return Path(src), Path(ref)
    else:
        internal_cfg = input_cfg.get('internal', {})
        root = internal_cfg.get('root_dir', 'inputs')
        src_name = internal_cfg.get('source_name', 'source.jpg')
        ref_name = internal_cfg.get('reference_name', 'reference.jpg')
        
        # 프로젝트 루트 기준 경로 추정 (api.py -> pose_transfer -> root)
        project_root = Path(__file__).parent.parent
        base_path = project_root / root
        print(f"ℹ️  [Input] Using YAML Internal Mode: {base_path}")
        return base_path / src_name, base_path / ref_name

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
    
    out_dirs = _setup_directories(output_root, job_id)

    print(f"\n🚀 [Start Job] {job_id}")

    try:
        pipeline = PoseTransferPipeline(pipeline_config, yaml_config=yaml_config)
        
        print("📊 Analyzing Inputs...")
        _save_analysis(pipeline, src_p, out_dirs["source"], "source")
        _save_analysis(pipeline, ref_p, out_dirs["reference"], "reference")
        
        print("✨ Running Transfer...")
        result = pipeline.transfer(src_p, ref_p)
        
        res_paths = {}
        
        path_json = out_dirs["result"] / "transferred_keypoints.json"
        save_json(result.to_json(), str(path_json))
        res_paths['json'] = str(path_json)
        
        path_skel = out_dirs["result"] / "transferred_skeleton.png"
        save_image(result.skeleton_image, str(path_skel))
        res_paths['skeleton'] = str(path_skel)
        
        path_overlay = out_dirs["result"] / "transferred_overlay.png"
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

# 내부 헬퍼 함수들
def _setup_directories(output_root: str, job_id: str):
    base_dir = Path(output_root) / job_id
    dirs = {
        "root": base_dir,
        "source": base_dir / "01_source",
        "reference": base_dir / "02_reference",
        "result": base_dir / "03_result"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def _save_analysis(pipeline, image_path: Path, output_dir: Path, prefix: str):
    json_data, skel_img, overlay_img = pipeline.extract_and_render(image_path)
    save_json(json_data, str(output_dir / f"{prefix}_keypoints.json"))
    save_image(skel_img, str(output_dir / f"{prefix}_skeleton.png"))
    save_image(overlay_img, str(output_dir / f"{prefix}_overlay.png"))

def _cleanup_inputs(src_path: Path, ref_path: Path, enable_archiving: bool, archive_root: str = "archive"):
    if enable_archiving:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path(archive_root)
        (archive_dir / "source").mkdir(parents=True, exist_ok=True)
        (archive_dir / "reference").mkdir(parents=True, exist_ok=True)
        
        dest_src = archive_dir / "source" / f"{timestamp}_{src_path.name}"
        dest_ref = archive_dir / "reference" / f"{timestamp}_{ref_path.name}"
        
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