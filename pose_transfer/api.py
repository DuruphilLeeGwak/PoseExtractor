"""
Pose Transfer API Module (v7.1 - Report Formatting Fix)

위치: pose_transfer/api.py
변경사항:
- [Fix] _create_report: 딕셔너리(Depth Stats 등)를 재귀적으로 출력하는 로직 추가
- [Fix] debug_report.txt에 누락되던 상세 정보 모두 기록
"""
import sys
import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Union, Any
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .pipeline import PipelineConfig, PoseTransferPipeline
from .utils.io import save_json, save_image, convert_to_openpose_format
from .utils.face_transfer_visualizer import generate_face_transfer_image
from .transfer import TransferConfig

class PoseTransferAPI:
    def __init__(self, base_dir: str = None):
        if base_dir: self.base_dir = Path(base_dir)
        else: self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / "pose_transfer" / "config" / "default.yaml"
        self._load_config()
        print(f"🚀 Initializing Pose Transfer Pipeline...")
        self.pipeline = PoseTransferPipeline(self.pipeline_config, self.transfer_config)
        print("✅ Pipeline Ready.")

    def _load_config(self):
        yaml_conf = {}
        print(f"\n🔍 [Config Check]")
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_conf = yaml.safe_load(f) or {}
                print(f"   ✅ Loaded: {self.config_path}")
            except: pass
        
        # Mappings (기존 동일)
        out_conf = yaml_conf.get('output', {})
        dbg_conf = out_conf.get('debug', {})
        self.output_config = {
            'save_keypoints': out_conf.get('save_json', True),
            'save_skeleton': out_conf.get('save_skeleton_image', True),
            'save_debug_image': dbg_conf.get('save_bbox', True),
            'save_source_modified': dbg_conf.get('save_overlay', True),
            'save_report': dbg_conf.get('save_text', True),
            'save_face_debug': dbg_conf.get('save_face_viz', False),
            'save_depth': dbg_conf.get('save_depth', False)
        }
        
        p_flat = {}
        rend_conf = yaml_conf.get('rendering', {})
        p_flat['point_radius'] = rend_conf.get('point_radius', 4)
        p_flat['line_thickness'] = rend_conf.get('line_thickness', 4)
        
        cf_conf = yaml_conf.get('cross_filter', {})
        p_flat['cross_filter_enabled'] = cf_conf.get('enabled', True)
        
        d_conf = yaml_conf.get('depth_anything', {})
        p_flat['depth_enabled'] = d_conf.get('enabled', False)
        p_flat['depth_model_type'] = d_conf.get('model', 'depth_anything_v2_vitl')
        
        p_flat['auto_crop_enabled'] = out_conf.get('auto_crop_enabled', False)
        p_flat['canvas_padding_ratio'] = out_conf.get('canvas_padding_ratio', 0.0)
        p_flat['debug_bbox_visualization'] = self.output_config['save_debug_image']
        
        print(f"   🎨 Rendering: Radius={p_flat['point_radius']}, Thick={p_flat['line_thickness']}")
        print(f"   🧭 Depth Enabled: {p_flat['depth_enabled']} (Model: {p_flat['depth_model_type']})")

        self.pipeline_config = PipelineConfig.from_dict(p_flat)
        self.transfer_config = TransferConfig.from_dict(yaml_conf.get('transfer', {}))

    def execute(self, source_path, reference_path, output_dir, prefix="trans"):
        # (기존 실행 로직 동일 - 생략)
        src_p = Path(source_path)
        ref_p = Path(reference_path)
        out_d = Path(output_dir)
        dir_trans = out_d / "trans"
        dir_src = out_d / "src"
        dir_ref = out_d / "ref"
        
        for d in [out_d, dir_trans, dir_src, dir_ref]: d.mkdir(parents=True, exist_ok=True)
            
        print(f"\n[API] Running Transfer: {src_p.name} -> {ref_p.name}")
        result = self.pipeline.transfer(src_p, ref_p)
        
        try: shutil.copy2(src_p, dir_src / src_p.name)
        except: pass
        if self.output_config['save_skeleton']:
            src_img_tmp = self.pipeline.canvas_mgr.load_image_safe(src_p)
            h, w = src_img_tmp.shape[:2]
            src_sk = self.pipeline.renderer.render_skeleton_only((h, w, 3), result.source_keypoints, result.source_scores)
            save_image(src_sk, dir_src / "src_sk.jpg")
            src_ov = self.pipeline.renderer.render(src_img_tmp, result.source_keypoints, result.source_scores)
            save_image(src_ov, dir_src / "src_rend.jpg")
        if self.output_config['save_debug_image'] and result.src_debug_image is not None:
            save_image(result.src_debug_image, dir_src / "src_debug_bbox.jpg")
        if self.output_config['save_keypoints']:
            src_json = convert_to_openpose_format(result.source_keypoints[None,...], result.source_scores[None,...], (h, w))
            save_json(src_json, dir_src / "src_kp.json")
            self._save_debug_txt(dir_src / "src_debug.txt", result.src_debug_text)
        if self.output_config['save_depth'] and result.src_depth_map is not None:
            save_image(result.src_depth_map, dir_src / "src_depth.jpg")

        try: shutil.copy2(ref_p, dir_ref / ref_p.name)
        except: pass
        if self.output_config['save_skeleton']:
            ref_img_tmp = self.pipeline.canvas_mgr.load_image_safe(ref_p)
            h, w = ref_img_tmp.shape[:2]
            ref_ov = self.pipeline.renderer.render(ref_img_tmp, result.reference_keypoints, result.reference_scores)
            save_image(ref_ov, dir_ref / "ref_rend.jpg")
            ref_sk = self.pipeline.renderer.render_skeleton_only((h, w, 3), result.reference_keypoints, result.reference_scores)
            save_image(ref_sk, dir_ref / "ref_sk.jpg")
        if self.output_config['save_debug_image'] and result.ref_debug_image is not None:
            save_image(result.ref_debug_image, dir_ref / "ref_debug_bbox.jpg")
        if self.output_config['save_keypoints']:
            ref_json = convert_to_openpose_format(result.reference_keypoints[None,...], result.reference_scores[None,...], (h, w))
            save_json(ref_json, dir_ref / "ref_kp.json")
            self._save_debug_txt(dir_ref / "ref_debug.txt", result.ref_debug_text)
        if self.output_config['save_depth'] and result.ref_depth_map is not None:
            save_image(result.ref_depth_map, dir_ref / "ref_depth.jpg")

        if self.output_config['save_skeleton']:
            save_image(result.skeleton_image, dir_trans / f"{prefix}_sk.jpg")
            trans_ov = self.pipeline.renderer.render(result.modified_source_image, result.transferred_keypoints, result.transferred_scores)
            save_image(trans_ov, dir_trans / f"{prefix}_rend.jpg")
        if self.output_config['save_source_modified']:
            save_image(result.modified_source_image, dir_trans / f"{prefix}_src_mod.jpg")
        if self.output_config['save_keypoints']:
            save_json(result.to_json(), dir_trans / f"{prefix}_kp.json")
        if self.output_config['save_report']:
            rpt = self._create_report(result)
            with open(dir_trans / "debug_report.txt", "w", encoding="utf-8") as f:
                f.write(rpt)
        if self.output_config['save_face_debug']:
            face_vis = generate_face_transfer_image(result.processing_info.get('transfer_log', {}), result.source_keypoints, result.source_scores, result.reference_keypoints, result.reference_scores, result.transferred_keypoints, result.transferred_scores)
            if face_vis is not None: save_image(face_vis, dir_trans / "face_debug.jpg")

        print(f"[API] Process Finished. Output saved to {out_d}")
        return {}

    def _save_debug_txt(self, path, content):
        if not content: return
        with open(path, "w", encoding="utf-8") as f: f.write(content)

    def _create_report(self, result):
        lines = [f"Pose Transfer Report - {datetime.now()}"]
        lines.append("-" * 50)
        
        if result.alignment_info:
            ai = result.alignment_info
            lines.append(f"[Layout]")
            lines.append(f"  Strategy: {ai.anchor_type}")
            lines.append(f"  Scale   : {ai.global_scale:.3f}")
            lines.append(f"  Offset  : {ai.offset_vector.astype(int)}")
            lines.append(f"  Anchor(S): {ai.anchor_point_src}")
        
        lines.append("\n" + "="*50)
        lines.append("[5] Transfer Processing Log")
        lines.append("="*50)
        
        log = result.processing_info.get('transfer_log', {})
        
        # [Fix] 재귀적 출력 함수
        def _print_dict(d, indent=0):
            res = []
            pad = "  " * indent
            for k, v in d.items():
                if isinstance(v, dict):
                    res.append(f"{pad}{k}:")
                    res.extend(_print_dict(v, indent+1))
                else:
                    res.append(f"{pad}{k}: {v}")
            return res

        lines.extend(_print_dict(log))
        return "\n".join(lines)