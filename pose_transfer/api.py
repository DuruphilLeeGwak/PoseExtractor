"""
Pose Transfer API Module (v5.3 - Config Path Fixed)

위치: pose_transfer/api.py
변경사항:
- [Fix] default.yaml 구조(Root Level Keys)에 맞게 매핑 로직 수정
- [Fix] rendering 섹션의 point_radius, line_thickness가 무시되던 버그 해결
- [Fix] auto_crop_enabled를 output 섹션에서 읽어오도록 수정
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
            except Exception as e:
                print(f"   ❌ Error loading config: {e}")
        else:
            print(f"   ⚠️ Config file NOT found. Using defaults.")
        
        # [1] Output & Debug Config Mapping
        # default.yaml의 구조: root -> output -> debug
        out_conf = yaml_conf.get('output', {})
        dbg_conf = out_conf.get('debug', {}) # output 밑에 있는 debug
        
        # 만약 root에 debug가 따로 있다면 그것도 고려 (호환성)
        root_dbg = yaml_conf.get('debug', {})
        
        self.output_config = {
            'save_keypoints': out_conf.get('save_json', True),
            'save_skeleton': out_conf.get('save_skeleton_image', True),
            'save_debug_image': dbg_conf.get('save_bbox', True),
            'save_source_modified': dbg_conf.get('save_overlay', True),
            'save_report': dbg_conf.get('save_text', True),
            'save_face_debug': dbg_conf.get('save_face_viz', False)
        }
        
        # [2] Pipeline Config Deep Mapping (Root Level Keys)
        # default.yaml의 키들이 'pipeline' 안에 있지 않고 최상위에 있음
        
        # 기본 설정 딕셔너리 생성
        p_flat = {}
        
        # (A) Rendering Section (Root -> rendering)
        rend_conf = yaml_conf.get('rendering', {})
        p_flat['point_radius'] = rend_conf.get('point_radius', 4)
        p_flat['line_thickness'] = rend_conf.get('line_thickness', 4)
        p_flat['kpt_threshold'] = rend_conf.get('kpt_threshold', 0.1)
        
        # (B) Cross Filter Section (Root -> cross_filter)
        cf_conf = yaml_conf.get('cross_filter', {})
        p_flat['cross_filter_enabled'] = cf_conf.get('enabled', True)
        
        # (C) Preprocessing / Output Options
        # auto_crop_enabled는 output 섹션에 있음
        p_flat['auto_crop_enabled'] = out_conf.get('auto_crop_enabled', False)
        
        # (D) Hand Refinement (Root -> hand_refinement)
        hr_conf = yaml_conf.get('hand_refinement', {})
        p_flat['hand_refinement_enabled'] = hr_conf.get('enabled', True)
        p_flat['min_hand_size'] = hr_conf.get('min_hand_size', 48)
        
        # (E) Person Filter (Root -> person_filter)
        pf_conf = yaml_conf.get('person_filter', {})
        p_flat['filter_enabled'] = pf_conf.get('enabled', True)
        p_flat['area_weight'] = pf_conf.get('area_weight', 0.6)
        p_flat['center_weight'] = pf_conf.get('center_weight', 0.4)
        p_flat['filter_confidence_threshold'] = pf_conf.get('confidence_threshold', 0.3)

        # (F) Debug Visualization Sync
        p_flat['debug_bbox_visualization'] = self.output_config['save_debug_image']

        print(f"   🎨 Rendering Config: Radius={p_flat['point_radius']}, Thick={p_flat['line_thickness']}")
        print(f"   ✂️ Auto Crop: {p_flat['auto_crop_enabled']}")

        # Config 객체 생성
        self.pipeline_config = PipelineConfig.from_dict(p_flat)
        
        # [3] Transfer Config Mapping (Root -> transfer)
        t_raw = yaml_conf.get('transfer', {})
        self.transfer_config = TransferConfig.from_dict(t_raw)

    def execute(self, source_path, reference_path, output_dir, prefix="trans"):
        src_p = Path(source_path)
        ref_p = Path(reference_path)
        out_d = Path(output_dir)
        
        dir_trans = out_d / "trans"
        dir_src = out_d / "src"
        dir_ref = out_d / "ref"
        
        for d in [out_d, dir_trans, dir_src, dir_ref]:
            d.mkdir(parents=True, exist_ok=True)
            
        print(f"\n[API] Running Transfer: {src_p.name} -> {ref_p.name}")
        result = self.pipeline.transfer(src_p, ref_p)
        generated_files = {}

        # 1. SRC Output
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

        # 2. REF Output
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

        # 3. TRANS Output
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
            face_vis = generate_face_transfer_image(
                result.processing_info.get('transfer_log', {}),
                result.source_keypoints, result.source_scores,
                result.reference_keypoints, result.reference_scores,
                result.transferred_keypoints, result.transferred_scores
            )
            if face_vis is not None:
                save_image(face_vis, dir_trans / "face_debug.jpg")

        print(f"[API] Process Finished. Output saved to {out_d}")
        return generated_files

    def _save_debug_txt(self, path, content):
        if not content: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _create_report(self, result):
        lines = [f"Pose Transfer Report - {datetime.now()}"]
        lines.append("-" * 50)
        if result.alignment_info:
            ai = result.alignment_info
            lines.append(f"[Layout]")
            lines.append(f"  Strategy: {ai.anchor_type}")
            lines.append(f"  Scale   : {ai.global_scale:.3f}")
            lines.append(f"  Offset  : {ai.offset_vector.astype(int)}")
        
        lines.append("-" * 50)
        lines.append("[Transfer Log]")
        log = result.processing_info.get('transfer_log', {})
        for k, v in log.items():
            if k != 'face_transfer_debug' and isinstance(v, (int, float, str)):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)