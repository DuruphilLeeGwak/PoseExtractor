"""
Pose Transfer API Module (v4.2 - Attribute Name Fixed)

위치: pose_transfer/api.py
변경사항:
- [Fix] AlignManager의 변수명 불일치 수정 (alignment_method -> anchor_type)
- [Fix] 리포트 출력 내용 보강 (Offset, Anchor Point 추가)
"""
import sys
import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Union

# OpenMP 중복 로딩 에러 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .pipeline import PipelineConfig, PoseTransferPipeline
from .utils.io import save_json, save_image, convert_to_openpose_format
from .utils.face_transfer_visualizer import generate_face_transfer_image
from .transfer import TransferConfig

class PoseTransferAPI:
    def __init__(self, base_dir: str = None):
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path(__file__).parent.parent
            
        self.config_path = self.base_dir / "pose_transfer" / "config" / "default.yaml"
        self._load_config()
        
        print(f"🚀 Initializing Pose Transfer Pipeline...")
        self.pipeline = PoseTransferPipeline(self.pipeline_config, self.transfer_config)
        print("✅ Pipeline Ready.")

    def _load_config(self):
        """설정 파일 로드 및 검증"""
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
            print(f"   ⚠️ Config file NOT found at: {self.config_path}")
            print(f"   -> Using internal defaults.")
        
        self.output_config = yaml_conf.get('output', {})
        print(f"   📄 Output Settings: {self.output_config}") 

        p_conf = yaml_conf.get('pipeline', {})
        self.pipeline_config = PipelineConfig(
            backend=p_conf.get('backend', 'onnxruntime'),
            device=p_conf.get('device', 'cuda'),
            debug_bbox_visualization=p_conf.get('debug_bbox_visualization', False),
            cross_filter_enabled=p_conf.get('cross_filter', {}).get('enabled', True),
            depth_enabled=yaml_conf.get('depth', {}).get('enabled', False)
        )
        
        t_conf = yaml_conf.get('transfer', {})
        self.transfer_config = TransferConfig(
            confidence_threshold=t_conf.get('confidence_threshold', 0.3),
            use_face=t_conf.get('use_face', True),
            use_hands=t_conf.get('use_hands', True)
        )

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

        # =========================================================
        # 1. SRC Output
        # =========================================================
        try: shutil.copy2(src_p, dir_src / src_p.name)
        except: pass
        
        if self.output_config.get('save_skeleton', True):
            src_sk = self.pipeline.renderer.render_skeleton_only(
                (result.image_size[0], result.image_size[1], 3),
                result.source_keypoints, result.source_scores
            )
            save_image(src_sk, dir_src / "src_sk.jpg")
            
            # Overlay (로드 방식 수정됨: canvas_mgr 사용)
            src_ov = self.pipeline.renderer.render(
                self.pipeline.canvas_mgr.load_image_safe(src_p), 
                result.source_keypoints, result.source_scores
            )
            save_image(src_ov, dir_src / "src_rend.jpg")

        if self.output_config.get('save_debug_image', False) and result.src_debug_image is not None:
            save_image(result.src_debug_image, dir_src / "src_debug_bbox.jpg")

        if self.output_config.get('save_keypoints', True):
            src_json = convert_to_openpose_format(result.source_keypoints[None,...], result.source_scores[None,...], result.image_size)
            save_json(src_json, dir_src / "src_kp.json")

        # =========================================================
        # 2. REF Output
        # =========================================================
        try: shutil.copy2(ref_p, dir_ref / ref_p.name)
        except: pass
        
        if self.output_config.get('save_skeleton', True):
            ref_img_tmp = self.pipeline.canvas_mgr.load_image_safe(ref_p)
            ref_ov = self.pipeline.renderer.render(ref_img_tmp, result.reference_keypoints, result.reference_scores)
            save_image(ref_ov, dir_ref / "ref_rend.jpg")
            
            h, w = ref_img_tmp.shape[:2]
            ref_sk = self.pipeline.renderer.render_skeleton_only((h, w, 3), result.reference_keypoints, result.reference_scores)
            save_image(ref_sk, dir_ref / "ref_sk.jpg")

        if self.output_config.get('save_debug_image', False) and result.ref_debug_image is not None:
            save_image(result.ref_debug_image, dir_ref / "ref_debug_bbox.jpg")
            
        if self.output_config.get('save_keypoints', True):
            ref_json = convert_to_openpose_format(result.reference_keypoints[None,...], result.reference_scores[None,...], (1000, 1000))
            save_json(ref_json, dir_ref / "ref_kp.json")

        # =========================================================
        # 3. TRANS Output
        # =========================================================
        if self.output_config.get('save_skeleton', True):
            save_image(result.skeleton_image, dir_trans / f"{prefix}_sk.jpg")
            
            trans_ov = self.pipeline.renderer.render(
                result.modified_source_image, 
                result.transferred_keypoints, 
                result.transferred_scores
            )
            save_image(trans_ov, dir_trans / f"{prefix}_rend.jpg")
            
        if self.output_config.get('save_source_modified', True):
            save_image(result.modified_source_image, dir_trans / f"{prefix}_src_mod.jpg")
            
        if self.output_config.get('save_keypoints', True):
            save_json(result.to_json(), dir_trans / f"{prefix}_kp.json")

        if self.output_config.get('save_report', True):
            rpt = self._create_report(result)
            with open(dir_trans / "debug_report.txt", "w", encoding="utf-8") as f:
                f.write(rpt)
                
        if self.output_config.get('save_face_debug', False):
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

    def _create_report(self, result):
        lines = [f"Pose Transfer Report - {datetime.now()}"]
        lines.append("-" * 50)
        
        if result.alignment_info:
            ai = result.alignment_info
            lines.append(f"[Layout]")
            # [FIX] 변수명 수정: alignment_method -> anchor_type
            lines.append(f"  Strategy: {ai.anchor_type}")
            lines.append(f"  Scale   : {ai.global_scale:.3f}")
            # [Added] Offset 및 Anchor 정보 추가
            lines.append(f"  Offset  : {ai.offset_vector.astype(int)}")
            lines.append(f"  Anchor(S): {ai.anchor_point_src}")
            lines.append(f"  Anchor(R): {ai.anchor_point_ref}")
        
        lines.append("-" * 50)
        lines.append("[Transfer Log]")
        log = result.processing_info.get('transfer_log', {})
        
        if 'face_transfer_debug' in log:
            lines.append("  [Face Details]")
            for k, v in log['face_transfer_debug'].items():
                if isinstance(v, float): lines.append(f"    {k}: {v:.3f}")
                else: lines.append(f"    {k}: {v}")

        for k, v in log.items():
            if k != 'face_transfer_debug' and isinstance(v, (int, float, str)):
                lines.append(f"  {k}: {v}")
                
        return "\n".join(lines)