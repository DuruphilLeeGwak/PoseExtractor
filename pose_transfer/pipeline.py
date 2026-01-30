"""
Pose Transfer Pipeline (Refactored v5.1 - Report Statistics Restored)

위치: pose_transfer/pipeline.py
변경사항:
- [Fix] Cross-Filter 리포트에 통계(평균, 최소/최대, 의심 개수) 섹션 추가
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass

from .extractors import (
    DWPoseExtractorFactory, PersonFilter, RTMLIB_AVAILABLE, DepthAnythingV2Extractor
)
from .extractors.body_extractor import BodyExtractor

from .transfer import PoseTransferEngine, TransferConfig
from .refiners import HandRefiner
from .renderers import SkeletonRenderer
from .utils import load_config, convert_to_openpose_format, load_image

# Logic Modules
from .logic.bbox_manager import BboxManager
from .logic.align_manager import AlignManager
from .logic.canvas_manager import CanvasManager
from .logic.cross_filter import CrossFilter, CrossFilterConfig

# (PipelineConfig, PipelineResult 클래스는 v5.0과 동일하므로 유지 - 생략 가능하지만 안전을 위해 전체 제공)
@dataclass
class PipelineConfig:
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    filter_enabled: bool = True
    hand_refinement_enabled: bool = True
    cross_filter_enabled: bool = True
    depth_enabled: bool = False
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    min_hand_size: int = 48
    line_thickness: int = 4
    point_radius: int = 4
    kpt_threshold: float = 0.3
    auto_crop_enabled: bool = False
    yolo_verification_enabled: bool = True
    person_bbox_margin: float = 0.0
    face_bbox_margin: float = 0.0
    debug_bbox_visualization: bool = False
    depth_model_type: str = 'depth_anything_v2_vitl'
    depth_z_scale: float = 1000.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data: return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

@dataclass
class PipelineResult:
    transferred_keypoints: np.ndarray
    transferred_scores: np.ndarray
    source_keypoints: np.ndarray
    source_scores: np.ndarray
    reference_keypoints: np.ndarray
    reference_scores: np.ndarray
    skeleton_image: np.ndarray
    image_size: Tuple[int, int]
    modified_source_image: Optional[np.ndarray] = None
    processing_info: Dict[str, Any] = None
    alignment_info: Any = None
    src_debug_image: Optional[np.ndarray] = None
    ref_debug_image: Optional[np.ndarray] = None
    src_debug_text: str = ""
    ref_debug_text: str = ""
    
    def to_json(self) -> Dict[str, Any]:
        return convert_to_openpose_format(
            self.transferred_keypoints[np.newaxis, ...],
            self.transferred_scores[np.newaxis, ...],
            self.image_size
        )

class PoseTransferPipeline:
    def __init__(self, config: PipelineConfig, transfer_config: TransferConfig = None):
        self.config = config
        self.transfer_config = transfer_config or TransferConfig()
        
        self.bbox_mgr = BboxManager(self.config)
        self.align_mgr = AlignManager(self.config)
        self.canvas_mgr = CanvasManager(self.config)
        
        self._init_modules()
    
    def _init_modules(self):
        if not RTMLIB_AVAILABLE: raise RuntimeError("rtmlib not installed.")
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend, device=self.config.device, mode=self.config.mode
        )
        self.body_extractor = None
        self.cross_filter = None
        if self.config.cross_filter_enabled:
            print("🛡️ Cross Filter Enabled")
            self.body_extractor = BodyExtractor(backend='onnxruntime', device='cpu')
            self.cross_filter = CrossFilter(CrossFilterConfig()) 
        self.depth_extractor = None
        if self.config.depth_enabled:
            try:
                self.depth_extractor = DepthAnythingV2Extractor(model=self.config.depth_model_type, device=self.config.device)
            except Exception: pass
        self.person_filter = PersonFilter(self.config.area_weight, self.config.center_weight, self.config.filter_confidence_threshold)
        self.transfer_engine = PoseTransferEngine(config=self.transfer_config)
        self.hand_refiner = HandRefiner(self.config.min_hand_size)
        self.renderer = SkeletonRenderer(line_thickness=self.config.line_thickness, point_radius=self.config.point_radius)

    def extract_pose(self, image, tag="Image"):
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        h, w = img.shape[:2]
        
        all_kpts, all_scores = self.extractor.extract(img)
        if len(all_kpts) == 0:
            return np.zeros((133, 2)), np.zeros(133), -1, (h, w), "No Person Detected"
            
        kpts, scores, idx = all_kpts[0], all_scores[0], 0
        if len(all_kpts) > 1 and self.config.filter_enabled:
            kpts, scores, idx, _ = self.person_filter.select_main_person(all_kpts, all_scores, (h, w))
            
        debug_report = f"================================================================================\n"
        debug_report += f"Cross-Filter 디버깅 정보 [{tag}]\n"
        debug_report += f"================================================================================\n"
        debug_report += f"📐 이미지 크기: {w}x{h}\n\n"
        
        if self.config.cross_filter_enabled and self.body_extractor:
            body_kpts_all, body_scores_all = self.body_extractor.extract(img)
            if len(body_kpts_all) > 0:
                body_kpts, body_scores = body_kpts_all[0], body_scores_all[0]
                debug_report += self._generate_cross_filter_report(body_kpts, body_scores, kpts, scores)
                kpts, scores, approved = self.cross_filter.filter(
                    body_keypoints=body_kpts, body_scores=body_scores,
                    dw_keypoints=kpts, dw_scores=scores
                )
            else:
                debug_report += "⚠️ Body Model detected NO person.\n"
        
        if self.config.hand_refinement_enabled:
            kpts, scores, _ = self.hand_refiner.refine_both_hands(img, kpts, scores, self.extractor)
            
        return kpts, scores, idx, (h, w), debug_report

    def _generate_cross_filter_report(self, body_kpts, body_scores, dw_kpts, dw_scores):
        report = []
        report.append("================================================================================")
        report.append("[1] Body vs DWPose Body 17 Keypoints 비교")
        report.append("================================================================================")
        report.append(f"{'No':<4} {'Name':<18} {'Body Conf':<12} {'DWPose Conf':<12} {'상태'}")
        report.append("-" * 80)
        
        coco_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        b_scores_list = []
        d_scores_list = []
        suspect_count = 0
        
        for i, name in enumerate(coco_names):
            b_conf = body_scores[i] if i < len(body_scores) else 0.0
            d_conf = dw_scores[i]
            
            b_scores_list.append(b_conf)
            d_scores_list.append(d_conf)
            
            status = ""
            if b_conf > 0.5 and d_conf > 0.5: status = "✅ 양쪽 높음" # Threshold 조정됨
            elif b_conf < 0.3 and d_conf > 0.6: 
                status = "⚠️ Body만 낮음 (할루시네이션 의심)"
                suspect_count += 1
            elif b_conf > 0.6 and d_conf < 0.3: status = "⚠️ DW만 낮음"
            else: status = "-"
            
            report.append(f"{i:<4} {name:<18} {b_conf:<12.3f} {d_conf:<12.3f} {status}")
            
        # [Fix] 통계 섹션 추가
        report.append("\n================================================================================")
        report.append("[2] 통계 요약")
        report.append("================================================================================")
        report.append(f"📊 Body Model 평균 점수: {np.mean(b_scores_list):.4f} (Max: {np.max(b_scores_list):.4f}, Min: {np.min(b_scores_list):.4f})")
        report.append(f"📊 DWPose 평균 점수   : {np.mean(d_scores_list):.4f} (Max: {np.max(d_scores_list):.4f}, Min: {np.min(d_scores_list):.4f})")
        report.append(f"🚨 할루시네이션 의심 포인트: {suspect_count}개")
        report.append("================================================================================\n")
        
        return "\n".join(report)

    def transfer(self, source_image, reference_image):
        if isinstance(source_image, (str, Path)): src_img = load_image(source_image)
        else: src_img = source_image
        if isinstance(reference_image, (str, Path)): ref_img = load_image(reference_image)
        else: ref_img = reference_image
        
        src_kpts, src_scores, src_idx, src_size, src_rpt = self.extract_pose(src_img, "SRC")
        ref_kpts, ref_scores, ref_idx, ref_size, ref_rpt = self.extract_pose(ref_img, "REF")
        
        src_bbox, src_face, src_dbg = self.bbox_mgr.get_bboxes(src_img, src_kpts, src_scores)
        ref_bbox, ref_face, ref_dbg = self.bbox_mgr.get_bboxes(ref_img, ref_kpts, ref_scores)
        
        layout = self.align_mgr.analyze_layout(src_bbox, ref_bbox, src_kpts, src_scores, ref_kpts, ref_scores)
        
        result = self.transfer_engine.transfer(
            src_kpts, src_scores, ref_kpts, ref_scores,
            src_size, ref_size,
            layout=layout
        )
        
        trans_kpts, trans_scores = result.keypoints, result.scores
        final_img, final_kpts, final_size = self.canvas_mgr.expand_canvas_to_fit(src_img, trans_kpts, trans_scores, padding_ratio=0.1)
        skeleton_img = self.renderer.render_skeleton_only((final_size[0], final_size[1], 3), final_kpts, trans_scores)
        
        src_debug_img = None
        ref_debug_img = None
        # [Fix] debug_bbox_visualization가 켜져야만 그림
        if self.config.debug_bbox_visualization:
            src_debug_img = self.bbox_mgr.draw_debug(src_img.copy(), src_dbg)
            ref_debug_img = self.bbox_mgr.draw_debug(ref_img.copy(), ref_dbg)

        return PipelineResult(
            transferred_keypoints=final_kpts,
            transferred_scores=trans_scores,
            source_keypoints=src_kpts, source_scores=src_scores,
            reference_keypoints=ref_kpts, reference_scores=ref_scores,
            skeleton_image=skeleton_img,
            image_size=final_size,
            modified_source_image=final_img,
            processing_info={'transfer_log': result.transfer_log},
            alignment_info=layout,
            src_debug_image=src_debug_img,
            ref_debug_image=ref_debug_img,
            src_debug_text=src_rpt,
            ref_debug_text=ref_rpt
        )