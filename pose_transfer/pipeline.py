"""
Pose Transfer Pipeline (Refactored v33 - CrossFilter & Canvas Fix)

위치: pose_transfer/pipeline.py
변경사항:
- [Fix] extract_pose 내에서 CrossFilter 로직 활성화 (Body Extractor 사용)
- [Fix] transfer 완료 후 CanvasManager를 통해 이미지 리사이징 수행
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from .extractors import (
    DWPoseExtractorFactory, PersonFilter, RTMLIB_AVAILABLE, DepthAnythingV2Extractor
)
# [FIX] BodyExtractor import 추가 (CrossFilter용)
from .extractors.body_extractor import BodyExtractor

from .transfer import PoseTransferEngine, TransferConfig
from .refiners import HandRefiner
from .renderers import SkeletonRenderer
from .utils import load_config, convert_to_openpose_format, load_image

# Logic Modules
from .logic.bbox_manager import BboxManager, BboxInfo
from .logic.align_manager import AlignManager, TransferLayout
from .logic.canvas_manager import CanvasManager
from .logic.cross_filter import CrossFilter, CrossFilterConfig

@dataclass
class PipelineConfig:
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    filter_enabled: bool = True
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    hand_refinement_enabled: bool = True
    min_hand_size: int = 48
    
    # Rendering
    line_thickness: int = 4
    point_radius: int = 4
    kpt_threshold: float = 0.3
    
    # Output
    auto_crop_enabled: bool = True
    
    # Alignment
    yolo_verification_enabled: bool = True
    person_bbox_margin: float = 0.0
    face_bbox_margin: float = 0.0
    
    # Cross Filter
    cross_filter_enabled: bool = True
    # (필요한 파라미터들...)
    
    # Depth
    depth_enabled: bool = False
    depth_model_type: str = 'depth_anything_v2_vitl'
    depth_z_scale: float = 1000.0
    
    # Debug
    debug_bbox_visualization: bool = False

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
        
        # Managers
        self.bbox_mgr = BboxManager(self.config)
        self.align_mgr = AlignManager(self.config)
        self.canvas_mgr = CanvasManager(self.config)
        
        self._init_modules()
    
    def _init_modules(self):
        if not RTMLIB_AVAILABLE: raise RuntimeError("rtmlib not installed.")
        
        # 1. Main Extractor (DWPose)
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend, device=self.config.device, mode=self.config.mode
        )
        
        # 2. Cross Filter (Body Extractor + Filter Logic)
        self.body_extractor = None
        self.cross_filter = None
        if self.config.cross_filter_enabled:
            print("🛡️ Cross Filter Enabled: Initializing Body Extractor...")
            self.body_extractor = BodyExtractor(backend='onnxruntime', device='cpu') # CPU for safety
            self.cross_filter = CrossFilter(CrossFilterConfig()) # Default config or pass self.config
            
        # 3. Depth (Optional)
        self.depth_extractor = None
        if self.config.depth_enabled:
            print("🧭 Depth Enabled: Initializing Depth Anything V2...")
            try:
                self.depth_extractor = DepthAnythingV2Extractor(
                    model=self.config.depth_model_type, device=self.config.device
                )
            except Exception as e:
                print(f"⚠️ Depth init failed: {e}")
                
        # 4. Engine & Renderer
        self.person_filter = PersonFilter(self.config.area_weight, self.config.center_weight, self.config.filter_confidence_threshold)
        self.transfer_engine = PoseTransferEngine(config=self.transfer_config)
        self.hand_refiner = HandRefiner(self.config.min_hand_size)
        self.renderer = SkeletonRenderer(self.config.line_thickness, self.config.point_radius)

    def extract_pose(self, image):
        """포즈 추출 (DWPose + Cross Filter)"""
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        h, w = img.shape[:2]
        
        # 1. DWPose 추출
        all_kpts, all_scores = self.extractor.extract(img)
        
        if len(all_kpts) == 0:
            return np.zeros((133, 2)), np.zeros(133), -1, (h, w)
            
        # 2. Person Filter (주인공 찾기)
        kpts, scores, idx = all_kpts[0], all_scores[0], 0
        if len(all_kpts) > 1 and self.config.filter_enabled:
            kpts, scores, idx, _ = self.person_filter.select_main_person(all_kpts, all_scores, (h, w))
            
        # 3. [FIX] Cross Filter 적용 (할루시네이션 제거)
        if self.config.cross_filter_enabled and self.body_extractor:
            # Body Extractor 추출
            body_kpts_all, body_scores_all = self.body_extractor.extract(img)
            
            if len(body_kpts_all) > 0:
                # 같은 인덱스의 사람을 찾거나, 가장 큰 사람 선택
                # 여기선 간단히 0번 사용 (개선 가능)
                body_kpts, body_scores = body_kpts_all[0], body_scores_all[0]
                
                # 필터링 실행
                kpts, scores, approved = self.cross_filter.filter(
                    body_keypoints=body_kpts, body_scores=body_scores,
                    dw_keypoints=kpts, dw_scores=scores
                )
                print(f"   🛡️ Cross Filter: kept {len(approved)} keypoints")
        
        # 4. Hand Refinement
        if self.config.hand_refinement_enabled:
            kpts, scores, _ = self.hand_refiner.refine_both_hands(img, kpts, scores, self.extractor)
            
        return kpts, scores, idx, (h, w)

    def transfer(self, source_image, reference_image):
        """전이 파이프라인 실행"""
        if isinstance(source_image, (str, Path)): src_img = load_image(source_image)
        else: src_img = source_image
        if isinstance(reference_image, (str, Path)): ref_img = load_image(reference_image)
        else: ref_img = reference_image
        
        # 1. Extraction
        src_kpts, src_scores, src_idx, src_size = self.extract_pose(src_img)
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # 2. Depth
        src_depth, ref_depth = None, None
        src_d_vals, ref_d_vals = None, None
        if self.depth_extractor:
            src_depth = self.depth_extractor.estimate(src_img)
            ref_depth = self.depth_extractor.estimate(ref_img)
            # Sample values...
        
        # 3. Layout (AlignManager)
        src_bbox, src_face, src_dbg = self.bbox_mgr.get_bboxes(src_img, src_kpts, src_scores)
        ref_bbox, ref_face, ref_dbg = self.bbox_mgr.get_bboxes(ref_img, ref_kpts, ref_scores)
        
        layout = self.align_mgr.analyze_layout(
            src_bbox, ref_bbox, src_kpts, src_scores, ref_kpts, ref_scores
        )
        
        # Debug Images
        src_debug_img = self.bbox_mgr.draw_debug(src_img.copy(), src_dbg) if self.config.debug_bbox_visualization else None
        ref_debug_img = self.bbox_mgr.draw_debug(ref_img.copy(), ref_dbg) if self.config.debug_bbox_visualization else None

        # 4. Engine Execution
        result = self.transfer_engine.transfer(
            src_kpts, src_scores, ref_kpts, ref_scores,
            src_img.shape[:2], ref_img.shape[:2],
            layout=layout
        )
        
        trans_kpts, trans_scores = result.keypoints, result.scores
        
        # 5. [FIX] Canvas Resize (확장)
        # 전이된 포즈에 맞춰 소스 이미지를 확장
        final_img, final_kpts, final_size = self.canvas_mgr.expand_canvas_to_fit(
            src_img, trans_kpts, trans_scores, padding_ratio=0.1
        )
        
        # 6. Render Skeleton
        skeleton_img = self.renderer.render_skeleton_only(
            (final_size[0], final_size[1], 3),
            final_kpts, trans_scores
        )
        
        # 7. Alignment Info Packaging
        align_info = layout # or wrap into AlignmentInfo
        
        return PipelineResult(
            transferred_keypoints=final_kpts,
            transferred_scores=trans_scores,
            source_keypoints=src_kpts, source_scores=src_scores,
            reference_keypoints=ref_kpts, reference_scores=ref_scores,
            skeleton_image=skeleton_img,
            image_size=final_size,
            modified_source_image=final_img,
            processing_info={'transfer_log': result.transfer_log},
            alignment_info=align_info,
            src_debug_image=src_debug_img,
            ref_debug_image=ref_debug_img
        )