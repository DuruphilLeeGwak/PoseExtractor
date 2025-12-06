"""
포즈 전이 파이프라인 v8.1 (Fix: PipelineResult 필드 누락 수정)
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Set, List
from dataclasses import dataclass, field
from enum import Enum

# 모듈 임포트
from .extractors import (
    DWPoseExtractor, DWPoseExtractorFactory, PersonFilter, RTMLIB_AVAILABLE
)
from .transfer import PoseTransferEngine, TransferConfig, FallbackStrategy
from .refiners import HandRefiner
from .renderers import SkeletonRenderer
from .utils import load_config, convert_to_openpose_format, load_image

# [NEW] 로직 모듈 임포트
from .logic import (
    BboxManager, AlignManager, PostProcessor,
    AlignmentCase, BodyType, DebugBboxData
)

# Bbox 색상 정의 (BGR)
COLOR_KPT_BBOX = (0, 255, 0)
COLOR_YOLO_BBOX = (255, 0, 0)
COLOR_HYBRID_PERSON = (127, 0, 255)
COLOR_HYBRID_FACE = (128, 128, 0)

@dataclass
class PipelineConfig:
    """파이프라인 통합 설정"""
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    
    # Filter
    filter_enabled: bool = True
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    
    # Hand
    hand_refinement_enabled: bool = True
    min_hand_size: int = 48
    
    # Fallback
    fallback_enabled: bool = True
    
    # Transfer
    transfer_confidence_threshold: float = 0.3
    ghost_legs_clipping_enabled: bool = True
    lower_body_confidence_threshold: float = 2.0
    lower_body_margin_ratio: float = 0.10
    visibility_margin: float = 0.2
    
    # Rendering
    line_thickness: int = 4
    face_line_thickness: int = 2
    hand_line_thickness: int = 2
    point_radius: int = 4
    kpt_threshold: float = 0.3
    
    # Output / Crop
    auto_crop_enabled: bool = True
    crop_padding_px: int = 50
    head_padding_ratio: float = 1.0
    
    # Alignment / Logic
    full_body_min_valid_lower: int = 4
    yolo_verification_enabled: bool = True
    yolo_person_conf: float = 0.5
    yolo_face_conf: float = 0.3
    face_scale_enabled: bool = True
    
    # Bbox Margin
    person_bbox_margin: float = 0.0
    face_bbox_margin: float = 0.0
    
    # Debug
    debug_bbox_visualization: bool = False
    viz_kpt_bbox: bool = False
    viz_yolo_bbox: bool = False
    viz_hybrid_bbox: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        config = load_config(yaml_path)
        rendering = config.get('rendering', {})
        transfer = config.get('transfer', {})
        output = config.get('output', {})
        alignment = config.get('alignment', {})
        debug = config.get('debug', {})
        bbox = config.get('bbox', {})
        
        return cls(
            backend=config.get('model', {}).get('backend', 'onnxruntime'),
            device=config.get('model', {}).get('device', 'cuda'),
            mode=config.get('model', {}).get('mode', 'performance'),
            to_openpose=config.get('model', {}).get('to_openpose', False),
            filter_enabled=config.get('person_filter', {}).get('enabled', True),
            area_weight=config.get('person_filter', {}).get('area_weight', 0.6),
            center_weight=config.get('person_filter', {}).get('center_weight', 0.4),
            filter_confidence_threshold=config.get('person_filter', {}).get('confidence_threshold', 0.3),
            hand_refinement_enabled=config.get('hand_refinement', {}).get('enabled', True),
            min_hand_size=config.get('hand_refinement', {}).get('min_hand_size', 48),
            fallback_enabled=config.get('fallback', {}).get('symmetric_mirror', True),
            transfer_confidence_threshold=transfer.get('confidence_threshold', 0.3),
            ghost_legs_clipping_enabled=transfer.get('ghost_legs_clipping_enabled', True),
            lower_body_confidence_threshold=transfer.get('lower_body_confidence_threshold', 2.0),
            lower_body_margin_ratio=transfer.get('lower_body_margin_ratio', 0.10),
            visibility_margin=transfer.get('visibility_margin', 0.2),
            line_thickness=rendering.get('line_thickness', 4),
            face_line_thickness=rendering.get('face_line_thickness', 2),
            hand_line_thickness=rendering.get('hand_line_thickness', 2),
            point_radius=rendering.get('point_radius', 4),
            kpt_threshold=rendering.get('kpt_threshold', 0.3),
            auto_crop_enabled=output.get('auto_crop_enabled', True),
            crop_padding_px=output.get('crop_padding_px', 50),
            head_padding_ratio=output.get('head_padding_ratio', 1.0),
            full_body_min_valid_lower=alignment.get('full_body_min_valid_lower', 4),
            yolo_verification_enabled=alignment.get('yolo_verification_enabled', True),
            yolo_person_conf=alignment.get('yolo_person_conf', 0.5),
            yolo_face_conf=alignment.get('yolo_face_conf', 0.3),
            face_scale_enabled=alignment.get('face_scale_enabled', True),
            person_bbox_margin=bbox.get('person_margin', 0.0),
            face_bbox_margin=bbox.get('face_margin', 0.0),
            debug_bbox_visualization=debug.get('bbox_visualization', False),
            viz_kpt_bbox=debug.get('visualize_keypoint_bbox', True),
            viz_yolo_bbox=debug.get('visualize_yolo_bbox', True),
            viz_hybrid_bbox=debug.get('visualize_hybrid_bbox', True),
        )

@dataclass
class AlignmentInfo:
    """정렬 정보 결과"""
    case: AlignmentCase
    src_body_type: BodyType
    ref_body_type: BodyType
    face_scale_ratio: float
    alignment_method: str
    yolo_log: Dict[str, bool]
    src_person_bbox: Any
    src_face_bbox: Any
    ref_face_bbox: Any

@dataclass
class PipelineResult:
    """최종 결과 데이터"""
    transferred_keypoints: np.ndarray
    transferred_scores: np.ndarray
    source_keypoints: np.ndarray
    source_scores: np.ndarray
    # [FIX] 누락되었던 필드 추가
    source_bone_lengths: Dict[str, float] 
    reference_keypoints: np.ndarray
    reference_scores: np.ndarray
    skeleton_image: np.ndarray
    image_size: Tuple[int, int]
    selected_person_idx: Dict[str, int] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    alignment_info: Optional[AlignmentInfo] = None
    src_debug_image: Optional[np.ndarray] = None
    ref_debug_image: Optional[np.ndarray] = None
    
    def to_json(self) -> Dict[str, Any]:
        return convert_to_openpose_format(
            self.transferred_keypoints[np.newaxis, ...],
            self.transferred_scores[np.newaxis, ...],
            self.image_size
        )

class PoseTransferPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, yaml_config: Optional[dict] = None):
        self.config = config or PipelineConfig()
        self.yaml_config = yaml_config
        self._init_modules()
        
        self.bbox_mgr = BboxManager(self.config)
        self.align_mgr = AlignManager(self.config)
        self.post_proc = PostProcessor(self.config)
    
    def _init_modules(self):
        if not RTMLIB_AVAILABLE: raise RuntimeError("rtmlib not installed.")
        
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend, device=self.config.device,
            mode=self.config.mode, to_openpose=self.config.to_openpose, force_new=True
        )
        self.person_filter = PersonFilter(
            area_weight=self.config.area_weight, center_weight=self.config.center_weight,
            confidence_threshold=self.config.filter_confidence_threshold
        )
        transfer_config = TransferConfig(
            confidence_threshold=self.config.transfer_confidence_threshold,
            visibility_margin=self.config.visibility_margin
        )
        self.transfer_engine = PoseTransferEngine(config=transfer_config, yaml_config=self.yaml_config)
        self.fallback_strategy = FallbackStrategy(confidence_threshold=self.config.transfer_confidence_threshold)
        self.hand_refiner = HandRefiner(min_hand_size=self.config.min_hand_size, confidence_threshold=self.config.transfer_confidence_threshold)
        self.renderer = SkeletonRenderer(
            line_thickness=self.config.line_thickness, point_radius=self.config.point_radius,
            kpt_threshold=self.config.kpt_threshold, face_line_thickness=self.config.face_line_thickness,
            hand_line_thickness=self.config.hand_line_thickness
        )

    def extract_pose(self, image: Union[np.ndarray, str, Path], filter_person: bool = True) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int,int]]:
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        
        image_size = img.shape[:2]
        all_kpts, all_scores = self.extractor.extract(img)
        
        if len(all_kpts) == 0:
            return np.zeros((133, 2)), np.zeros(133), -1, image_size
        
        if filter_person and self.config.filter_enabled and len(all_kpts) > 1:
            kpts, scores, idx, _ = self.person_filter.select_main_person(
                all_kpts, all_scores, image_size
            )
        else:
            kpts, scores, idx = all_kpts[0], all_scores[0], 0
        
        if self.config.hand_refinement_enabled:
            kpts, scores, _ = self.hand_refiner.refine_both_hands(
                img, kpts, scores, self.extractor
            )
        
        return kpts, scores, idx, image_size

    def transfer(self, source_image, reference_image, output_image_size=None):
        if isinstance(source_image, (str, Path)): src_img = load_image(source_image)
        else: src_img = source_image
        if isinstance(reference_image, (str, Path)): ref_img = load_image(reference_image)
        else: ref_img = reference_image
        
        src_h, src_w = src_img.shape[:2]; ref_h, ref_w = ref_img.shape[:2]
        
        # [STEP 1] 포즈 추출
        print("\n[STEP 1] Extracting poses...")
        src_kpts, src_scores, src_idx, src_size = self.extract_pose(src_img)
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # [STEP 2] 신체 유형 판별
        print("\n[STEP 2] Determining Body Type...")
        src_type, ref_type, case = self.align_mgr.determine_case(src_kpts, src_scores, ref_kpts, ref_scores)
        print(f"   Case {case.value} ({src_type.value} -> {ref_type.value})")
        
        # [STEP 3] Bbox 추출 (Manager 위임)
        print("\n[STEP 3] Bbox Calculation...")
        src_person, src_face, src_debug = self.bbox_mgr.get_bboxes(src_img, src_kpts, src_scores)
        ref_person, ref_face, ref_debug = self.bbox_mgr.get_bboxes(ref_img, ref_kpts, ref_scores)
        
        # 디버그 이미지 생성
        src_debug_img = None; ref_debug_img = None
        if self.config.debug_bbox_visualization:
            src_ov = self.renderer.render(src_img, src_kpts, src_scores)
            src_debug_img = self.bbox_mgr.draw_debug(src_ov, src_debug)
            ref_ov = self.renderer.render(ref_img, ref_kpts, ref_scores)
            ref_debug_img = self.bbox_mgr.draw_debug(ref_ov, ref_debug)

        # [STEP 4] 포즈 전이 (Engine)
        print("\n[STEP 4] Transferring...")
        result = self.transfer_engine.transfer(
            src_kpts, src_scores, ref_kpts, ref_scores,
            source_image_size=(src_h, src_w), reference_image_size=(ref_h, ref_w)
        )
        trans_kpts, trans_scores = result.keypoints, result.scores
        
        # [STEP 5] 케이스별 키포인트 후처리
        print("\n[STEP 5] Post-processing Keys...")
        trans_kpts, trans_scores = self.post_proc.process_by_case(trans_kpts, trans_scores, case, src_scores)
        
        # [STEP 7] 얼굴 크기 기반 스케일링
        print("\n[STEP 7] Scaling...")
        scale = self.align_mgr.calc_scale(src_face.size, ref_face.size)
        trans_kpts *= scale
        print(f"   Scale Factor: {scale:.2f}")
        
        # [STEP 8] 케이스별 정렬 (좌표 이동)
        print("\n[STEP 8] Aligning...")
        trans_kpts = self.align_mgr.align_coordinates(
            trans_kpts, trans_scores, case, src_person, src_face,
            lambda k, s: self.bbox_mgr._kpt_to_face(k, s) 
        )
        
        # [STEP 9] 머리 방향 패딩 계산
        print("\n[STEP 9] Head Padding...")
        head_pad = self.post_proc.apply_head_padding(trans_kpts, trans_scores)
        
        # [STEP 10] 최종 크롭 (캔버스 확정)
        print("\n[STEP 10] Finalizing...")
        final_kpts, final_size = self.post_proc.finalize_canvas(trans_kpts, trans_scores, head_pad)
        
        # 최종 렌더링
        skeleton_image = self.renderer.render_skeleton_only(
            (final_size[0], final_size[1], 3), final_kpts, trans_scores
        )
        
        align_info = AlignmentInfo(
            case=case, src_body_type=src_type, ref_body_type=ref_type,
            face_scale_ratio=scale, alignment_method="feet" if case==AlignmentCase.A else "face",
            yolo_log=src_debug.hybrid_person.source if src_debug.hybrid_person else "none",
            src_person_bbox=src_person, src_face_bbox=src_face, ref_face_bbox=ref_face
        )
        
        return PipelineResult(
            transferred_keypoints=final_kpts, transferred_scores=trans_scores,
            source_keypoints=src_kpts, source_scores=src_scores,
            reference_keypoints=ref_kpts, reference_scores=ref_scores,
            source_bone_lengths=result.source_bone_lengths, # [FIX] 이제 클래스 정의에 포함됨
            skeleton_image=skeleton_image, image_size=final_size,
            selected_person_idx={'source': src_idx, 'reference': ref_idx},
            processing_info={'transfer_log': result.transfer_log},
            alignment_info=align_info,
            src_debug_image=src_debug_img, ref_debug_image=ref_debug_img
        )

    def extract_and_render(self, image):
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        image_size = img.shape[:2]
        kpts, scores, _, _ = self.extract_pose(img)
        json_data = convert_to_openpose_format(kpts[np.newaxis, ...], scores[np.newaxis, ...], image_size)
        skel_img = self.renderer.render_skeleton_only((image_size[0], image_size[1], 3), kpts, scores)
        overlay_img = self.renderer.render(img, kpts, scores)
        return json_data, skel_img, overlay_img