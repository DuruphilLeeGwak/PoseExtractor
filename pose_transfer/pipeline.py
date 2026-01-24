"""
포즈 전이 파이프라인 v20 (Cross-Filter Only)

변경사항:
1. Ghost Filter 완전 제거 (Cross-Filter로 통합)
2. Cross-Filter가 할루시네이션 제거 전담
3. 파이프라인 단순화 및 성능 향상
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Set, List
from dataclasses import dataclass, field
from enum import Enum

from .extractors import (
    DWPoseExtractor, DWPoseExtractorFactory, PersonFilter, RTMLIB_AVAILABLE
)
from .extractors.keypoint_constants import BODY_KEYPOINTS
from .transfer import PoseTransferEngine, TransferConfig, FallbackStrategy
from .refiners import HandRefiner
from .renderers import SkeletonRenderer
from .utils import load_config, convert_to_openpose_format, load_image

# Logic Modules
from .logic import (
    BboxManager, AlignManager, PostProcessor, CanvasManager,
    DebugBboxData, BboxInfo,
    COLOR_KPT_BBOX, COLOR_YOLO_BBOX, COLOR_HYBRID_PERSON, COLOR_HYBRID_FACE
)

# [v4.0] Cross Filter - Body + DWPose 결합
# (실제 import는 _init_modules()에서 lazy loading)


@dataclass
class PipelineConfig:
    """파이프라인 통합 설정"""
    # Model
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    
    # Person Filter
    filter_enabled: bool = True
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    
    # Hand Refinement
    hand_refinement_enabled: bool = True
    min_hand_size: int = 48
    
    # Fallback
    fallback_enabled: bool = True
    
    # Transfer
    transfer_confidence_threshold: float = 0.3
    lower_body_confidence_threshold: float = 2.0
    lower_body_margin_ratio: float = 0.10
    visibility_margin: float = 0.2
    
    # Rendering
    line_thickness: int = 4
    face_line_thickness: int = 2
    hand_line_thickness: int = 2
    point_radius: int = 4
    kpt_threshold: float = 0
    
    # Output / Crop
    auto_crop_enabled: bool = True
    crop_padding_px: int = 50
    head_padding_ratio: float = 1.0
    canvas_padding_ratio: float = 0.1
    
    # Alignment / Logic
    full_body_min_valid_lower: int = 4
    yolo_verification_enabled: bool = True
    yolo_person_conf: float = 0.5
    yolo_face_conf: float = 0.3
    face_scale_enabled: bool = True
    
    # [v4.0] Cross Filter - Body + DWPose 교차 필터링
    cross_filter_enabled: bool = True
    cross_body_confidence_threshold: float = 0.25  # Body 검출 임계값 (knee 보존용)
    cross_enable_hand_dependency: bool = True
    cross_enable_foot_dependency: bool = True
    cross_enable_face_dependency: bool = True
    cross_dw_min_confidence: float = 0.05
    cross_dw_high_confidence_threshold: float = 8.0  # DWPose 고신뢰도 보호
    cross_dw_full_body_confidence_threshold: float = 6.0  # DWPose 전신 확신 모드
    cross_dw_suspicious_threshold: float = 2.0  # 의심 키포인트 범위 (0.05~2.0)
    cross_clean_mode_body_threshold: float = 0.2  # Clean mode일 때 body 임계값 (완화)
    cross_hand_hallucination_check: bool = True  # 손목 없으면 의심 손가락 제거
    cross_hand_dw_min_confidence: float = 2.0  # 손가락 DWPose 최소 신뢰도 (손가락 전용)
    cross_foot_hallucination_check: bool = True  # 발목 Body 낮으면 발가락 제거
    cross_foot_body_confidence_threshold: float = 0.25  # 발 할루시네이션 판정용
    cross_foot_dw_min_confidence: float = 2.5  # 발가락 DWPose 최소 신뢰도 (발가락 전용)
    
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
        cross = config.get('cross_filter', {})
        
        return cls(
            # Model
            backend=config.get('model', {}).get('backend', 'onnxruntime'),
            device=config.get('model', {}).get('device', 'cuda'),
            mode=config.get('model', {}).get('mode', 'performance'),
            to_openpose=config.get('model', {}).get('to_openpose', False),
            
            # Person Filter
            filter_enabled=config.get('person_filter', {}).get('enabled', True),
            area_weight=config.get('person_filter', {}).get('area_weight', 0.6),
            center_weight=config.get('person_filter', {}).get('center_weight', 0.4),
            filter_confidence_threshold=config.get('person_filter', {}).get('confidence_threshold', 0.3),
            
            # Hand
            hand_refinement_enabled=config.get('hand_refinement', {}).get('enabled', True),
            min_hand_size=config.get('hand_refinement', {}).get('min_hand_size', 48),
            
            # Fallback
            fallback_enabled=config.get('fallback', {}).get('symmetric_mirror', True),
            
            # Transfer
            transfer_confidence_threshold=transfer.get('confidence_threshold', 0.3),
            lower_body_confidence_threshold=transfer.get('lower_body_confidence_threshold', 2.0),
            lower_body_margin_ratio=transfer.get('lower_body_margin_ratio', 0.10),
            visibility_margin=transfer.get('visibility_margin', 0.2),
            
            # Rendering
            line_thickness=rendering.get('line_thickness', 4),
            face_line_thickness=rendering.get('face_line_thickness', 2),
            hand_line_thickness=rendering.get('hand_line_thickness', 2),
            point_radius=rendering.get('point_radius', 4),
            kpt_threshold=rendering.get('kpt_threshold', 0.3),
            
            # Output
            auto_crop_enabled=output.get('auto_crop_enabled', True),
            crop_padding_px=output.get('crop_padding_px', 50),
            head_padding_ratio=output.get('head_padding_ratio', 1.0),
            canvas_padding_ratio=output.get('canvas_padding_ratio', 0.1),
            
            # Alignment
            full_body_min_valid_lower=alignment.get('full_body_min_valid_lower', 4),
            yolo_verification_enabled=alignment.get('yolo_verification_enabled', True),
            yolo_person_conf=alignment.get('yolo_person_conf', 0.5),
            yolo_face_conf=alignment.get('yolo_face_conf', 0.3),
            face_scale_enabled=alignment.get('face_scale_enabled', True),
            
            # [v4.0] Cross Filter
            cross_filter_enabled=cross.get('enabled', True),  # 기본값 True로 변경
            cross_body_confidence_threshold=cross.get('body_confidence_threshold', 0.25),
            cross_enable_hand_dependency=cross.get('enable_hand_dependency', True),
            cross_enable_foot_dependency=cross.get('enable_foot_dependency', True),
            cross_enable_face_dependency=cross.get('enable_face_dependency', True),
            cross_dw_min_confidence=cross.get('dw_min_confidence', 0.05),
            cross_dw_high_confidence_threshold=cross.get('dw_high_confidence_threshold', 8.0),
            cross_dw_full_body_confidence_threshold=cross.get('dw_full_body_confidence_threshold', 6.0),
            cross_dw_suspicious_threshold=cross.get('dw_suspicious_threshold', 2.0),
            cross_clean_mode_body_threshold=cross.get('clean_mode_body_threshold', 0.2),
            cross_hand_hallucination_check=cross.get('hand_hallucination_check', True),
            cross_hand_dw_min_confidence=cross.get('hand_dw_min_confidence', 2.0),
            cross_foot_hallucination_check=cross.get('foot_hallucination_check', True),
            cross_foot_body_confidence_threshold=cross.get('foot_body_confidence_threshold', 0.25),
            cross_foot_dw_min_confidence=cross.get('foot_dw_min_confidence', 2.5),
            
            # Bbox
            person_bbox_margin=bbox.get('person_margin', 0.0),
            face_bbox_margin=bbox.get('face_margin', 0.0),
            
            # Debug
            debug_bbox_visualization=debug.get('bbox_visualization', False),
            viz_kpt_bbox=debug.get('visualize_keypoint_bbox', True),
            viz_yolo_bbox=debug.get('visualize_yolo_bbox', True),
            viz_hybrid_bbox=debug.get('visualize_hybrid_bbox', True),
        )


@dataclass
class AlignmentInfo:
    """정렬 정보 (단순화)"""
    should_transfer_lower: bool  # 하반신 전이 여부
    align_by_feet: bool  # 발 정렬 사용 여부
    src_person_bbox: Any
    src_face_bbox: Any
    ref_face_bbox: Any
    face_scale_ratio: float
    alignment_method: str  # "feet" or "face"
    yolo_log: Dict[str, bool]


@dataclass
class PipelineResult:
    """파이프라인 결과"""
    transferred_keypoints: np.ndarray
    transferred_scores: np.ndarray
    source_keypoints: np.ndarray
    source_scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    reference_keypoints: np.ndarray
    reference_scores: np.ndarray
    skeleton_image: np.ndarray
    image_size: Tuple[int, int]
    modified_source_image: Optional[np.ndarray] = None
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
    """포즈 전이 파이프라인"""
    
    def __init__(self, config: Optional[PipelineConfig] = None, yaml_config: Optional[dict] = None):
        self.config = config or PipelineConfig()
        self.yaml_config = yaml_config
        
        # Logic Managers
        self.bbox_mgr = BboxManager(self.config)
        self.align_mgr = AlignManager(self.config)
        self.post_proc = PostProcessor(self.config)
        self.canvas_mgr = CanvasManager(self.config)
        
        self._init_modules()
    
    def _init_modules(self):
        """모듈 초기화"""
        if not RTMLIB_AVAILABLE:
            raise RuntimeError("rtmlib not installed.")
        
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend,
            device=self.config.device,
            mode=self.config.mode,
            to_openpose=self.config.to_openpose,
            force_new=True
        )
        
        # [v4.0] Body extractor (17 keypoints, Cross-Filter용)
        self.body_extractor = None
        if self.config.cross_filter_enabled:
            from .extractors import BodyExtractor
            self.body_extractor = BodyExtractor(
                mode='balanced',
                backend='onnxruntime',
                device='cpu'
            )
            print(f"✅ Body Extractor 초기화 완료 (Cross-Filter 모드)")
        
        # [v4.0] Cross Filter (Body + DWPose 결합)
        self.cross_filter = None
        if self.config.cross_filter_enabled:
            from .logic import CrossFilter, CrossFilterConfig
            self.cross_filter = CrossFilter(
                config=CrossFilterConfig(
                    body_confidence_threshold=self.config.cross_body_confidence_threshold,
                    enable_hand_dependency=self.config.cross_enable_hand_dependency,
                    enable_foot_dependency=self.config.cross_enable_foot_dependency,
                    enable_face_dependency=self.config.cross_enable_face_dependency,
                    dw_min_confidence=self.config.cross_dw_min_confidence,
                    dw_high_confidence_threshold=self.config.cross_dw_high_confidence_threshold,
                    dw_full_body_confidence_threshold=self.config.cross_dw_full_body_confidence_threshold,
                    dw_suspicious_threshold=self.config.cross_dw_suspicious_threshold,
                    clean_mode_body_threshold=self.config.cross_clean_mode_body_threshold,
                    hand_hallucination_check=self.config.cross_hand_hallucination_check,
                    hand_dw_min_confidence=self.config.cross_hand_dw_min_confidence,
                    foot_hallucination_check=self.config.cross_foot_hallucination_check,
                    foot_body_confidence_threshold=self.config.cross_foot_body_confidence_threshold,
                    foot_dw_min_confidence=self.config.cross_foot_dw_min_confidence
                )
            )
            print(f"✅ Cross Filter 초기화 완료")
            print(f"   - Body Confidence Threshold: {self.config.cross_body_confidence_threshold}")
            print(f"   - Foot Body Threshold: {self.config.cross_foot_body_confidence_threshold}")
            print(f"   - DWPose High Confidence Threshold: {self.config.cross_dw_high_confidence_threshold}")
            print(f"   - DWPose Full Body Confidence Threshold: {self.config.cross_dw_full_body_confidence_threshold}")
            print(f"   - Clean Mode Suspicious Threshold: {self.config.cross_dw_suspicious_threshold}")
            print(f"   - Hand Dependency: {self.config.cross_enable_hand_dependency}")
            print(f"   - Foot Dependency: {self.config.cross_enable_foot_dependency}")
            print(f"   - Face Dependency: {self.config.cross_enable_face_dependency}")
        
        self.person_filter = PersonFilter(
            area_weight=self.config.area_weight,
            center_weight=self.config.center_weight,
            confidence_threshold=self.config.filter_confidence_threshold
        )
        
        transfer_config = TransferConfig(
            confidence_threshold=self.config.transfer_confidence_threshold,
            visibility_margin=self.config.visibility_margin
        )
        self.transfer_engine = PoseTransferEngine(
            config=transfer_config,
            yaml_config=self.yaml_config
        )
        
        self.fallback_strategy = FallbackStrategy(
            confidence_threshold=self.config.transfer_confidence_threshold
        )
        
        self.hand_refiner = HandRefiner(
            min_hand_size=self.config.min_hand_size,
            confidence_threshold=self.config.transfer_confidence_threshold
        )
        
        # 렌더링 설정
        rendering_config = self.yaml_config.get('rendering', {})
        draw_neck = rendering_config.get('draw_neck', False)
        
        self.renderer = SkeletonRenderer(
            line_thickness=self.config.line_thickness,
            point_radius=self.config.point_radius,
            kpt_threshold=self.config.kpt_threshold,
            face_line_thickness=self.config.face_line_thickness,
            hand_line_thickness=self.config.hand_line_thickness,
            draw_neck=draw_neck
        )

    def extract_pose(
        self, 
        image: Union[np.ndarray, str, Path], 
        filter_person: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
        """포즈 추출 (DWPose Wholebody 133 keypoints)"""
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
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
        
        # [v4.0] Cross-Filter: Body + DWPose 결합
        if self.config.cross_filter_enabled and self.body_extractor and self.cross_filter:
            # Body 17 keypoints 추출
            body_kpts_all, body_scores_all = self.body_extractor.extract(img)
            if len(body_kpts_all) > 0:
                # DWPose와 동일한 person 선택
                if filter_person and self.config.filter_enabled and len(body_kpts_all) > 1:
                    body_kpts = body_kpts_all[idx] if idx < len(body_kpts_all) else body_kpts_all[0]
                    body_scores = body_scores_all[idx] if idx < len(body_scores_all) else body_scores_all[0]
                else:
                    body_kpts = body_kpts_all[0]
                    body_scores = body_scores_all[0]
                
                # CrossFilter 적용: Body가 승인한 부위만 DWPose 사용
                filtered_kpts, filtered_scores, approved_indices = self.cross_filter.filter(
                    body_keypoints=body_kpts,
                    body_scores=body_scores,
                    dw_keypoints=kpts,
                    dw_scores=scores
                )
                
                # 필터링된 결과로 교체
                kpts = filtered_kpts
                scores = filtered_scores
                
                print(f"✅ Cross-Filter 적용: {len(approved_indices)}/133 keypoints 승인")
        
        if self.config.hand_refinement_enabled:
            kpts, scores, _ = self.hand_refiner.refine_both_hands(
                img, kpts, scores, self.extractor
            )
        
        return kpts, scores, idx, image_size

    def _sync_scale_to_source_face(
        self,
        trans_kpts: np.ndarray,
        trans_scores: np.ndarray,
        src_face: Any,
        align_by_feet: bool,
    ) -> Tuple[np.ndarray, float]:
        """
        Source 얼굴 크기에 맞춰 전이된 키포인트 스케일 동기화
        
        Args:
            trans_kpts: 전이된 키포인트
            trans_scores: 키포인트 신뢰도
            src_face: Source 얼굴 bbox
            align_by_feet: 발 정렬 사용 여부
        
        정책:
        - align_by_feet=True (발 정렬): 얼굴 관련 키포인트만 Pivot 기준 스케일링
          (바닥 정렬 안정성을 위해 하체는 그대로 유지)
        - align_by_feet=False (얼굴 정렬): 전체 키포인트 동일 배율 스케일링
        
        Returns:
            (scaled_trans_kpts, scale_factor)
        """
        if not getattr(self.config, 'face_scale_enabled', True):
            return trans_kpts, 1.0

        current_trans_face = self.bbox_mgr._kpt_to_face_public(trans_kpts, trans_scores)
        if current_trans_face.size <= 1 or src_face.size <= 1:
            return trans_kpts, 1.0

        scale_factor = float(np.clip(src_face.size / current_trans_face.size, 0.5, 2.0))
        if abs(scale_factor - 1.0) < 1e-6:
            return trans_kpts, 1.0

        scaled = trans_kpts.copy()

        # 발 정렬 모드: 얼굴만 스케일링 (바디/발 정렬 영향 최소화)
        if align_by_feet:
            # 얼굴 관련 인덱스: body(0~4) + face(23~90)
            face_indices = list(range(0, 5)) + list(range(23, 91))

            # Pivot: 코를 기준으로 스케일링 (목 길이 보존)
            NOSE = 0
            if NOSE < len(trans_scores) and trans_scores[NOSE] > 0.1:
                pivot = trans_kpts[NOSE].astype(np.float32)
            else:
                pivot = np.array(current_trans_face.center, dtype=np.float32)

            for idx in face_indices:
                if idx < len(trans_scores) and trans_scores[idx] > 0.1:
                    scaled[idx] = pivot + (scaled[idx] - pivot) * scale_factor

            return scaled, scale_factor

        # 얼굴 정렬 모드: 전체 스켈레톤 스케일링
        scaled *= scale_factor
        return scaled, scale_factor

    def transfer(self, source_image, reference_image, output_image_size=None):
        """포즈 전이 메인 메서드"""
        print("\n" + "#"*70)
        print("# 🔍 [DEBUG] PoseTransferPipeline.transfer() START")
        print("#"*70)
        
        # 이미지 로드
        if isinstance(source_image, (str, Path)): src_img = load_image(source_image)
        else: src_img = source_image
        if isinstance(reference_image, (str, Path)): ref_img = load_image(reference_image)
        else: ref_img = reference_image
        
        src_h, src_w = src_img.shape[:2]
        ref_h, ref_w = ref_img.shape[:2]
        print(f"\n📐 Image Sizes: src={src_w}x{src_h}, ref={ref_w}x{ref_h}")
        
        # [STEP 1] 포즈 추출
        print("\n[STEP 1] Extracting poses...")
        src_kpts, src_scores, src_idx, src_size = self.extract_pose(src_img)
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # Cross-Filter가 extract_pose()에서 이미 적용됨
        src_filtered_scores = src_scores
        ref_filtered_scores = ref_scores
        
        # [STEP 2] 정렬 방식 결정 (단순화)
        print("\n[STEP 2] Determining alignment strategy...")
        should_transfer_lower, align_by_feet = self.align_mgr.should_align_by_feet(
            src_filtered_scores, ref_filtered_scores
        )
        print(f"   Result: transfer_lower={should_transfer_lower}, align_by_feet={align_by_feet}")
        
        # [STEP 3] Bbox 계산
        src_person, src_face, src_debug = self.bbox_mgr.get_bboxes(src_img, src_kpts, src_filtered_scores)
        ref_person, ref_face, ref_debug = self.bbox_mgr.get_bboxes(ref_img, ref_kpts, ref_filtered_scores)
        
        src_debug_img = None
        ref_debug_img = None
        if self.config.debug_bbox_visualization:
            src_ov = self.renderer.render(
                src_img, src_kpts, src_filtered_scores
            )
            src_debug_img = self.bbox_mgr.draw_debug(src_ov, src_debug)
            ref_ov = self.renderer.render(
                ref_img, ref_kpts, ref_filtered_scores
            )
            ref_debug_img = self.bbox_mgr.draw_debug(ref_ov, ref_debug)
        
        # [STEP 6] 포즈 전이
        print("\n[STEP 4] Transferring...")
        result = self.transfer_engine.transfer(
            src_kpts, src_filtered_scores, ref_kpts, ref_filtered_scores,
            source_image_size=(src_h, src_w), reference_image_size=(ref_h, ref_w),
            target_image_size=(src_h, src_w),
            alignment_case=should_transfer_lower
        )
        trans_kpts, trans_scores = result.keypoints, result.scores
        
        # [STEP 5] Post-processing
        trans_kpts, trans_scores = self.post_proc.process(
            trans_kpts, trans_scores, src_filtered_scores, ref_filtered_scores,
            should_transfer_lower
        )
        
        # [STEP 6] Scaling
        trans_kpts, scale = self._sync_scale_to_source_face(trans_kpts, trans_scores, src_face, align_by_feet)
        
        # [STEP 10] Aligning
        trans_kpts = self.align_mgr.align_coordinates(
            trans_kpts, trans_scores, align_by_feet, src_person, src_face,
            lambda k, s: self.bbox_mgr._kpt_to_face_public(k, s)
        )
        
        # [STEP 11] Head Padding
        head_pad = self.post_proc.apply_head_padding(trans_kpts, trans_scores)
        
        # [STEP 12] Canvas Expansion
        final_src_img, final_kpts, final_size = self.canvas_mgr.expand_canvas_to_fit(
            src_img, trans_kpts, trans_scores, head_pad_px=head_pad
        )
        
        # [STEP 12.5] Auto Crop (Optional)
        if self.config.auto_crop_enabled:
            print("\n[STEP 12.5] Auto Crop - keypoints bounds 기준 크롭...")
            final_src_img, final_kpts, final_size = self.canvas_mgr.crop_to_keypoints(
                final_src_img,
                final_kpts,
                trans_scores,
                head_pad_px=head_pad,
            )
        
        final_h, final_w = final_size
        final_filtered_scores = trans_scores
        # use_face 설정 반영: Face landmarks (23~90) 비활성화
        if getattr(self.transfer_engine.config, 'use_face', True) is False:
            final_filtered_scores = final_filtered_scores.copy()
            final_filtered_scores[23:91] = 0
        
        # [RENDER] Skeleton
        skeleton_image = self.renderer.render_skeleton_only(
            (final_h, final_w, 3), 
            final_kpts, 
            final_filtered_scores
        )
        
        # Result Packaging
        align_info = AlignmentInfo(
            should_transfer_lower=should_transfer_lower,
            align_by_feet=align_by_feet,
            src_person_bbox=src_person, 
            src_face_bbox=src_face, 
            ref_face_bbox=ref_face,
            face_scale_ratio=scale, 
            alignment_method="feet" if align_by_feet else "face",
            yolo_log={'person': src_debug.yolo_person is not None}
        )
        
        print("\n" + "#"*70)
        print("# ✅ Transfer Complete")
        print("#"*70 + "\n")
        
        return PipelineResult(
            transferred_keypoints=final_kpts, transferred_scores=final_filtered_scores,
            source_keypoints=src_kpts, source_scores=src_filtered_scores,
            reference_keypoints=ref_kpts, reference_scores=ref_filtered_scores,
            source_bone_lengths=result.source_bone_lengths,
            skeleton_image=skeleton_image, image_size=final_size,
            modified_source_image=final_src_img,
            selected_person_idx={'source': src_idx, 'reference': ref_idx},
            processing_info={'transfer_log': result.transfer_log},
            alignment_info=align_info,
            src_debug_image=src_debug_img, ref_debug_image=ref_debug_img
        )

    def extract_and_render(self, image, use_face=None):
        """단일 이미지 추출 및 렌더링
        
        Args:
            image: 입력 이미지
            use_face: Face landmarks 표시 여부 (None이면 config 설정 따름)
        """
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]
        kpts, scores, _, _ = self.extract_pose(img)
        
        # Cross-Filter가 extract_pose()에서 이미 적용됨
        filtered_scores = scores
        
        # use_face 설정 반영: Face landmarks (23~90) 제어
        if use_face is False:
            # Body Face (0~4)는 유지, Face Landmarks (23~90)만 제거
            filtered_scores[23:91] = 0
        
        json_data = convert_to_openpose_format(kpts[np.newaxis, ...], filtered_scores[np.newaxis, ...], image_size)
        skel_img = self.renderer.render_skeleton_only(
            (image_size[0], image_size[1], 3), 
            kpts, 
            filtered_scores
        )
        overlay_img = self.renderer.render(
            img, kpts, filtered_scores
        )
        
        return json_data, skel_img, overlay_img