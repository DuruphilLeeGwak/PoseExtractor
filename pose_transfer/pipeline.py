"""
포즈 전이 파이프라인 v2
- Ghost Legs 클리핑을 extract_pose() 단계에서 수행
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field

from .extractors import (
    DWPoseExtractor,
    DWPoseExtractorFactory,
    PersonFilter,
    filter_main_person,
    RTMLIB_AVAILABLE
)
from .extractors.keypoint_constants import BODY_KEYPOINTS, FEET_KEYPOINTS
from .analyzers import BoneCalculator, DirectionExtractor
from .transfer import PoseTransferEngine, TransferConfig, FallbackStrategy
from .refiners import HandRefiner
from .renderers import SkeletonRenderer, render_skeleton
from .utils import (
    load_config, save_json, load_image, save_image,
    convert_to_openpose_format, PoseResult
)


# ============================================================
# Ghost Legs 클리핑을 위한 계층 구조
# ============================================================
LOWER_BODY_HIERARCHY = {
    'left_hip': ['left_knee'],
    'right_hip': ['right_knee'],
    'left_knee': ['left_ankle'],
    'right_knee': ['right_ankle'],
    'left_ankle': ['left_big_toe', 'left_small_toe', 'left_heel'],
    'right_ankle': ['right_big_toe', 'right_small_toe', 'right_heel'],
}


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 모델 설정
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    
    # 다중 인물 필터링
    filter_enabled: bool = True
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    
    # 손 정밀화
    hand_refinement_enabled: bool = True
    min_hand_size: int = 48
    
    # 폴백
    fallback_enabled: bool = True
    
    # 전이 신뢰도 임계값
    transfer_confidence_threshold: float = 0.3
    
    # 렌더링
    line_thickness: int = 4
    face_line_thickness: int = 2
    hand_line_thickness: int = 2
    point_radius: int = 4
    kpt_threshold: float = 0.3
    
    # [NEW] Ghost Legs 클리핑 설정
    ghost_legs_clipping_enabled: bool = True
    lower_body_confidence_threshold: float = 2.0  # 이 미만이면 저신뢰
    lower_body_margin_ratio: float = 0.10  # 이미지 하단 10% = 경계
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """YAML 파일에서 설정 로드"""
        config = load_config(yaml_path)
        rendering = config.get('rendering', {})
        transfer = config.get('transfer', {})
        
        print("\n[DEBUG] Loading YAML config...")
        print(f"  model.backend: {config.get('model', {}).get('backend')}")
        print(f"  rendering.kpt_threshold: {rendering.get('kpt_threshold')}")
        print(f"  transfer.lower_body_confidence_threshold: {transfer.get('lower_body_confidence_threshold')}")
        print(f"  transfer.lower_body_margin_ratio: {transfer.get('lower_body_margin_ratio')}")
        
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
            line_thickness=rendering.get('line_thickness', 4),
            face_line_thickness=rendering.get('face_line_thickness', 2),
            hand_line_thickness=rendering.get('hand_line_thickness', 2),
            point_radius=rendering.get('point_radius', 4),
            kpt_threshold=rendering.get('kpt_threshold', 0.3),
            # [NEW] Ghost Legs 설정
            ghost_legs_clipping_enabled=transfer.get('ghost_legs_clipping_enabled', True),
            lower_body_confidence_threshold=transfer.get('lower_body_confidence_threshold', 2.0),
            lower_body_margin_ratio=transfer.get('lower_body_margin_ratio', 0.10),
        )


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
    selected_person_idx: Dict[str, int] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    
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
        self._init_modules()
    
    def _init_modules(self):
        """모듈 초기화"""
        if not RTMLIB_AVAILABLE:
            raise RuntimeError("rtmlib is not installed.")
        
        print("\n[DEBUG] Initializing modules with config:")
        print(f"  kpt_threshold: {self.config.kpt_threshold}")
        print(f"  ghost_legs_clipping_enabled: {self.config.ghost_legs_clipping_enabled}")
        print(f"  lower_body_confidence_threshold: {self.config.lower_body_confidence_threshold}")
        print(f"  lower_body_margin_ratio: {self.config.lower_body_margin_ratio}")
        
        # 추출기
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend,
            device=self.config.device,
            mode=self.config.mode,
            to_openpose=self.config.to_openpose,
            force_new=True
        )
        
        # 인물 필터
        self.person_filter = PersonFilter(
            area_weight=self.config.area_weight,
            center_weight=self.config.center_weight,
            confidence_threshold=self.config.filter_confidence_threshold
        )
        
        # 전이 엔진
        transfer_config = TransferConfig(
            confidence_threshold=self.config.transfer_confidence_threshold
        )
        self.transfer_engine = PoseTransferEngine(
            config=transfer_config,
            yaml_config=self.yaml_config
        )
        
        # 폴백 전략
        self.fallback_strategy = FallbackStrategy(
            confidence_threshold=self.config.transfer_confidence_threshold
        )
        
        # 손 정밀화
        self.hand_refiner = HandRefiner(
            min_hand_size=self.config.min_hand_size,
            confidence_threshold=self.config.transfer_confidence_threshold
        )
        
        # 렌더러
        self.renderer = SkeletonRenderer(
            line_thickness=self.config.line_thickness,
            point_radius=self.config.point_radius,
            kpt_threshold=self.config.kpt_threshold,
            face_line_thickness=self.config.face_line_thickness,
            hand_line_thickness=self.config.hand_line_thickness
        )
    
    # ============================================================
    # [NEW] Ghost Legs 클리핑 함수들
    # ============================================================
    def _clip_ghost_legs(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray, 
        image_height: int,
        image_width: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        프레임 경계 밖 또는 저신뢰도 하반신 키포인트 제거
        
        Returns:
            keypoints, scores, clipped_count
        """
        if not self.config.ghost_legs_clipping_enabled:
            return keypoints, scores, 0
        
        boundary_y = image_height * (1 - self.config.lower_body_margin_ratio)
        conf_threshold = self.config.lower_body_confidence_threshold
        
        # 무효화할 인덱스 수집
        invalid_indices = self._get_invalid_lower_body_indices(
            keypoints, scores, boundary_y, conf_threshold
        )
        
        # 무효화 적용
        clipped_count = 0
        for idx in invalid_indices:
            if scores[idx] > 0:
                scores[idx] = 0.0
                clipped_count += 1
        
        if clipped_count > 0:
            print(f"   🔧 [Ghost Legs] Clipped {clipped_count} keypoints (boundary_y={boundary_y:.0f}, conf_thresh={conf_threshold})")
        
        return keypoints, scores, clipped_count
    
    def _get_invalid_lower_body_indices(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        boundary_y: float,
        conf_threshold: float
    ) -> Set[int]:
        """무효화할 하반신 키포인트 인덱스 집합 반환"""
        invalid = set()
        
        # 체크할 하반신 부위 (순서 중요: 부모 먼저)
        lower_body_parts = [
            ('left_hip', BODY_KEYPOINTS.get('left_hip', 11)),
            ('right_hip', BODY_KEYPOINTS.get('right_hip', 12)),
            ('left_knee', BODY_KEYPOINTS.get('left_knee', 13)),
            ('right_knee', BODY_KEYPOINTS.get('right_knee', 14)),
            ('left_ankle', BODY_KEYPOINTS.get('left_ankle', 15)),
            ('right_ankle', BODY_KEYPOINTS.get('right_ankle', 16)),
        ]
        
        # 발 키포인트 (FEET_KEYPOINTS가 있을 경우)
        feet_parts = []
        if FEET_KEYPOINTS:
            feet_parts = [
                ('left_big_toe', FEET_KEYPOINTS.get('left_big_toe', 17)),
                ('left_small_toe', FEET_KEYPOINTS.get('left_small_toe', 18)),
                ('left_heel', FEET_KEYPOINTS.get('left_heel', 19)),
                ('right_big_toe', FEET_KEYPOINTS.get('right_big_toe', 20)),
                ('right_small_toe', FEET_KEYPOINTS.get('right_small_toe', 21)),
                ('right_heel', FEET_KEYPOINTS.get('right_heel', 22)),
            ]
        
        for part_name, idx in lower_body_parts + feet_parts:
            if idx >= len(keypoints):
                continue
                
            y = keypoints[idx][1]
            conf = scores[idx]
            
            # 무효화 조건: 경계 밖 OR 저신뢰도
            over_boundary = y >= boundary_y
            low_confidence = conf < conf_threshold and conf > 0  # 이미 0인 건 스킵
            
            if over_boundary or low_confidence:
                invalid.add(idx)
                # 자식들도 무효화
                self._invalidate_children(part_name, invalid)
        
        return invalid
    
    def _invalidate_children(self, parent_name: str, invalid: Set[int]):
        """부모가 무효화되면 자식도 재귀적으로 무효화"""
        if parent_name not in LOWER_BODY_HIERARCHY:
            return
        
        for child_name in LOWER_BODY_HIERARCHY[parent_name]:
            # BODY_KEYPOINTS에서 먼저 찾고, 없으면 FEET_KEYPOINTS에서 찾기
            if child_name in BODY_KEYPOINTS:
                child_idx = BODY_KEYPOINTS[child_name]
            elif FEET_KEYPOINTS and child_name in FEET_KEYPOINTS:
                child_idx = FEET_KEYPOINTS[child_name]
            else:
                continue
            
            invalid.add(child_idx)
            self._invalidate_children(child_name, invalid)
    
    # ============================================================
    # 포즈 추출 (Ghost Legs 클리핑 포함)
    # ============================================================
    def extract_pose(
        self,
        image: Union[np.ndarray, str, Path],
        filter_person: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
        """
        포즈 추출 + Ghost Legs 클리핑
        
        Returns:
            keypoints, scores, selected_idx, image_size
        """
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]  # (height, width)
        img_h, img_w = image_size
        
        # 1. DWPose 추출
        all_keypoints, all_scores = self.extractor.extract(img)
        
        if len(all_keypoints) == 0:
            return np.zeros((133, 2)), np.zeros(133), -1, image_size
        
        # 2. 인물 필터링
        if filter_person and self.config.filter_enabled and len(all_keypoints) > 1:
            keypoints, scores, selected_idx, best = self.person_filter.select_main_person(
                all_keypoints, all_scores, image_size
            )
        else:
            keypoints = all_keypoints[0]
            scores = all_scores[0]
            selected_idx = 0
        
        # 3. 손 정밀화
        if self.config.hand_refinement_enabled:
            keypoints, scores, _ = self.hand_refiner.refine_both_hands(
                img, keypoints, scores, self.extractor
            )
        
        # 4. [NEW] Ghost Legs 클리핑
        keypoints, scores, clipped = self._clip_ghost_legs(
            keypoints, scores, img_h, img_w
        )
        
        return keypoints, scores, selected_idx, image_size
    
    # ============================================================
    # 전이 (Transfer)
    # ============================================================
    def transfer(
        self,
        source_image: Union[np.ndarray, str, Path],
        reference_image: Union[np.ndarray, str, Path],
        output_image_size: Optional[Tuple[int, int]] = None
    ) -> PipelineResult:
        """Source와 Reference 이미지 간 포즈 전이"""
        
        # 이미지 로드
        if isinstance(source_image, (str, Path)):
            source_img = load_image(source_image)
        else:
            source_img = source_image
        
        src_h, src_w = source_img.shape[:2]
        
        if isinstance(reference_image, (str, Path)):
            ref_img = load_image(reference_image)
        else:
            ref_img = reference_image
        
        ref_h, ref_w = ref_img.shape[:2]
        
        # 포즈 추출 (Ghost Legs 클리핑 포함)
        source_kpts, source_scores, source_idx, source_size = self.extract_pose(source_img)
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # 전이 실행
        transfer_result = self.transfer_engine.transfer(
            source_kpts, source_scores,
            ref_kpts, ref_scores,
            source_image_size=(src_h, src_w),
            reference_image_size=(ref_h, ref_w)
        )
        
        # 폴백 적용 (필요시)
        if self.config.fallback_enabled:
            pass
            
        transferred_kpts = transfer_result.keypoints
        transferred_scores = transfer_result.scores
        
        output_size = output_image_size or source_size
        
        skeleton_image = self.renderer.render_skeleton_only(
            (output_size[0], output_size[1], 3),
            transferred_kpts, transferred_scores
        )
        
        return PipelineResult(
            transferred_keypoints=transferred_kpts,
            transferred_scores=transferred_scores,
            source_keypoints=source_kpts,
            source_scores=source_scores,
            source_bone_lengths=transfer_result.source_bone_lengths,
            reference_keypoints=ref_kpts,
            reference_scores=ref_scores,
            skeleton_image=skeleton_image,
            image_size=output_size,
            selected_person_idx={'source': source_idx, 'reference': ref_idx},
            processing_info={'transfer_log': transfer_result.transfer_log}
        )
    
    # ============================================================
    # 추출 + 렌더링 (단일 이미지용)
    # ============================================================
    def extract_and_render(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """단일 이미지 추출 및 렌더링 (Ghost Legs 클리핑 포함)"""
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]
        
        # 추출 (Ghost Legs 클리핑 포함)
        keypoints, scores, selected_idx, _ = self.extract_pose(img)
        
        # JSON 변환
        json_data = convert_to_openpose_format(
            keypoints[np.newaxis, ...], scores[np.newaxis, ...], image_size
        )
        
        # 렌더링
        skeleton_image = self.renderer.render_skeleton_only(
            (image_size[0], image_size[1], 3), keypoints, scores
        )
        
        overlay_image = self.renderer.render(img, keypoints, scores)
        
        return json_data, skeleton_image, overlay_image