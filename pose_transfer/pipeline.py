"""
포즈 전이 파이프라인 v3
- Ghost Legs 클리핑
- [NEW] 키포인트 기반 자동 패딩/크롭 (trans_sk용)
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
    
    # Ghost Legs 클리핑 설정
    ghost_legs_clipping_enabled: bool = True
    lower_body_confidence_threshold: float = 2.0
    lower_body_margin_ratio: float = 0.10
    
    # [NEW] 키포인트 기반 크롭 설정
    auto_crop_enabled: bool = True
    crop_padding_px: int = 50  # 바운딩 박스 외부에 추가할 패딩 (픽셀)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """YAML 파일에서 설정 로드"""
        config = load_config(yaml_path)
        rendering = config.get('rendering', {})
        transfer = config.get('transfer', {})
        output = config.get('output', {})
        
        print("\n[DEBUG] Loading YAML config...")
        print(f"  model.backend: {config.get('model', {}).get('backend')}")
        print(f"  rendering.kpt_threshold: {rendering.get('kpt_threshold')}")
        print(f"  transfer.lower_body_confidence_threshold: {transfer.get('lower_body_confidence_threshold')}")
        print(f"  transfer.lower_body_margin_ratio: {transfer.get('lower_body_margin_ratio')}")
        print(f"  output.auto_crop_enabled: {output.get('auto_crop_enabled')}")
        print(f"  output.crop_padding_px: {output.get('crop_padding_px')}")
        
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
            # Ghost Legs 설정
            ghost_legs_clipping_enabled=transfer.get('ghost_legs_clipping_enabled', True),
            lower_body_confidence_threshold=transfer.get('lower_body_confidence_threshold', 2.0),
            lower_body_margin_ratio=transfer.get('lower_body_margin_ratio', 0.10),
            # [NEW] 크롭 설정
            auto_crop_enabled=output.get('auto_crop_enabled', True),
            crop_padding_px=output.get('crop_padding_px', 50),
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
        print(f"  auto_crop_enabled: {self.config.auto_crop_enabled}")
        print(f"  crop_padding_px: {self.config.crop_padding_px}")
        
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
    # [NEW] 키포인트 바운딩 박스 계산
    # ============================================================
    def _get_keypoint_bbox(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        score_threshold: float = 0.1
    ) -> Tuple[float, float, float, float]:
        """
        유효한 키포인트들의 바운딩 박스 계산
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        valid_mask = scores > score_threshold
        valid_kpts = keypoints[valid_mask]
        
        if len(valid_kpts) == 0:
            return (0, 0, 100, 100)
        
        min_x = np.min(valid_kpts[:, 0])
        min_y = np.min(valid_kpts[:, 1])
        max_x = np.max(valid_kpts[:, 0])
        max_y = np.max(valid_kpts[:, 1])
        
        return (min_x, min_y, max_x, max_y)
    
    # ============================================================
    # [NEW] 키포인트 기반 캔버스 크기 계산 및 좌표 변환
    # ============================================================
    def _calculate_canvas_and_offset(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        base_size: Tuple[int, int],
        padding: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int], np.ndarray]:
        """
        키포인트가 모두 들어오도록 캔버스 크기와 오프셋 계산
        
        Args:
            keypoints: 키포인트 좌표
            scores: 키포인트 신뢰도
            base_size: 기준 이미지 크기 (height, width)
            padding: 바운딩 박스 외부 패딩 (픽셀)
        
        Returns:
            (canvas_size, offset, adjusted_keypoints)
            - canvas_size: (height, width)
            - offset: (offset_x, offset_y) - 원본 좌표에서 캔버스 좌표로의 오프셋
            - adjusted_keypoints: 오프셋이 적용된 키포인트
        """
        base_h, base_w = base_size
        
        # 바운딩 박스 계산
        min_x, min_y, max_x, max_y = self._get_keypoint_bbox(
            keypoints, scores, self.config.kpt_threshold
        )
        
        # 패딩 적용한 바운딩 박스
        bbox_left = min_x - padding
        bbox_top = min_y - padding
        bbox_right = max_x + padding
        bbox_bottom = max_y + padding
        
        # 필요한 확장 계산
        expand_left = max(0, -bbox_left)
        expand_top = max(0, -bbox_top)
        expand_right = max(0, bbox_right - base_w)
        expand_bottom = max(0, bbox_bottom - base_h)
        
        # 캔버스 크기 (확장 포함)
        canvas_w = int(base_w + expand_left + expand_right)
        canvas_h = int(base_h + expand_top + expand_bottom)
        
        # 오프셋 (원본 좌표 -> 캔버스 좌표)
        offset_x = expand_left
        offset_y = expand_top
        
        # 키포인트 좌표 조정
        adjusted_kpts = keypoints.copy()
        adjusted_kpts[:, 0] += offset_x
        adjusted_kpts[:, 1] += offset_y
        
        print(f"   📐 [Canvas] base={base_w}x{base_h} -> canvas={canvas_w}x{canvas_h}")
        print(f"       expand: L={expand_left:.0f}, T={expand_top:.0f}, R={expand_right:.0f}, B={expand_bottom:.0f}")
        print(f"       offset: ({offset_x:.0f}, {offset_y:.0f})")
        
        return (canvas_h, canvas_w), (int(offset_x), int(offset_y)), adjusted_kpts
    
    # ============================================================
    # [NEW] 키포인트 기반 최종 크롭
    # ============================================================
    def _crop_to_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        padding: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        키포인트 바운딩 박스 + 패딩으로 이미지 크롭
        
        Returns:
            (cropped_image, cropped_keypoints)
        """
        h, w = image.shape[:2]
        
        # 바운딩 박스 계산
        min_x, min_y, max_x, max_y = self._get_keypoint_bbox(
            keypoints, scores, self.config.kpt_threshold
        )
        
        # 패딩 적용 + 경계 클리핑
        crop_x1 = max(0, int(min_x - padding))
        crop_y1 = max(0, int(min_y - padding))
        crop_x2 = min(w, int(max_x + padding))
        crop_y2 = min(h, int(max_y + padding))
        
        # 크롭
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 키포인트 좌표 조정
        cropped_kpts = keypoints.copy()
        cropped_kpts[:, 0] -= crop_x1
        cropped_kpts[:, 1] -= crop_y1
        
        print(f"   ✂️ [Crop] ({crop_x1}, {crop_y1}) ~ ({crop_x2}, {crop_y2}) -> {crop_x2-crop_x1}x{crop_y2-crop_y1}")
        
        return cropped, cropped_kpts
    
    # ============================================================
    # Ghost Legs 클리핑 함수들
    # ============================================================
    def _clip_ghost_legs(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray, 
        image_height: int,
        image_width: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """프레임 경계 밖 또는 저신뢰도 하반신 키포인트 제거"""
        if not self.config.ghost_legs_clipping_enabled:
            return keypoints, scores, 0
        
        boundary_y = image_height * (1 - self.config.lower_body_margin_ratio)
        conf_threshold = self.config.lower_body_confidence_threshold
        
        invalid_indices = self._get_invalid_lower_body_indices(
            keypoints, scores, boundary_y, conf_threshold
        )
        
        clipped_count = 0
        for idx in invalid_indices:
            if scores[idx] > 0:
                scores[idx] = 0.0
                clipped_count += 1
        
        if clipped_count > 0:
            print(f"   🔧 [Ghost Legs] Clipped {clipped_count} keypoints")
        
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
        
        lower_body_parts = [
            ('left_hip', BODY_KEYPOINTS.get('left_hip', 11)),
            ('right_hip', BODY_KEYPOINTS.get('right_hip', 12)),
            ('left_knee', BODY_KEYPOINTS.get('left_knee', 13)),
            ('right_knee', BODY_KEYPOINTS.get('right_knee', 14)),
            ('left_ankle', BODY_KEYPOINTS.get('left_ankle', 15)),
            ('right_ankle', BODY_KEYPOINTS.get('right_ankle', 16)),
        ]
        
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
            
            over_boundary = y >= boundary_y
            low_confidence = conf < conf_threshold and conf > 0
            
            if over_boundary or low_confidence:
                invalid.add(idx)
                self._invalidate_children(part_name, invalid)
        
        return invalid
    
    def _invalidate_children(self, parent_name: str, invalid: Set[int]):
        """부모가 무효화되면 자식도 재귀적으로 무효화"""
        if parent_name not in LOWER_BODY_HIERARCHY:
            return
        
        for child_name in LOWER_BODY_HIERARCHY[parent_name]:
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
        """포즈 추출 + Ghost Legs 클리핑"""
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]
        img_h, img_w = image_size
        
        all_keypoints, all_scores = self.extractor.extract(img)
        
        if len(all_keypoints) == 0:
            return np.zeros((133, 2)), np.zeros(133), -1, image_size
        
        if filter_person and self.config.filter_enabled and len(all_keypoints) > 1:
            keypoints, scores, selected_idx, best = self.person_filter.select_main_person(
                all_keypoints, all_scores, image_size
            )
        else:
            keypoints = all_keypoints[0]
            scores = all_scores[0]
            selected_idx = 0
        
        if self.config.hand_refinement_enabled:
            keypoints, scores, _ = self.hand_refiner.refine_both_hands(
                img, keypoints, scores, self.extractor
            )
        
        keypoints, scores, clipped = self._clip_ghost_legs(
            keypoints, scores, img_h, img_w
        )
        
        return keypoints, scores, selected_idx, image_size
    
    # ============================================================
    # 전이 (Transfer) - 자동 패딩/크롭 포함
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
        
        # 포즈 추출
        source_kpts, source_scores, source_idx, source_size = self.extract_pose(source_img)
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # 전이 실행
        transfer_result = self.transfer_engine.transfer(
            source_kpts, source_scores,
            ref_kpts, ref_scores,
            source_image_size=(src_h, src_w),
            reference_image_size=(ref_h, ref_w)
        )
        
        transferred_kpts = transfer_result.keypoints
        transferred_scores = transfer_result.scores
        
        # [NEW] 자동 패딩/크롭 적용
        if self.config.auto_crop_enabled:
            skeleton_image, final_kpts, final_size = self._render_with_auto_crop(
                transferred_kpts, transferred_scores,
                source_size, self.config.crop_padding_px
            )
        else:
            output_size = output_image_size or source_size
            skeleton_image = self.renderer.render_skeleton_only(
                (output_size[0], output_size[1], 3),
                transferred_kpts, transferred_scores
            )
            final_kpts = transferred_kpts
            final_size = output_size
        
        return PipelineResult(
            transferred_keypoints=final_kpts,
            transferred_scores=transferred_scores,
            source_keypoints=source_kpts,
            source_scores=source_scores,
            source_bone_lengths=transfer_result.source_bone_lengths,
            reference_keypoints=ref_kpts,
            reference_scores=ref_scores,
            skeleton_image=skeleton_image,
            image_size=final_size,
            selected_person_idx={'source': source_idx, 'reference': ref_idx},
            processing_info={'transfer_log': transfer_result.transfer_log}
        )
    
    # ============================================================
    # [NEW] 자동 패딩/크롭으로 렌더링
    # ============================================================
    def _render_with_auto_crop(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        base_size: Tuple[int, int],
        padding: int
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        키포인트가 모두 포함되도록 자동으로 캔버스 확장 후 크롭
        
        Returns:
            (skeleton_image, adjusted_keypoints, final_size)
        """
        # 1. 캔버스 크기 및 오프셋 계산
        canvas_size, offset, adjusted_kpts = self._calculate_canvas_and_offset(
            keypoints, scores, base_size, padding
        )
        
        # 2. 확장된 캔버스에 렌더링
        canvas_h, canvas_w = canvas_size
        skeleton_image = self.renderer.render_skeleton_only(
            (canvas_h, canvas_w, 3),
            adjusted_kpts, scores
        )
        
        # 3. 키포인트 바운딩 박스 + 패딩으로 크롭
        cropped_image, cropped_kpts = self._crop_to_keypoints(
            skeleton_image, adjusted_kpts, scores, padding
        )
        
        final_size = cropped_image.shape[:2]
        
        return cropped_image, cropped_kpts, final_size
    
    # ============================================================
    # 추출 + 렌더링 (단일 이미지용)
    # ============================================================
    def extract_and_render(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """단일 이미지 추출 및 렌더링"""
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]
        
        keypoints, scores, selected_idx, _ = self.extract_pose(img)
        
        json_data = convert_to_openpose_format(
            keypoints[np.newaxis, ...], scores[np.newaxis, ...], image_size
        )
        
        skeleton_image = self.renderer.render_skeleton_only(
            (image_size[0], image_size[1], 3), keypoints, scores
        )
        
        overlay_image = self.renderer.render(img, keypoints, scores)
        
        return json_data, skeleton_image, overlay_image