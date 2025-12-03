"""
포즈 전이 파이프라인 (Updated)
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from .extractors import (
    DWPoseExtractor,
    DWPoseExtractorFactory,
    PersonFilter,
    filter_main_person,
    RTMLIB_AVAILABLE
)
from .analyzers import BoneCalculator, DirectionExtractor
from .transfer import PoseTransferEngine, TransferConfig, FallbackStrategy
from .refiners import HandRefiner
from .renderers import SkeletonRenderer, render_skeleton
from .utils import (
    load_config, save_json, load_image, save_image,
    convert_to_openpose_format, PoseResult
)


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
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """YAML 파일에서 설정 로드"""
        config = load_config(yaml_path)
        rendering = config.get('rendering', {})
        
        print("\n[DEBUG] Loading YAML config...")
        print(f"  model.backend: {config.get('model', {}).get('backend')}")
        print(f"  rendering.kpt_threshold: {rendering.get('kpt_threshold')}")
        
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
            transfer_confidence_threshold=config.get('transfer', {}).get('confidence_threshold', 0.3),
            line_thickness=rendering.get('line_thickness', 4),
            face_line_thickness=rendering.get('face_line_thickness', 2),
            hand_line_thickness=rendering.get('hand_line_thickness', 2),
            point_radius=rendering.get('point_radius', 4),
            kpt_threshold=rendering.get('kpt_threshold', 0.3),
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
        # [NEW] Raw YAML Config 저장 (Engine에 전달하기 위함)
        self.yaml_config = yaml_config
        self._init_modules()
    
    def _init_modules(self):
        """모듈 초기화"""
        if not RTMLIB_AVAILABLE:
            raise RuntimeError("rtmlib is not installed.")
        
        print("\n[DEBUG] Initializing modules with config:")
        print(f"  kpt_threshold: {self.config.kpt_threshold}")
        
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
        # [NEW] yaml_config를 전달하여 얼굴 설정 등을 반영
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
    
    def extract_pose(
        self,
        image: Union[np.ndarray, str, Path],
        filter_person: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]
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
        
        return keypoints, scores, selected_idx, image_size
    
    def transfer(
        self,
        source_image: Union[np.ndarray, str, Path],
        reference_image: Union[np.ndarray, str, Path],
        output_image_size: Optional[Tuple[int, int]] = None
    ) -> PipelineResult:
        # 이미지 로드 및 사이즈 추출
        if isinstance(source_image, (str, Path)):
            source_img = load_image(source_image)
        else:
            source_img = source_image
        
        src_h, src_w = source_img.shape[:2]
        source_kpts, source_scores, source_idx, source_size = self.extract_pose(source_img)
        
        if isinstance(reference_image, (str, Path)):
            ref_img = load_image(reference_image)
        else:
            ref_img = reference_image
        
        ref_h, ref_w = ref_img.shape[:2]
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # [NEW] 이미지 사이즈 명시적 전달 (하반신 검증용)
        transfer_result = self.transfer_engine.transfer(
            source_kpts, source_scores,
            ref_kpts, ref_scores,
            source_image_size=(src_h, src_w),
            reference_image_size=(ref_h, ref_w)
        )
        
        # 폴백 적용 (필요시)
        if self.config.fallback_enabled:
            # FallbackStrategy는 별도 로직이므로 기존 유지
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
    
    def extract_and_render(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
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