"""
포즈 전이 파이프라인

원본 이미지의 신체 비율을 유지하면서
레퍼런스 이미지의 포즈로 전이합니다.
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
    
    # 다중 인물 필터링
    filter_enabled: bool = True
    area_weight: float = 0.6
    center_weight: float = 0.4
    
    # 손 정밀화
    hand_refinement_enabled: bool = True
    min_hand_size: int = 48
    
    # 폴백
    fallback_enabled: bool = True
    
    # 신뢰도 임계값
    confidence_threshold: float = 0.3
    
    # 렌더링
    line_thickness: int = 4
    point_radius: int = 4
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """YAML 파일에서 설정 로드"""
        config = load_config(yaml_path)
        
        return cls(
            backend=config.get('model', {}).get('backend', 'onnxruntime'),
            device=config.get('model', {}).get('device', 'cuda'),
            mode=config.get('model', {}).get('mode', 'performance'),
            filter_enabled=config.get('person_filter', {}).get('enabled', True),
            area_weight=config.get('person_filter', {}).get('area_weight', 0.6),
            center_weight=config.get('person_filter', {}).get('center_weight', 0.4),
            hand_refinement_enabled=config.get('hand_refinement', {}).get('enabled', True),
            min_hand_size=config.get('hand_refinement', {}).get('min_hand_size', 48),
            fallback_enabled=config.get('fallback', {}).get('symmetric_mirror', True),
            confidence_threshold=config.get('transfer', {}).get('confidence_threshold', 0.3),
            line_thickness=config.get('rendering', {}).get('line_thickness', 4),
            point_radius=config.get('rendering', {}).get('point_radius', 4),
        )


@dataclass
class PipelineResult:
    """파이프라인 결과"""
    # 전이된 결과
    transferred_keypoints: np.ndarray
    transferred_scores: np.ndarray
    
    # 원본 정보
    source_keypoints: np.ndarray
    source_scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    
    # 레퍼런스 정보
    reference_keypoints: np.ndarray
    reference_scores: np.ndarray
    
    # 이미지
    skeleton_image: np.ndarray
    
    # 메타데이터
    image_size: Tuple[int, int]
    selected_person_idx: Dict[str, int] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> Dict[str, Any]:
        """OpenPose 호환 JSON 포맷으로 변환"""
        return convert_to_openpose_format(
            self.transferred_keypoints[np.newaxis, ...],
            self.transferred_scores[np.newaxis, ...],
            self.image_size
        )


class PoseTransferPipeline:
    """
    포즈 전이 파이프라인
    
    사용법:
    ```python
    pipeline = PoseTransferPipeline()
    result = pipeline.transfer(source_image, reference_image)
    
    # JSON 저장
    save_json(result.to_json(), 'output.json')
    
    # 스켈레톤 이미지 저장
    save_image(result.skeleton_image, 'skeleton.png')
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config or PipelineConfig()
        
        # 모듈 초기화
        self._init_modules()
    
    def _init_modules(self):
        """모듈 초기화"""
        if not RTMLIB_AVAILABLE:
            raise RuntimeError(
                "rtmlib is not installed. "
                "Please run: pip install rtmlib onnxruntime-gpu"
            )
        
        # 추출기 (싱글톤)
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend,
            device=self.config.device,
            mode=self.config.mode,
            to_openpose=True
        )
        
        # 인물 필터
        self.person_filter = PersonFilter(
            area_weight=self.config.area_weight,
            center_weight=self.config.center_weight,
            confidence_threshold=self.config.confidence_threshold
        )
        
        # 본 계산기
        self.bone_calculator = BoneCalculator(
            confidence_threshold=self.config.confidence_threshold
        )
        
        # 방향 추출기
        self.direction_extractor = DirectionExtractor(
            confidence_threshold=self.config.confidence_threshold
        )
        
        # 전이 엔진
        transfer_config = TransferConfig(
            confidence_threshold=self.config.confidence_threshold
        )
        self.transfer_engine = PoseTransferEngine(config=transfer_config)
        
        # 폴백 전략
        self.fallback_strategy = FallbackStrategy(
            confidence_threshold=self.config.confidence_threshold
        )
        
        # 손 정밀화
        self.hand_refiner = HandRefiner(
            min_hand_size=self.config.min_hand_size,
            confidence_threshold=self.config.confidence_threshold
        )
        
        # 렌더러
        self.renderer = SkeletonRenderer(
            line_thickness=self.config.line_thickness,
            point_radius=self.config.point_radius,
            kpt_threshold=self.config.confidence_threshold
        )
    
    def extract_pose(
        self,
        image: Union[np.ndarray, str, Path],
        filter_person: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
        """
        이미지에서 포즈 추출
        
        Args:
            image: 이미지 (배열 또는 경로)
            filter_person: 다중 인물 필터링 여부
        
        Returns:
            keypoints: (K, 2) 키포인트
            scores: (K,) 신뢰도
            selected_idx: 선택된 인물 인덱스
            image_size: (H, W)
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]
        
        # 키포인트 추출
        all_keypoints, all_scores = self.extractor.extract(img)
        
        if len(all_keypoints) == 0:
            # 검출 실패
            return np.zeros((133, 2)), np.zeros(133), -1, image_size
        
        # 다중 인물 필터링
        if filter_person and self.config.filter_enabled and len(all_keypoints) > 1:
            keypoints, scores, selected_idx, _ = self.person_filter.select_main_person(
                all_keypoints, all_scores, image_size
            )
        else:
            keypoints = all_keypoints[0]
            scores = all_scores[0]
            selected_idx = 0
        
        # 손 정밀화
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
        """
        포즈 전이 수행
        
        Args:
            source_image: 원본 이미지 (신체 비율 소스)
            reference_image: 레퍼런스 이미지 (포즈 소스)
            output_image_size: 출력 이미지 크기 (None이면 원본 크기)
        
        Returns:
            PipelineResult: 전이 결과
        """
        # 원본 이미지 로드 및 포즈 추출
        if isinstance(source_image, (str, Path)):
            source_img = load_image(source_image)
        else:
            source_img = source_image
        
        source_kpts, source_scores, source_idx, source_size = self.extract_pose(
            source_img
        )
        
        # 레퍼런스 이미지 로드 및 포즈 추출
        if isinstance(reference_image, (str, Path)):
            ref_img = load_image(reference_image)
        else:
            ref_img = reference_image
        
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # 포즈 전이
        transfer_result = self.transfer_engine.transfer(
            source_kpts, source_scores,
            ref_kpts, ref_scores
        )
        
        # 폴백 적용
        if self.config.fallback_enabled:
            fallback_result = self.fallback_strategy.apply_fallback(
                transfer_result.keypoints,
                transfer_result.scores,
                source_size
            )
            transferred_kpts = fallback_result.keypoints
            transferred_scores = fallback_result.scores
            fallback_info = fallback_result.fallback_applied
        else:
            transferred_kpts = transfer_result.keypoints
            transferred_scores = transfer_result.scores
            fallback_info = {}
        
        # 출력 이미지 크기 결정
        if output_image_size is None:
            output_size = source_size
        else:
            output_size = output_image_size
        
        # 스켈레톤 이미지 렌더링
        skeleton_image = self.renderer.render_skeleton_only(
            (output_size[0], output_size[1], 3),
            transferred_kpts,
            transferred_scores
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
            selected_person_idx={
                'source': source_idx,
                'reference': ref_idx
            },
            processing_info={
                'transfer_log': transfer_result.transfer_log,
                'fallback_applied': fallback_info
            }
        )
    
    def extract_and_render(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:# <--- 리턴 타입 힌트 변경
        """
        단일 이미지에서 포즈 추출 및 렌더링
        
        Args:
            image: 입력 이미지
        
        Returns:
            json_data: OpenPose 호환 JSON
            skeleton_image: 스켈레톤 이미지 (검은 배경)
            overlay_image: 원본 오버레이 이미지 (NEW)
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image
        
        image_size = img.shape[:2]
        
        # 포즈 추출
        keypoints, scores, selected_idx, _ = self.extract_pose(img)
        
        # JSON 변환
        json_data = convert_to_openpose_format(
            keypoints[np.newaxis, ...],
            scores[np.newaxis, ...],
            image_size
        )
        
        # 스켈레톤 렌더링
        skeleton_image = self.renderer.render_skeleton_only(
            (image_size[0], image_size[1], 3),
            keypoints,
            scores
        )

        # 2. 오버레이 렌더링 (원본 위 - 확인용) <--- 추가된 부분
        overlay_image = self.renderer.render(
            img,  # 원본 이미지
            keypoints,
            scores
        )
        
        return json_data, skeleton_image, overlay_image  # <--- 오버레이 이미지 추가 반환251201


# =============================================================================
# 편의 함수 (외부 호출용)
# =============================================================================

_pipeline_instance: Optional[PoseTransferPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> PoseTransferPipeline:
    """파이프라인 인스턴스 반환 (싱글톤)"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PoseTransferPipeline(config)
    return _pipeline_instance


def extract_pose_from_image(
    image: Union[np.ndarray, str, Path]
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    이미지에서 포즈 추출
    
    Args:
        image: 입력 이미지
    
    Returns:
        json_data: OpenPose 호환 JSON
        skeleton_image: 스켈레톤 이미지
    """
    pipeline = get_pipeline()
    return pipeline.extract_and_render(image)


def transfer_pose(
    source_image: Union[np.ndarray, str, Path],
    reference_image: Union[np.ndarray, str, Path]
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    포즈 전이 수행
    
    Args:
        source_image: 원본 이미지 (신체 비율 소스)
        reference_image: 레퍼런스 이미지 (포즈 소스)
    
    Returns:
        json_data: OpenPose 호환 JSON (전이된 포즈)
        skeleton_image: 전이된 스켈레톤 이미지
    """
    pipeline = get_pipeline()
    result = pipeline.transfer(source_image, reference_image)
    
    return result.to_json(), result.skeleton_image
