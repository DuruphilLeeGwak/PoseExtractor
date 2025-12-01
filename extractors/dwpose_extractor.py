"""
DWPose 기반 키포인트 추출기

rtmlib를 사용하여 COCO-WholeBody 133 키포인트를 추출합니다.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

try:
    from rtmlib import Wholebody, draw_skeleton
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False
    print("Warning: rtmlib not installed. Run: pip install rtmlib onnxruntime-gpu")


class DWPoseExtractor:
    """
    DWPose/RTMPose 기반 Wholebody 키포인트 추출기
    
    133 키포인트 구성:
    - Body: 17개 (0-16)
    - Feet: 6개 (17-22)
    - Face: 68개 (23-90)
    - Left Hand: 21개 (91-111)
    - Right Hand: 21개 (112-132)
    """
    
    def __init__(
        self,
        backend: str = 'onnxruntime',
        device: str = 'cuda',
        mode: str = 'performance',
        to_openpose: bool = True
    ):
        """
        Args:
            backend: 'onnxruntime', 'opencv', 'openvino'
            device: 'cpu', 'cuda', 'mps'
            mode: 'performance', 'balanced', 'lightweight'
            to_openpose: OpenPose 스타일 스켈레톤 출력 여부
        """
        if not RTMLIB_AVAILABLE:
            raise RuntimeError("rtmlib is not installed")
        
        self.backend = backend
        self.device = device
        self.mode = mode
        self.to_openpose = to_openpose
        
        # 모델 초기화
        self._init_model()
    
    def _init_model(self):
        """모델 초기화 (싱글톤 패턴으로 한 번만 로드)"""
        print(f"Initializing DWPose model...")
        print(f"  Backend: {self.backend}")
        print(f"  Device: {self.device}")
        print(f"  Mode: {self.mode}")
        
        self.model = Wholebody(
            to_openpose=self.to_openpose,
            mode=self.mode,
            backend=self.backend,
            device=self.device
        )
        print("Model initialized successfully!")
    
    def extract(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 키포인트 추출
        
        Args:
            image: BGR 이미지 배열 또는 이미지 경로
        
        Returns:
            keypoints: (N, 133, 2) 또는 (N, 118, 2) if to_openpose
            scores: (N, 133) 또는 (N, 118)
            
            N = 검출된 인물 수
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot load image: {image}")
        else:
            img = image
        
        # 추론
        keypoints, scores = self.model(img)
        
        return keypoints, scores
    
    def extract_single(
        self,
        image: Union[np.ndarray, str, Path],
        person_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 인물의 키포인트 추출
        
        Args:
            image: BGR 이미지 배열 또는 이미지 경로
            person_idx: 추출할 인물 인덱스
        
        Returns:
            keypoints: (133, 2)
            scores: (133,)
        """
        keypoints, scores = self.extract(image)
        
        if len(keypoints) == 0:
            # 검출된 인물이 없으면 빈 배열 반환
            return np.zeros((133, 2)), np.zeros(133)
        
        if person_idx >= len(keypoints):
            person_idx = 0
        
        return keypoints[person_idx], scores[person_idx]
    
    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.3,
        line_thickness: int = 4,
        point_radius: int = 4
    ) -> np.ndarray:
        """
        이미지에 스켈레톤 그리기
        
        Args:
            image: BGR 이미지
            keypoints: (N, K, 2) 또는 (K, 2)
            scores: (N, K) 또는 (K,)
            kpt_thr: 키포인트 신뢰도 임계값
            line_thickness: 선 굵기
            point_radius: 점 반지름
        
        Returns:
            스켈레톤이 그려진 이미지
        """
        img_show = image.copy()
        
        # rtmlib의 draw_skeleton 사용
        img_show = draw_skeleton(
            img_show,
            keypoints,
            scores,
            kpt_thr=kpt_thr
        )
        
        return img_show
    
    def draw_skeleton_only(
        self,
        image_shape: Tuple[int, int, int],
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.3
    ) -> np.ndarray:
        """
        검은 배경에 스켈레톤만 그리기 (ControlNet용)
        
        Args:
            image_shape: (H, W, C) 이미지 형태
            keypoints: (N, K, 2) 또는 (K, 2)
            scores: (N, K) 또는 (K,)
            kpt_thr: 키포인트 신뢰도 임계값
        
        Returns:
            검은 배경의 스켈레톤 이미지
        """
        # 검은 배경 생성
        canvas = np.zeros(image_shape, dtype=np.uint8)
        
        # 스켈레톤 그리기
        canvas = draw_skeleton(
            canvas,
            keypoints,
            scores,
            kpt_thr=kpt_thr
        )
        
        return canvas
    
    def get_image_size(self, image: Union[np.ndarray, str, Path]) -> Tuple[int, int]:
        """이미지 크기 반환 (height, width)"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            return img.shape[:2]
        return image.shape[:2]


class DWPoseExtractorFactory:
    """DWPose 추출기 팩토리 (싱글톤 패턴)"""
    
    _instance: Optional[DWPoseExtractor] = None
    _config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get_instance(
        cls,
        backend: str = 'onnxruntime',
        device: str = 'cuda',
        mode: str = 'performance',
        to_openpose: bool = True,
        force_new: bool = False
    ) -> DWPoseExtractor:
        """
        DWPoseExtractor 인스턴스 반환 (싱글톤)
        
        Args:
            force_new: True면 기존 인스턴스 무시하고 새로 생성
        """
        new_config = {
            'backend': backend,
            'device': device,
            'mode': mode,
            'to_openpose': to_openpose
        }
        
        # 설정이 변경되었거나 인스턴스가 없으면 새로 생성
        if force_new or cls._instance is None or cls._config != new_config:
            cls._instance = DWPoseExtractor(**new_config)
            cls._config = new_config
        
        return cls._instance
    
    @classmethod
    def release(cls):
        """인스턴스 해제"""
        cls._instance = None
        cls._config = None


# 편의 함수
def extract_pose(
    image: Union[np.ndarray, str, Path],
    backend: str = 'onnxruntime',
    device: str = 'cuda',
    mode: str = 'performance'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    간편 포즈 추출 함수
    
    Returns:
        keypoints: (N, 133, 2)
        scores: (N, 133)
    """
    extractor = DWPoseExtractorFactory.get_instance(
        backend=backend,
        device=device,
        mode=mode
    )
    return extractor.extract(image)


def draw_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = 0.3,
    black_background: bool = False
) -> np.ndarray:
    """
    간편 스켈레톤 그리기 함수
    """
    extractor = DWPoseExtractorFactory.get_instance()
    
    if black_background:
        return extractor.draw_skeleton_only(
            image.shape, keypoints, scores, kpt_thr
        )
    else:
        return extractor.draw_skeleton(
            image, keypoints, scores, kpt_thr
        )
