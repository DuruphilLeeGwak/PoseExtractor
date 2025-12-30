"""
DWPose 기반 키포인트 추출기

이 모듈은 rtmlib의 Wholebody 모델을 사용하여
COCO-WholeBody 형식(133개 키포인트)으로 전신 포즈를 추출합니다.
- Body(17) + Face(68) + Hands(42) + Feet(6) = 133 keypoints
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

# rtmlib 라이브러리 임포트 시도 (선택적 의존성)
try:
    from rtmlib import Wholebody, draw_skeleton
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False
    print("Warning: rtmlib not installed. Run: pip install rtmlib onnxruntime-gpu")


class DWPoseExtractor:
    """
    DWPose/RTMPose 기반 Wholebody 키포인트 추출기
    
    전신 포즈(몸, 얼굴, 손, 발)를 133개 키포인트로 추출합니다.
    COCO-WholeBody 형식을 사용하며, OpenPose 형식으로 변환하지 않습니다.
    
    Attributes:
        backend (str): 추론 백엔드 ('onnxruntime' 또는 기타)
        device (str): 실행 디바이스 ('cuda' 또는 'cpu')
        mode (str): 성능 모드 ('performance', 'lightweight', 'balanced')
        to_openpose (bool): 외부 설정용 플래그 (실제로는 항상 False 사용)
        model: rtmlib의 Wholebody 모델 인스턴스
    """
    
    def __init__(
        self,
        backend: str = 'onnxruntime',
        device: str = 'cuda',
        mode: str = 'performance',
        to_openpose: bool = True  # 외부 설정용 (내부에서는 False 사용)
    ):
        """
        DWPose 추출기 초기화
        
        Args:
            backend: 추론 백엔드 엔진 (기본: 'onnxruntime')
            device: GPU/CPU 선택 (기본: 'cuda')
            mode: 모델 성능 모드 (기본: 'performance')
            to_openpose: 외부 API 호환성을 위한 파라미터 (실제로는 무시됨)
        
        Raises:
            RuntimeError: rtmlib가 설치되지 않은 경우
        """
        # rtmlib 설치 여부 확인
        if not RTMLIB_AVAILABLE:
            raise RuntimeError("rtmlib is not installed")
        
        # 설정 저장
        self.backend = backend
        self.device = device
        self.mode = mode
        self.to_openpose = to_openpose
        
        # 모델 초기화
        self._init_model()
    
    def _init_model(self):
        """
        Wholebody 모델 초기화
        
        중요: to_openpose=False로 고정하여 COCO-WholeBody 원본 형식(133 keypoints)을 사용합니다.
        OpenPose 형식(135 keypoints)으로 변환하지 않음으로써 키포인트 인덱스 혼란을 방지합니다.
        """
        print(f"Initializing DWPose model...")
        print(f"  Backend: {self.backend}")
        print(f"  Device: {self.device}")
        print(f"  Mode: {self.mode}")
        
        # 핵심: to_openpose=False로 COCO-WholeBody 원본 형식 사용
        # 이렇게 하면 133개 키포인트가 표준 COCO-WholeBody 인덱스를 따릅니다
        self.model = Wholebody(
            to_openpose=False,  # ← 항상 False! (OpenPose 형식 변환 비활성화)
            mode=self.mode,
            backend=self.backend,
            device=self.device
        )
        print("Model initialized successfully!")
    
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 모든 사람의 키포인트 추출
        
        Args:
            image: 입력 이미지 (numpy 배열, BGR 형식)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - keypoints: (N, 133, 2) 형태의 키포인트 좌표 배열
                  N은 검출된 사람의 수, 133은 키포인트 개수, 2는 (x, y)
                - scores: (N, 133) 형태의 신뢰도 점수 배열
                  각 키포인트의 검출 신뢰도 (0.0 ~ 1.0)
        
        Note:
            - 사람이 검출되지 않으면 빈 배열 반환
            - 좌표는 이미지 범위 내로 클리핑됨
        """
        # 이미지 크기 추출
        img_h, img_w = image.shape[:2]
        
        # 모델 추론 (여러 사람 검출 가능)
        keypoints, scores = self.model(image)
        
        # 검출된 사람이 없는 경우 빈 배열 반환
        if keypoints is None or len(keypoints) == 0:
            return np.array([]), np.array([])
        
        # 리스트를 numpy 배열로 변환
        keypoints = np.array(keypoints)
        scores = np.array(scores)
        
        # 좌표를 이미지 범위 내로 클리핑 (경계 밖으로 나간 키포인트 보정)
        keypoints[..., 0] = np.clip(keypoints[..., 0], 0, img_w - 1)  # x 좌표
        keypoints[..., 1] = np.clip(keypoints[..., 1], 0, img_h - 1)  # y 좌표
        
        return keypoints, scores
    
    def extract_single(self, image: Union[np.ndarray, str, Path], person_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 단일 사람의 키포인트 추출
        
        여러 사람이 검출되더라도 지정된 인덱스의 사람만 반환합니다.
        
        Args:
            image: 입력 이미지 (numpy 배열 또는 파일 경로)
            person_idx: 추출할 사람의 인덱스 (기본: 0, 첫 번째 사람)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - keypoints: (133, 2) 형태의 키포인트 좌표 배열
                - scores: (133,) 형태의 신뢰도 점수 배열
        
        Note:
            - 사람이 검출되지 않으면 모두 0인 배열 반환
            - person_idx가 범위를 벗어나면 첫 번째 사람(idx=0) 반환
        """
        # 전체 사람의 키포인트 추출
        keypoints, scores = self.extract(image)
        
        # 아무도 검출되지 않은 경우 빈 키포인트 반환
        if len(keypoints) == 0:
            return np.zeros((133, 2)), np.zeros(133)
        
        # 인덱스가 범위를 벗어나면 첫 번째 사람으로 폴백
        if person_idx >= len(keypoints):
            person_idx = 0
        
        # 지정된 사람의 데이터만 반환
        return keypoints[person_idx], scores[person_idx]
    
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.3) -> np.ndarray:
        """
        원본 이미지 위에 스켈레톤 시각화
        
        Args:
            image: 원본 이미지 (배경으로 사용)
            keypoints: (133, 2) 형태의 키포인트 좌표
            scores: (133,) 형태의 신뢰도 점수
            kpt_thr: 키포인트 표시 임계값 (이 값 이상의 신뢰도만 표시)
        
        Returns:
            np.ndarray: 스켈레톤이 그려진 이미지
        
        Note:
            - 원본 이미지를 복사하여 수정하므로 원본은 유지됨
            - rtmlib의 draw_skeleton 함수 사용
        """
        # 원본 이미지 보존을 위해 복사
        img_show = image.copy()
        # rtmlib의 스켈레톤 그리기 함수 호출
        img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=kpt_thr)
        return img_show
    
    def draw_skeleton_only(self, image_shape: Tuple[int, int, int], keypoints: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.3) -> np.ndarray:
        """
        검은 배경에 스켈레톤만 시각화
        
        원본 이미지 없이 스켈레톤만 표시할 때 사용합니다.
        
        Args:
            image_shape: 캔버스 크기 (height, width, channels)
            keypoints: (133, 2) 형태의 키포인트 좌표
            scores: (133,) 형태의 신뢰도 점수
            kpt_thr: 키포인트 표시 임계값
        
        Returns:
            np.ndarray: 검은 배경에 스켈레톤이 그려진 이미지
        
        Note:
            - 포즈만 시각화하고 싶을 때 유용 (배경 제거)
        """
        # 검은 캔버스 생성
        canvas = np.zeros(image_shape, dtype=np.uint8)
        # 스켈레톤 그리기
        canvas = draw_skeleton(canvas, keypoints, scores, kpt_thr=kpt_thr)
        return canvas


class DWPoseExtractorFactory:
    """
    DWPoseExtractor 싱글톤 팩토리
    
    모델 로딩은 시간이 오래 걸리므로, 동일한 설정이면 재사용합니다.
    메모리 효율성과 성능 향상을 위한 싱글톤 패턴 구현.
    
    Class Attributes:
        _instance: 싱글톤 인스턴스
        _config: 현재 인스턴스의 설정
    """
    _instance: Optional[DWPoseExtractor] = None
    _config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get_instance(cls, backend: str = 'onnxruntime', device: str = 'cuda', 
                     mode: str = 'performance', to_openpose: bool = True, 
                     force_new: bool = False) -> DWPoseExtractor:
        """
        DWPoseExtractor 인스턴스 가져오기
        
        동일한 설정이면 기존 인스턴스를 재사용하고,
        설정이 변경되었거나 force_new=True이면 새로 생성합니다.
        
        Args:
            backend: 추론 백엔드
            device: 실행 디바이스
            mode: 성능 모드
            to_openpose: OpenPose 형식 변환 여부 (실제로는 항상 False 사용)
            force_new: True면 기존 인스턴스를 무시하고 새로 생성
        
        Returns:
            DWPoseExtractor: 추출기 인스턴스 (재사용 또는 새 생성)
        """
        # 현재 요청된 설정
        new_config = {'backend': backend, 'device': device, 'mode': mode, 'to_openpose': to_openpose}
        
        # 새 인스턴스 생성 조건: force_new이거나, 인스턴스가 없거나, 설정이 변경됨
        if force_new or cls._instance is None or cls._config != new_config:
            cls._instance = DWPoseExtractor(**new_config)
            cls._config = new_config
        
        # 기존 또는 새로 생성된 인스턴스 반환
        return cls._instance
    
    @classmethod
    def release(cls):
        """
        싱글톤 인스턴스 해제
        
        메모리 정리가 필요할 때 호출합니다.
        다음 get_instance() 호출 시 새 인스턴스가 생성됩니다.
        """
        cls._instance = None
        cls._config = None


def extract_pose(image: Union[np.ndarray, str, Path], backend: str = 'onnxruntime',
                 device: str = 'cuda', mode: str = 'performance') -> Tuple[np.ndarray, np.ndarray]:
    """
    편의 함수: 이미지에서 포즈 추출
    
    팩토리를 통해 추출기를 가져와 포즈를 추출합니다.
    간단한 사용을 위한 고수준 API입니다.
    
    Args:
        image: 입력 이미지 (numpy 배열 또는 파일 경로)
        backend: 추론 백엔드 (기본: 'onnxruntime')
        device: 실행 디바이스 (기본: 'cuda')
        mode: 성능 모드 (기본: 'performance')
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - keypoints: (N, 133, 2) 형태의 키포인트 좌표
            - scores: (N, 133) 형태의 신뢰도 점수
    
    Example:
        >>> keypoints, scores = extract_pose('person.jpg')
        >>> print(keypoints.shape)  # (1, 133, 2) - 1명의 사람, 133개 키포인트
    """
    # 팩토리에서 추출기 가져오기 (재사용 또는 새 생성)
    extractor = DWPoseExtractorFactory.get_instance(backend=backend, device=device, mode=mode)
    return extractor.extract(image)


def draw_pose(image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
              kpt_thr: float = 0.3, black_background: bool = False) -> np.ndarray:
    """
    편의 함수: 포즈 시각화
    
    원본 이미지 위에 또는 검은 배경에 스켈레톤을 그립니다.
    
    Args:
        image: 원본 이미지 (배경으로 사용)
        keypoints: (133, 2) 형태의 키포인트 좌표
        scores: (133,) 형태의 신뢰도 점수
        kpt_thr: 키포인트 표시 임계값 (기본: 0.3)
        black_background: True면 검은 배경에만 스켈레톤 표시
    
    Returns:
        np.ndarray: 스켈레톤이 그려진 이미지
    
    Example:
        >>> keypoints, scores = extract_pose('person.jpg')
        >>> skeleton_img = draw_pose(image, keypoints[0], scores[0])
        >>> skeleton_only = draw_pose(image, keypoints[0], scores[0], black_background=True)
    """
    # 팩토리에서 추출기 가져오기 (설정 없이 호출하면 기존 인스턴스 재사용)
    extractor = DWPoseExtractorFactory.get_instance()
    
    # 검은 배경 모드 선택
    if black_background:
        return extractor.draw_skeleton_only(image.shape, keypoints, scores, kpt_thr)
    return extractor.draw_skeleton(image, keypoints, scores, kpt_thr)
