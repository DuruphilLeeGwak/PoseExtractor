"""
rtmlib Body 모델 기반 키포인트 추출기

Body 모델은 COCO 17 keypoints만 추출하여
"존재 유무" 판단에 특화되어 있습니다.

교차 필터링(Cross-Filtering)에서 감시자(Validator) 역할을 수행합니다.
"""
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

try:
    from rtmlib import Body
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False
    print("Warning: rtmlib not installed. Run: pip install rtmlib onnxruntime-gpu")


# 디버그 플래그 (default: False)
import yaml
import os

def _load_body_debug_flags():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'default.yaml')
    flags = {
        'show_body_init': False,
        'show_body_clean_mode': False,
        'show_body_cross_filter': False
    }
    if not os.path.exists(config_path):
        return flags
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        debug = config.get('debug', {})
        flags['show_body_init'] = debug.get('show_body_init', False)
        flags['show_body_clean_mode'] = debug.get('show_body_clean_mode', False)
        flags['show_body_cross_filter'] = debug.get('show_body_cross_filter', False)
    except Exception:
        pass
    return flags

BODY_DEBUG_FLAGS = _load_body_debug_flags()


class BodyExtractor:
    """
    rtmlib Body 모델 래퍼 (COCO 17 keypoints)
    
    교차 필터링에서 "존재 유무" 판단을 위한 가벼운 모델입니다.
    DWPose Wholebody가 환각(Hallucination)을 일으킬 때 이를 감지합니다.
    
    COCO 17 Keypoints:
        0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
        5: L-Shoulder, 6: R-Shoulder
        7: L-Elbow, 8: R-Elbow
        9: L-Wrist, 10: R-Wrist
        11: L-Hip, 12: R-Hip
        13: L-Knee, 14: R-Knee
        15: L-Ankle, 16: R-Ankle
    """
    
    def __init__(
        self,
        backend: str = 'onnxruntime',
        device: str = 'cuda',
        mode: str = 'balanced'  # Body는 가벼워서 balanced로 충분
    ):
        """
        Body 추출기 초기화
        
        Args:
            backend: 추론 백엔드 ('onnxruntime')
            device: 디바이스 ('cuda' or 'cpu')
            mode: 성능 모드 ('lightweight', 'balanced', 'performance')
        """
        if not RTMLIB_AVAILABLE:
            raise RuntimeError("rtmlib is not installed")
        
        self.backend = backend
        self.device = device
        self.mode = mode
        
        self._init_model()
    
    def _init_model(self):
        """Body 모델 초기화"""
        if BODY_DEBUG_FLAGS['show_body_init']:
            print(f"Initializing Body model (COCO 17 keypoints)...")
            print(f"  Backend: {self.backend}")
            print(f"  Device: {self.device}")
            print(f"  Mode: {self.mode}")
        
        self.model = Body(
            mode=self.mode,
            backend=self.backend,
            device=self.device
        )
        if BODY_DEBUG_FLAGS['show_body_init']:
            print("Body model initialized!")
    
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 Body 키포인트 추출 (COCO 17)
        
        Args:
            image: 입력 이미지 (BGR)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - keypoints: (N, 17, 2) 좌표
                - scores: (N, 17) 신뢰도
        """
        keypoints, scores = self.model(image)
        
        if keypoints is None or len(keypoints) == 0:
            return np.array([]), np.array([])
        
        keypoints = np.array(keypoints)
        scores = np.array(scores)
        
        return keypoints, scores
    
    def extract_single(self, image: np.ndarray, person_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 사람 추출
        
        Args:
            image: 입력 이미지
            person_idx: 사람 인덱스
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - keypoints: (17, 2)
                - scores: (17,)
        """
        keypoints, scores = self.extract(image)
        
        if len(keypoints) == 0:
            return np.zeros((17, 2)), np.zeros(17)
        
        if person_idx >= len(keypoints):
            person_idx = 0
        
        return keypoints[person_idx], scores[person_idx]
