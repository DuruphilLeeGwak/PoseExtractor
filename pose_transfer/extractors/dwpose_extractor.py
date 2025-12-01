"""
DWPose 기반 키포인트 추출기
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
    """DWPose/RTMPose 기반 Wholebody 키포인트 추출기"""
    
    def __init__(
        self,
        backend: str = 'onnxruntime',
        device: str = 'cuda',
        mode: str = 'performance',
        to_openpose: bool = True  # 외부 설정용 (내부에서는 False 사용)
    ):
        if not RTMLIB_AVAILABLE:
            raise RuntimeError("rtmlib is not installed")
        
        self.backend = backend
        self.device = device
        self.mode = mode
        self.to_openpose = to_openpose
        self._init_model()
    
    def _init_model(self):
        print(f"Initializing DWPose model...")
        print(f"  Backend: {self.backend}")
        print(f"  Device: {self.device}")
        print(f"  Mode: {self.mode}")
        
        # 핵심: to_openpose=False로 COCO-WholeBody 원본 형식 사용
        self.model = Wholebody(
            to_openpose=False,  # ← 항상 False!
            mode=self.mode,
            backend=self.backend,
            device=self.device
        )
        print("Model initialized successfully!")
    
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """키포인트 추출"""
        img_h, img_w = image.shape[:2]
        
        keypoints, scores = self.model(image)
        
        if keypoints is None or len(keypoints) == 0:
            return np.array([]), np.array([])
        
        keypoints = np.array(keypoints)
        scores = np.array(scores)
        
        # 좌표 클리핑
        keypoints[..., 0] = np.clip(keypoints[..., 0], 0, img_w - 1)
        keypoints[..., 1] = np.clip(keypoints[..., 1], 0, img_h - 1)
        
        return keypoints, scores
    
    def extract_single(self, image: Union[np.ndarray, str, Path], person_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        keypoints, scores = self.extract(image)
        
        if len(keypoints) == 0:
            return np.zeros((133, 2)), np.zeros(133)
        
        if person_idx >= len(keypoints):
            person_idx = 0
        
        return keypoints[person_idx], scores[person_idx]
    
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.3) -> np.ndarray:
        img_show = image.copy()
        img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=kpt_thr)
        return img_show
    
    def draw_skeleton_only(self, image_shape: Tuple[int, int, int], keypoints: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.3) -> np.ndarray:
        canvas = np.zeros(image_shape, dtype=np.uint8)
        canvas = draw_skeleton(canvas, keypoints, scores, kpt_thr=kpt_thr)
        return canvas


class DWPoseExtractorFactory:
    _instance: Optional[DWPoseExtractor] = None
    _config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get_instance(cls, backend: str = 'onnxruntime', device: str = 'cuda', 
                     mode: str = 'performance', to_openpose: bool = True, 
                     force_new: bool = False) -> DWPoseExtractor:
        new_config = {'backend': backend, 'device': device, 'mode': mode, 'to_openpose': to_openpose}
        
        if force_new or cls._instance is None or cls._config != new_config:
            cls._instance = DWPoseExtractor(**new_config)
            cls._config = new_config
        
        return cls._instance
    
    @classmethod
    def release(cls):
        cls._instance = None
        cls._config = None


def extract_pose(image: Union[np.ndarray, str, Path], backend: str = 'onnxruntime',
                 device: str = 'cuda', mode: str = 'performance') -> Tuple[np.ndarray, np.ndarray]:
    extractor = DWPoseExtractorFactory.get_instance(backend=backend, device=device, mode=mode)
    return extractor.extract(image)


def draw_pose(image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
              kpt_thr: float = 0.3, black_background: bool = False) -> np.ndarray:
    extractor = DWPoseExtractorFactory.get_instance()
    if black_background:
        return extractor.draw_skeleton_only(image.shape, keypoints, scores, kpt_thr)
    return extractor.draw_skeleton(image, keypoints, scores, kpt_thr)
