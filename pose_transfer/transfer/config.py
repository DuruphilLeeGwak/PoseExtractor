from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class FacePartConfig:
    enabled: bool = True
    color: Tuple[int, int, int] = (255, 255, 255)

@dataclass
class FaceRenderingConfig:
    enabled: bool = True
    parts: Dict[str, FacePartConfig] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FaceRenderingConfig':
        config = cls()
        if not data: return config
        config.enabled = data.get('enabled', True)
        parts_data = data.get('parts', {})
        default_parts = ['jawline', 'left_eyebrow', 'right_eyebrow', 'nose', 
                         'left_eye', 'right_eye', 'mouth_outer', 'mouth_inner']
        config.parts = {}
        for part in default_parts:
            p_data = parts_data.get(part, {})
            config.parts[part] = FacePartConfig(
                enabled=p_data.get('enabled', True),
                color=tuple(p_data.get('color', [255, 255, 255]))
            )
        return config

@dataclass
class TransferConfig:
    confidence_threshold: float = 0.3
    use_face: bool = True
    use_hands: bool = True
    enable_symmetric_fallback: bool = True
    
    # 하반신 검증용
    lower_body_confidence_threshold: float = 0.5
    lower_body_margin_ratio: float = 0.05
    
    # [FIX] 누락되었던 필드 추가 (화면 밖 허용 마진)
    visibility_margin: float = 0.15
    
    face_rendering: FaceRenderingConfig = field(default_factory=FaceRenderingConfig)

@dataclass
class TransferResult:
    keypoints: np.ndarray
    scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    reference_directions: Dict[str, np.ndarray]
    transfer_log: Dict[str, str] = field(default_factory=dict)