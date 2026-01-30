"""
Transfer Config Module (Refactored v4.5)

위치: pose_transfer/transfer/config.py
역할: 전이(Transfer) 관련 설정 클래스 정의 (Engine에서 분리됨)
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TransferConfig:
    confidence_threshold: float = 0.3
    use_face: bool = True
    use_hands: bool = True
    enable_symmetric_fallback: bool = True
    visibility_margin: float = 0.2
    enable_upper_ratio_tuning: bool = True
    enable_lower_ratio_tuning: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """YAML 설정을 Config 객체로 변환"""
        if not data:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        # YAML의 키 중 Config에 정의된 것만 필터링하여 주입
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)