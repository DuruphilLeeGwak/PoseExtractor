"""
Pose Transfer Package Init (Refactored v4.6 - Preserve & Fix)

위치: pose_transfer/transfer/__init__.py
"""
# [Config] 설정 클래스
# FaceRenderingConfig가 config.py에 없다면 이 줄에서 에러가 날 수 있습니다.
# 만약 에러가 난다면 FaceRenderingConfig는 지워주세요 (현재 리팩토링 코드에는 포함되지 않았습니다).
try:
    from .config import TransferConfig, FaceRenderingConfig
except ImportError:
    from .config import TransferConfig
    # FaceRenderingConfig가 삭제되었다면 패스

# [Engine] 엔진 클래스
from .engine import PoseTransferEngine

# [TransferResult] 결과 데이터 (utils/io.py에 있음)
from ..utils.io import TransferResult

# [Fallback] 폴백 전략 (기존 파일 유지)
from .fallback import FallbackStrategy, FallbackResult, apply_fallback

__all__ = [
    'TransferConfig',
    'TransferResult',
    'PoseTransferEngine',
    'FallbackStrategy',
    'FallbackResult',
    'apply_fallback'
]