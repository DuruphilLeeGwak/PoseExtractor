"""
Logic Module - 정렬, BBox, 후처리, Ghost Filter 등 (Refactored v2.0)

변경사항:
- AlignmentCase, BodyType Enum 제거 (단순화)
"""
from .bbox_manager import (
    BboxManager, BboxInfo, DebugBboxData, 
    COLOR_KPT_BBOX, COLOR_YOLO_BBOX, COLOR_HYBRID_PERSON, COLOR_HYBRID_FACE
)
from .align_manager import AlignManager
from .post_processor import PostProcessor
from .canvas_manager import CanvasManager

# [v3.0] Ghost Filter - 새로운 통합 필터
from .ghost_filter import (
    GhostFilter,
    GhostFilterConfig,
    FilterResult,
    create_ghost_filter,
    filter_keypoints
)
from .cross_filter import CrossFilter, CrossFilterConfig

# [v3.2] Keypoint Generator - 누락 키포인트 생성
from .keypoint_generator import KeypointGenerator

# [v4.0] Cross Filter - Body + DWPose 교차 필터링
from .cross_filter import CrossFilter, CrossFilterConfig

__all__ = [
    # Bbox
    'BboxManager', 
    'BboxInfo', 
    'DebugBboxData',
    'COLOR_KPT_BBOX', 
    'COLOR_YOLO_BBOX', 
    'COLOR_HYBRID_PERSON', 
    'COLOR_HYBRID_FACE',
    # Align
    'AlignManager', 
    # Post
    'PostProcessor',
    # Canvas
    'CanvasManager',
    # Ghost Filter v3.0
    'GhostFilter',
    'GhostFilterConfig',
    'FilterResult',
    'create_ghost_filter',
    # Cross Filter v4.0
    'CrossFilter',
    'CrossFilterConfig',
    'filter_keypoints',
    # Keypoint Generator v3.2
    'KeypointGenerator',
]