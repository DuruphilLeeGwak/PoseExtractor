"""
Logic Module - 정렬, BBox, 후처리, Ghost Filter 등
"""
from .bbox_manager import (
    BboxManager, BboxInfo, DebugBboxData, 
    COLOR_KPT_BBOX, COLOR_YOLO_BBOX, COLOR_HYBRID_PERSON, COLOR_HYBRID_FACE
)
from .align_manager import AlignManager, AlignmentCase, BodyType
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
    'AlignmentCase', 
    'BodyType',
    # Post
    'PostProcessor',
    # Canvas
    'CanvasManager',
    # Ghost Filter v3.0
    'GhostFilter',
    'GhostFilterConfig',
    'FilterResult',
    'create_ghost_filter',
    'filter_keypoints'
]