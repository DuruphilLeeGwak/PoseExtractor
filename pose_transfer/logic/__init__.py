"""
Logic Module - 정렬, BBox, 후처리, Cross-Filter 등 (Refactored v3.0)

변경사항:
- AlignmentCase, BodyType Enum 제거 (단순화)
- Ghost Filter 제거 (Cross-Filter로 통합)
"""
from .bbox_manager import (
    BboxManager, BboxInfo, DebugBboxData, 
    COLOR_KPT_BBOX, COLOR_YOLO_BBOX, COLOR_HYBRID_PERSON, COLOR_HYBRID_FACE
)
from .align_manager import AlignManager
from .post_processor import PostProcessor
from .canvas_manager import CanvasManager

# [v4.0] Cross Filter - Body + DWPose 교차 필터링
from .cross_filter import CrossFilter, CrossFilterConfig

# [v3.2] Keypoint Generator - 누락 키포인트 생성
from .keypoint_generator import KeypointGenerator

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
    # Cross Filter v4.0
    'CrossFilter',
    'CrossFilterConfig',
    # Keypoint Generator v3.2
    'KeypointGenerator',
]