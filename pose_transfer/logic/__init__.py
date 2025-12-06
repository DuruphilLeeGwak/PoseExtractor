from .bbox_manager import (
    BboxManager, BboxInfo, DebugBboxData, 
    COLOR_KPT_BBOX, COLOR_YOLO_BBOX, COLOR_HYBRID_PERSON, COLOR_HYBRID_FACE
)
from .align_manager import AlignManager, AlignmentCase, BodyType
from .post_processor import PostProcessor
from .canvas_manager import CanvasManager  # [NEW] 추가됨