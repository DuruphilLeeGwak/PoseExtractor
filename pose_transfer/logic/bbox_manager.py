"""
Bbox Manager Module (Refactored v3.1 - Constants Fixed)

위치: pose_transfer/logic/bbox_manager.py
변경사항:
- [Fix] 외부에서 참조하는 색상 상수(COLOR_HYBRID_PERSON 등) 누락 복구
- Skull Logic(두개골 확장) 포함 유지
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# =========================================================
# [Color Constants] 외부 참조용 상수 복구
# =========================================================
COLOR_KPT_BBOX = (0, 255, 0)        # Green
COLOR_YOLO_BBOX = (0, 0, 255)       # Red
COLOR_HYBRID_PERSON = (255, 255, 0) # Cyan (KPT + YOLO Combined)
COLOR_HYBRID_FACE = (255, 0, 255)   # Magenta

@dataclass
class BboxInfo:
    x1: int
    y1: int
    x2: int
    y2: int
    center: Tuple[int, int]
    
    # AlignManager 필수 정보
    has_lower_body: bool = False
    has_face: bool = False
    feet_center: Tuple[int, int] = (0, 0)
    face_center: Tuple[int, int] = (0, 0)

    @property
    def width(self) -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    def to_tuple(self) -> Tuple[int, int, int, int]: return (self.x1, self.y1, self.x2, self.y2)

@dataclass
class DebugBboxData:
    kpt_person: Optional[Tuple[int, int, int, int]] = None
    yolo_person: Optional[Tuple[int, int, int, int]] = None
    kpt_face: Optional[Tuple[int, int, int, int]] = None
    # Hybrid 정보가 필요하다면 추가 가능

class BboxManager:
    def __init__(self, config):
        self.config = config
        self.person_model = None # YOLO는 현재 비활성화 (필요시 추가)

    def get_bboxes(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[BboxInfo, BboxInfo, DebugBboxData]:
        h, w = image.shape[:2]
        debug_data = DebugBboxData()
        
        # 1. Face BBox 계산 (Skull 보정 포함)
        face_bbox = self._kpt_to_face(keypoints, scores, (h, w), margin=self.config.face_bbox_margin)
        
        # 2. Person BBox 계산 (Face BBox의 Top을 반영하여 보정)
        person_bbox = self._kpt_to_person(keypoints, scores, (h, w), face_bbox_top=face_bbox.y1, margin=self.config.person_bbox_margin)
        
        debug_data.kpt_person = person_bbox.to_tuple()
        debug_data.kpt_face = face_bbox.to_tuple()
        
        return person_bbox, face_bbox, debug_data

    def _kpt_to_face(self, kpts, scores, img_size, margin=0.0) -> BboxInfo:
        """얼굴 BBox 계산 (이마/두개골 확장 로직 적용)"""
        H, W = img_size
        
        # 얼굴 키포인트: 0(코), 1,2(눈), 3,4(귀)
        face_indices = [0, 1, 2, 3, 4]
        valid_pts = []
        for i in face_indices:
            if i < len(scores) and scores[i] > 0.1:
                valid_pts.append(kpts[i])
        
        if not valid_pts:
            return BboxInfo(0, 0, 0, 0, (0, 0))

        valid_pts = np.array(valid_pts)
        x1, y1 = np.min(valid_pts, axis=0)
        x2, y2 = np.max(valid_pts, axis=0)
        
        # --- [Skull Logic] 두개골 확장 ---
        face_width = x2 - x1
        skull_extension = face_width * 0.6 # 이마 높이 추정
        
        y1_skull = y1 - skull_extension
        y1 = int(max(0, y1_skull))
        
        # Margin 적용
        w_box, h_box = x2 - x1, y2 - y1
        pad_x = w_box * margin
        pad_y = h_box * margin
        
        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(W, x2 + pad_x))
        y2 = int(min(H, y2 + pad_y))
        
        center = (int((x1+x2)/2), int((y1+y2)/2))
        
        return BboxInfo(x1, y1, x2, y2, center, has_face=True, face_center=center)

    def _kpt_to_person(self, kpts, scores, img_size, face_bbox_top=None, margin=0.0) -> BboxInfo:
        """전신 BBox 계산"""
        H, W = img_size
        valid_mask = scores > 0.1
        
        if not np.any(valid_mask):
            return BboxInfo(0, 0, W, H, (W//2, H//2))
            
        valid_kpts = kpts[valid_mask]
        x1, y1 = np.min(valid_kpts, axis=0)
        x2, y2 = np.max(valid_kpts, axis=0)
        
        # [Skull Logic] 키포인트 최상단보다 Face BBox의 상단(두개골)이 더 높다면 교체
        if face_bbox_top is not None and face_bbox_top < y1:
            y1 = face_bbox_top
            
        # Margin
        w_box, h_box = x2 - x1, y2 - y1
        pad_x = w_box * margin
        pad_y = h_box * margin
        
        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(W, x2 + pad_x))
        y2 = int(min(H, y2 + pad_y))
        
        center = (int((x1+x2)/2), int((y1+y2)/2))
        
        # 신체 정보 분석
        has_lower = False
        feet_pts = []
        for idx in [15, 16, 19, 22]: # 발목, 발뒷꿈치
            if idx < len(scores) and scores[idx] > 0.2:
                has_lower = True
                feet_pts.append(kpts[idx])
        
        if feet_pts:
            fc = np.mean(feet_pts, axis=0)
            feet_center = (int(fc[0]), int(fc[1]))
        else:
            feet_center = (center[0], y2)
            
        # 얼굴 존재 여부
        has_face = False
        face_pts = []
        for idx in range(5):
            if idx < len(scores) and scores[idx] > 0.2:
                has_face = True
                face_pts.append(kpts[idx])
                
        if face_pts:
            fc_c = np.mean(face_pts, axis=0)
            face_center = (int(fc_c[0]), int(fc_c[1]))
        else:
            face_center = (center[0], y1)

        return BboxInfo(
            x1, y1, x2, y2, center,
            has_lower_body=has_lower,
            has_face=has_face,
            feet_center=feet_center,
            face_center=face_center
        )

    def draw_debug(self, image: np.ndarray, debug_data: DebugBboxData) -> np.ndarray:
        vis = image.copy()
        if debug_data.kpt_person:
            x1, y1, x2, y2 = debug_data.kpt_person
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_KPT_BBOX, 2)
        if debug_data.kpt_face:
            x1, y1, x2, y2 = debug_data.kpt_face
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_HYBRID_FACE, 2)
        return vis