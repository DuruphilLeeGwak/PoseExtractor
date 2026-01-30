"""
Bbox Manager Module (Restored Original Logic - Constants Fixed)

위치: pose_transfer/logic/bbox_manager.py
역할:
- YOLO와 Keypoint 박스를 결합하여 최적의 BBox 산출
- 좌상단 텍스트 라벨링 기능 포함
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

# YOLO 로드 시도
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# [Fix] 시스템이 참조하는 상수명으로 복구
COLOR_KPT_BBOX = (0, 255, 0)        # Green
COLOR_YOLO_BBOX = (0, 0, 255)       # Red
COLOR_HYBRID_PERSON = (255, 255, 0) # Cyan
COLOR_HYBRID_FACE = (255, 0, 255)   # Magenta

@dataclass
class BboxInfo:
    x1: int
    y1: int
    x2: int
    y2: int
    center: Tuple[int, int]
    
    # 정렬용 추가 정보
    has_lower_body: bool = False
    has_face: bool = False
    feet_center: Tuple[int, int] = (0, 0)
    face_center: Tuple[int, int] = (0, 0)
    
    # 디버그용 라벨
    label: str = ""

    @property
    def width(self) -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    def to_tuple(self): return (self.x1, self.y1, self.x2, self.y2)

@dataclass
class DebugBboxData:
    kpt_person: Optional[BboxInfo] = None
    yolo_person: Optional[BboxInfo] = None
    final_person: Optional[BboxInfo] = None
    final_face: Optional[BboxInfo] = None

class BboxManager:
    def __init__(self, config):
        self.config = config
        self.person_model = None
        self.face_model = None
        
        # 모델 로드 (경로 자동 탐색)
        if YOLO_AVAILABLE:
            base_dir = Path(__file__).parent.parent.parent
            person_ckpt = base_dir / "models" / "yolo11n.pt"
            face_ckpt = base_dir / "models" / "yolo11n-face.pt"
            
            if person_ckpt.exists():
                try: self.person_model = YOLO(str(person_ckpt))
                except: pass
            if face_ckpt.exists():
                try: self.face_model = YOLO(str(face_ckpt))
                except: pass

    def get_bboxes(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[BboxInfo, BboxInfo, DebugBboxData]:
        h, w = image.shape[:2]
        debug_data = DebugBboxData()
        
        # 1. KPT 기반 계산
        kpt_face = self._kpt_to_face(keypoints, scores, (h, w))
        kpt_person = self._kpt_to_person(keypoints, scores, (h, w), face_top=kpt_face.y1)
        kpt_person.label = "KPT"
        debug_data.kpt_person = kpt_person
        
        # 2. YOLO 기반 계산
        yolo_person = self._get_yolo_bbox(self.person_model, image, 0, "YOLO")
        if yolo_person: debug_data.yolo_person = yolo_person
        
        # 3. Hybrid Merge (기존 로직: Union)
        if yolo_person and kpt_person.width > 0:
            final_p = self._merge_bboxes(kpt_person, yolo_person, "Hybrid")
        elif yolo_person:
            final_p = yolo_person
            final_p.label = "YOLO-Only"
        else:
            final_p = kpt_person
            final_p.label = "KPT-Only"
            
        # Face 처리 (YOLO Face 있으면 우선 사용)
        yolo_face = self._get_yolo_bbox(self.face_model, image, None, "YOLO-F")
        if yolo_face:
            final_f = self._merge_bboxes(kpt_face, yolo_face, "Hybrid-F")
        else:
            final_f = kpt_face
            
        # 4. Margin & Anchor Update
        final_p = self._apply_margin(final_p, (h, w), self.config.person_bbox_margin)
        final_f = self._apply_margin(final_f, (h, w), self.config.face_bbox_margin)
        
        final_p = self._update_anchors(final_p, keypoints, scores, final_f)
        final_f = self._update_anchors(final_f, keypoints, scores, None, is_face=True)
        
        debug_data.final_person = final_p
        debug_data.final_face = final_f
        
        return final_p, final_f, debug_data

    def _get_yolo_bbox(self, model, image, target_cls, label) -> Optional[BboxInfo]:
        if model is None: return None
        try:
            results = model(image, verbose=False)
            if not results or not results[0].boxes: return None
            
            # 가장 큰 박스 선택
            best_box = None
            max_area = 0
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if target_cls is not None and cls_id != target_cls: continue
                
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                area = (xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1])
                if area > max_area:
                    max_area = area
                    best_box = xyxy
            
            if best_box is not None:
                cx, cy = (best_box[0]+best_box[2])//2, (best_box[1]+best_box[3])//2
                return BboxInfo(*best_box, (cx, cy), label=label)
        except: pass
        return None

    def _merge_bboxes(self, b1: BboxInfo, b2: BboxInfo, label) -> BboxInfo:
        x1 = min(b1.x1, b2.x1); y1 = min(b1.y1, b2.y1)
        x2 = max(b1.x2, b2.x2); y2 = max(b1.y2, b2.y2)
        return BboxInfo(x1, y1, x2, y2, ((x1+x2)//2, (y1+y2)//2), label=label)

    def _apply_margin(self, bbox: BboxInfo, img_size, margin) -> BboxInfo:
        if margin <= 0: return bbox
        H, W = img_size
        px = int(bbox.width * margin); py = int(bbox.height * margin)
        bbox.x1 = max(0, bbox.x1 - px); bbox.y1 = max(0, bbox.y1 - py)
        bbox.x2 = min(W, bbox.x2 + px); bbox.y2 = min(H, bbox.y2 + py)
        bbox.center = ((bbox.x1+bbox.x2)//2, (bbox.y1+bbox.y2)//2)
        return bbox

    def _update_anchors(self, bbox: BboxInfo, kpts, scores, face_bbox, is_face=False) -> BboxInfo:
        if is_face:
            bbox.has_face = True
            bbox.face_center = bbox.center
            return bbox
            
        # 발 좌표 계산 (발목+뒷꿈치 평균)
        feet_indices = [15, 16, 19, 22]
        valid_feet = [kpts[i] for i in feet_indices if i < len(scores) and scores[i] > 0.1]
        
        if valid_feet:
            fc = np.mean(valid_feet, axis=0).astype(int)
            bbox.feet_center = (fc[0], fc[1])
            bbox.has_lower_body = True
        else:
            # 발이 없으면 박스 하단 중앙을 강제 할당 (정렬 실패 방지)
            bbox.feet_center = (bbox.center[0], bbox.y2)
            bbox.has_lower_body = False 
            
        if face_bbox and face_bbox.width > 0:
            bbox.has_face = True
            bbox.face_center = face_bbox.center
            
        return bbox

    def _kpt_to_person(self, kpts, scores, img_size, face_top=None) -> BboxInfo:
        H, W = img_size
        valid = kpts[scores > 0.1]
        if len(valid) == 0: return BboxInfo(0,0,W,H, (W//2, H//2), label="Fail")
        
        x1, y1 = np.min(valid, axis=0).astype(int)
        x2, y2 = np.max(valid, axis=0).astype(int)
        if face_top is not None and face_top < y1: y1 = face_top # 머리 위 보정
        
        return BboxInfo(x1, y1, x2, y2, ((x1+x2)//2, (y1+y2)//2))

    def _kpt_to_face(self, kpts, scores, img_size) -> BboxInfo:
        valid = [kpts[i] for i in range(5) if scores[i] > 0.1]
        if not valid: return BboxInfo(0,0,0,0, (0,0))
        valid = np.array(valid)
        x1, y1 = np.min(valid, axis=0).astype(int)
        x2, y2 = np.max(valid, axis=0).astype(int)
        # 이마 보정
        y1 = max(0, int(y1 - (x2-x1)*0.6))
        return BboxInfo(x1, y1, x2, y2, ((x1+x2)//2, (y1+y2)//2))

    def draw_debug(self, image: np.ndarray, debug_data: DebugBboxData) -> np.ndarray:
        vis = image.copy()
        
        def _draw(box: BboxInfo, color, thick=2):
            if box and box.width > 0:
                cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), color, thick)
                # 텍스트 라벨 복구
                label = box.label if box.label else "Box"
                cv2.putText(vis, label, (box.x1, max(20, box.y1-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _draw(debug_data.kpt_person, COLOR_KPT_BBOX, 1)
        _draw(debug_data.yolo_person, COLOR_YOLO_BBOX, 1)
        _draw(debug_data.final_person, COLOR_HYBRID_PERSON, 3) # 최종은 굵게
        _draw(debug_data.final_face, COLOR_HYBRID_FACE, 2)
        
        return vis