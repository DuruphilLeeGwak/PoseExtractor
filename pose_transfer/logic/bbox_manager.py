import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# Bbox 색상 정의
COLOR_KPT_BBOX = (0, 255, 0)
COLOR_YOLO_BBOX = (255, 0, 0)
COLOR_HYBRID_PERSON = (127, 0, 255)
COLOR_HYBRID_FACE = (128, 128, 0)

# 인덱스 정의
BODY_INDICES = {
    'nose': 0, 'eyes': [1, 2], 'ears': [3, 4], 
    'shoulders': [5, 6], 'hips': [11, 12]
}
JAW_INDICES = list(range(23, 40))
BROW_INDICES = list(range(40, 50))

@dataclass
class BboxInfo:
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    size: float
    source: str

@dataclass
class DebugBboxData:
    kpt_person: Optional[BboxInfo] = None
    kpt_face: Optional[BboxInfo] = None
    yolo_person: Optional[BboxInfo] = None
    yolo_face: Optional[BboxInfo] = None
    hybrid_person: Optional[BboxInfo] = None
    hybrid_face: Optional[BboxInfo] = None

class BboxManager:
    def __init__(self, config):
        self.config = config
        self._yolo_person = None
        self._yolo_face = None
        if config.yolo_verification_enabled:
            self._init_models()

    def _init_models(self):
        try:
            from ultralytics import YOLO
            from pathlib import Path
            from huggingface_hub import hf_hub_download
            
            # api.py 기준 상위 폴더들 탐색
            # 현재 파일 위치: pose_transfer/logic/bbox_manager.py
            base_dir = Path(__file__).parent.parent.parent 
            models_dir = base_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            p_path = models_dir / "yolo11n.pt"
            f_path = models_dir / "yolo11n-face.pt"
            
            if p_path.exists(): self._yolo_person = YOLO(str(p_path))
            else: self._yolo_person = YOLO('yolo11n.pt') 
            
            if f_path.exists(): self._yolo_face = YOLO(str(f_path))
            else:
                path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
                self._yolo_face = YOLO(path)
        except Exception as e:
            print(f"   ⚠️ YOLO Init Failed: {e}")

    def get_bboxes(self, image, kpts, scores) -> Tuple[BboxInfo, BboxInfo, DebugBboxData]:
        h, w = image.shape[:2]
        kpt_p = self._kpt_to_person(kpts, scores, (h, w))
        kpt_f = self._kpt_to_face(kpts, scores)
        
        debug_data = DebugBboxData(kpt_person=kpt_p, kpt_face=kpt_f)
        
        if self.config.yolo_verification_enabled and self._yolo_person:
            debug_data, _ = self._run_yolo(image, kpt_p, kpt_f, debug_data)
        else:
            debug_data.hybrid_person = kpt_p
            debug_data.hybrid_face = kpt_f
            
        return debug_data.hybrid_person, debug_data.hybrid_face, debug_data

    def _kpt_to_person(self, kpts, scores, size):
        h, w = size
        valid = kpts[scores > self.config.kpt_threshold]
        if len(valid) == 0: return BboxInfo((0,0,w,h), (w/2,h/2), max(w,h), "fallback")
        mn, mx = valid.min(0), valid.max(0)
        margin = self.config.person_bbox_margin
        wd, ht = mx - mn
        mx_pad, my_pad = wd * margin, ht * margin
        x1, y1 = max(0, int(mn[0]-mx_pad)), max(0, int(mn[1]-my_pad))
        x2, y2 = min(w, int(mx[0]+mx_pad)), min(h, int(mx[1]+my_pad))
        return BboxInfo((x1,y1,x2,y2), ((x1+x2)/2, (y1+y2)/2), max(x2-x1, y2-y1), "keypoint")

    def _kpt_to_face(self, kpts, scores):
        idx = [BODY_INDICES['nose']] + BODY_INDICES['eyes'] + BODY_INDICES['ears'] + JAW_INDICES + BROW_INDICES
        valid = [kpts[i] for i in idx if i < len(scores) and scores[i] > self.config.kpt_threshold]
        if len(valid) < 2: return BboxInfo((0,0,100,100), (50,50), 100, "fallback")
        v = np.array(valid)
        mn, mx = v.min(0), v.max(0)
        margin = self.config.face_bbox_margin
        wd, ht = mx - mn
        mx_pad, my_pad = wd * margin, ht * margin
        x1, y1 = int(mn[0]-mx_pad), int(mn[1]-my_pad)
        x2, y2 = int(mx[0]+mx_pad), int(mx[1]+my_pad)
        size = max(x2-x1, y2-y1)
        return BboxInfo((x1,y1,x2,y2), ((x1+x2)/2, (y1+y2)/2), size, "keypoint")
    
    # helper for pipeline alignment callback
    def _kpt_to_face_public(self, kpts, scores):
        return self._kpt_to_face(kpts, scores)

    def _run_yolo(self, img, kp_p, kp_f, debug):
        # Person
        res = self._yolo_person.predict(img, conf=self.config.yolo_person_conf, verbose=False)[0].boxes
        mask = res.cls == 0
        h_p = kp_p
        if mask.sum() > 0:
            b = res.xyxy[mask].cpu().numpy()
            yb = b[np.argmax((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))].astype(int)
            y_info = BboxInfo((yb[0], yb[1], yb[2], yb[3]), ((yb[0]+yb[2])/2, (yb[1]+yb[3])/2), max(yb[2]-yb[0], yb[3]-yb[1]), "yolo")
            debug.yolo_person = y_info
            if self._calc_iou(kp_p.bbox, yb) > 0.3:
                kb = kp_p.bbox
                u_box = (min(kb[0],yb[0]), min(kb[1],yb[1]), max(kb[2],yb[2]), max(kb[3],yb[3]))
                h_p = BboxInfo(u_box, ((u_box[0]+u_box[2])/2, (u_box[1]+u_box[3])/2), max(u_box[2]-u_box[0], u_box[3]-u_box[1]), "hybrid")
        debug.hybrid_person = h_p

        # Face
        px1, py1, px2, py2 = h_p.bbox
        h, w = img.shape[:2]
        px1, py1 = max(0, px1), max(0, py1); px2, py2 = min(w, px2), min(h, py2)
        crop = img[py1:py2, px1:px2]
        h_f = kp_f
        if crop.size > 0:
            f_res = self._yolo_face.predict(crop, conf=self.config.yolo_face_conf, verbose=False)[0].boxes
            if len(f_res) > 0:
                fb = f_res[0].xyxy[0].cpu().numpy().astype(int)
                fx1, fy1, fx2, fy2 = fb[0]+px1, fb[1]+py1, fb[2]+px1, fb[3]+py1
                y_info = BboxInfo((fx1, fy1, fx2, fy2), ((fx1+fx2)/2, (fy1+fy2)/2), max(fx2-fx1, fy2-fy1), "yolo")
                debug.yolo_face = y_info
                if self._calc_iou(kp_f.bbox, (fx1, fy1, fx2, fy2)) > 0.1:
                    kb = kp_f.bbox
                    u_box = (min(kb[0],fx1), min(kb[1],fy1), max(kb[2],fx2), max(kb[3],fy2))
                    h_f = BboxInfo(u_box, ((u_box[0]+u_box[2])/2, (u_box[1]+u_box[3])/2), max(u_box[2]-u_box[0], u_box[3]-u_box[1]), "hybrid")
                else:
                    if kp_f.source == "fallback": h_f = y_info
                    else: h_f = y_info
        debug.hybrid_face = h_f
        return debug, None

    def _calc_iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

    def draw_debug(self, img, data: DebugBboxData):
        if not (self.config.viz_kpt_bbox or self.config.viz_yolo_bbox or self.config.viz_hybrid_bbox): return img
        out = img.copy()
        thick = max(1, self.config.line_thickness // 2)
        if self.config.viz_kpt_bbox:
            if data.kpt_person: self._draw(out, data.kpt_person, COLOR_KPT_BBOX, "KPT-P", thick)
            if data.kpt_face: self._draw(out, data.kpt_face, COLOR_KPT_BBOX, "KPT-F", thick)
        if self.config.viz_yolo_bbox:
            if data.yolo_person: self._draw(out, data.yolo_person, COLOR_YOLO_BBOX, "YOLO-P", thick)
            if data.yolo_face: self._draw(out, data.yolo_face, COLOR_YOLO_BBOX, "YOLO-F", thick)
        if self.config.viz_hybrid_bbox:
            if data.hybrid_person: self._draw(out, data.hybrid_person, COLOR_HYBRID_PERSON, "Hybrid-P", thick+1)
            if data.hybrid_face: self._draw(out, data.hybrid_face, COLOR_HYBRID_FACE, "Hybrid-F", thick+1)
        return out

    def _draw(self, img, info, color, label, thick):
        x1, y1, x2, y2 = info.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)