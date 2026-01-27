"""
BboxManager: Bounding Box 관리 및 검증 모듈

주요 기능:
1. Keypoint 기반 Bbox 생성 (Person/Face)
2. YOLO 모델을 통한 Bbox 검증 (선택적)
3. Keypoint + YOLO Hybrid Bbox 생성
4. 디버그 시각화 지원

작동 원리:
- Keypoint → Bbox: 유효한 keypoint들로부터 bounding box 계산
- YOLO Verification: keypoint bbox와 YOLO 감지 결과를 IoU로 비교
- Hybrid Strategy: IoU > threshold이면 두 bbox를 union하여 최종 bbox 생성
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# ============================================================
# Bbox 시각화 색상 정의
# ============================================================
COLOR_KPT_BBOX = (0, 255, 0)        # 녹색: Keypoint 기반 Bbox
COLOR_YOLO_BBOX = (255, 0, 0)       # 파란색: YOLO 감지 Bbox
COLOR_HYBRID_PERSON = (127, 0, 255) # 자주색: Hybrid Person Bbox
COLOR_HYBRID_FACE = (128, 128, 0)   # 청록색: Hybrid Face Bbox

# ============================================================
# Keypoint 인덱스 정의 (COCO-WholeBody 133 format)
# ============================================================
BODY_INDICES = {
    'nose': 0,              # 코
    'eyes': [1, 2],         # 왼쪽/오른쪽 눈
    'ears': [3, 4],         # 왼쪽/오른쪽 귀
    'shoulders': [5, 6],    # 왼쪽/오른쪽 어깨
    'hips': [11, 12]        # 왼쪽/오른쪽 엉덩이
}
JAW_INDICES = list(range(23, 40))   # 턱선 keypoints (Face 영역)
BROW_INDICES = list(range(40, 50))  # 눈썹 keypoints (Face 영역)

@dataclass
class BboxInfo:
    """
    Bounding Box 정보를 담는 데이터 클래스
    
    Attributes:
        bbox: (x1, y1, x2, y2) 좌표 튜플
        center: (cx, cy) 중심점 좌표
        size: max(width, height) - bbox의 최대 변 길이
        source: bbox 생성 출처 ("keypoint", "yolo", "hybrid", "fallback")
    """
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    size: float
    source: str

@dataclass
class DebugBboxData:
    """
    디버그용 모든 Bbox 정보를 저장하는 컨테이너
    
    Attributes:
        kpt_person: Keypoint 기반 Person Bbox
        kpt_face: Keypoint 기반 Face Bbox
        yolo_person: YOLO 감지 Person Bbox
        yolo_face: YOLO 감지 Face Bbox
        hybrid_person: 최종 Hybrid Person Bbox (사용됨)
        hybrid_face: 최종 Hybrid Face Bbox (사용됨)
    """
    kpt_person: Optional[BboxInfo] = None
    kpt_face: Optional[BboxInfo] = None
    yolo_person: Optional[BboxInfo] = None
    yolo_face: Optional[BboxInfo] = None
    hybrid_person: Optional[BboxInfo] = None
    hybrid_face: Optional[BboxInfo] = None

class BboxManager:
    """
    Bounding Box 생성 및 검증을 담당하는 관리자 클래스
    
    주요 역할:
    1. Keypoint 기반 Person/Face Bbox 계산
    2. YOLO 모델을 통한 Bbox 검증 (선택적)
    3. Keypoint + YOLO Hybrid Bbox 생성 (IoU 기반 융합)
    4. 디버그 시각화 지원
    
    Hybrid Strategy:
    - IoU(keypoint_bbox, yolo_bbox) > threshold → Union(두 bbox)
    - IoU <= threshold → keypoint_bbox 우선 (keypoint가 더 신뢰성 있음)
    - keypoint가 fallback인 경우 → yolo_bbox 사용
    """
    
    def __init__(self, config):
        """
        BboxManager 초기화
        
        Args:
            config: BboxConfig 객체 (yolo_verification_enabled 등 설정 포함)
        """
        self.config = config
        self._yolo_person = None  # YOLO Person 감지 모델
        self._yolo_face = None    # YOLO Face 감지 모델
        
        # YOLO 검증이 활성화된 경우에만 모델 로드
        if config.yolo_verification_enabled:
            self._init_models()

    def _init_models(self):
        """
        YOLO 모델 초기화 (Person + Face Detection)
        
        모델 로드 우선순위:
        1. 로컬 models/ 폴더의 .pt 파일
        2. Ultralytics 기본 다운로드 (Person)
        3. HuggingFace Hub 다운로드 (Face)
        
        파일 경로:
        - models/yolo11n.pt: Person Detection
        - models/yolo11n-face.pt: Face Detection
        """
        try:
            from ultralytics import YOLO
            from pathlib import Path
            
            # 현재 파일 위치: pose_transfer/logic/bbox_manager.py
            # 프로젝트 루트로 이동: ../../
            base_dir = Path(__file__).parent.parent.parent 
            models_dir = base_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Person Detection 모델 로드
            p_path = models_dir / "yolo11n.pt"
            if p_path.exists(): 
                self._yolo_person = YOLO(str(p_path))
                print(f"   ✅ Loaded Person YOLO from {p_path}")
            else: 
                self._yolo_person = YOLO('yolo11n.pt')  # Ultralytics 자동 다운로드
                print(f"   ✅ Downloaded Person YOLO")
            
            # Face Detection 모델 로드
            f_path = models_dir / "yolo11n-face.pt"
            if f_path.exists(): 
                self._yolo_face = YOLO(str(f_path))
                print(f"   ✅ Loaded Face YOLO from {f_path}")
            else:
                # HuggingFace Hub에서 다운로드 (없으면 스킵)
                try:
                    from huggingface_hub import hf_hub_download
                    path = hf_hub_download(
                        repo_id="AdamCodd/YOLOv11n-face-detection", 
                        filename="model.pt"
                    )
                    self._yolo_face = YOLO(path)
                    print(f"   ✅ Downloaded Face YOLO from HuggingFace")
                except ImportError:
                    print(f"   ⚠️ huggingface_hub not installed, Face YOLO disabled")
                    self._yolo_face = None
        except Exception as e:
            print(f"   ⚠️ YOLO Init Failed: {e}")


    def get_bboxes(self, image, kpts, scores) -> Tuple[BboxInfo, BboxInfo, DebugBboxData]:
        h, w = image.shape[:2]
        
        # Step 1: Person Bbox 먼저 계산
        kpt_p = self._kpt_to_person(kpts, scores, (h, w))
        
        # Step 2: Face Bbox 계산 시 Person Bbox 정보를 제약조건으로 전달
        # (kpt_p를 넘겨줘서 이 영역을 벗어나지 않게 함)
        kpt_f = self._kpt_to_face(kpts, scores, size=(h, w), person_bbox_info=kpt_p)
        
        # 디버그 데이터 초기화
        debug_data = DebugBboxData(kpt_person=kpt_p, kpt_face=kpt_f)
        
        # Step 3: YOLO 검증 (기존 로직 유지)
        if self.config.yolo_verification_enabled and self._yolo_person:
            debug_data, _ = self._run_yolo(image, kpt_p, kpt_f, debug_data)
        else:
            debug_data.hybrid_person = kpt_p
            debug_data.hybrid_face = kpt_f
            
        return debug_data.hybrid_person, debug_data.hybrid_face, debug_data

    def _kpt_to_person(self, kpts, scores, size):
        """
        Keypoint로부터 Person Bounding Box 생성
        
        알고리즘:
        1. Body keypoint만 선택 (0-16: 몸통/팔다리 관절)
        2. score > threshold인 keypoint만 사용
        3. 선택된 keypoint들의 min/max 좌표로 bbox 계산
        4. margin(%) 만큼 확장하여 여유 공간 확보
        
        Note: 얼굴(23-90), 손(91-132) keypoint는 제외!
              Person bbox는 몸통만 포함해야 정확함
        
        Args:
            kpts: Keypoint 좌표 배열 (133x2)
            scores: Keypoint 신뢰도 점수 (133,)
            size: 이미지 크기 (height, width)
        
        Returns:
            BboxInfo: Person Bbox 정보
                - 유효 keypoint가 없으면 전체 이미지를 fallback bbox로 반환
        """
        h, w = size
        
        # Body + Feet keypoint만 선택 (0-22번)
        # 0-16: Body (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
        # 17-22: Feet (toes, heels)
        body_kpts = kpts[:23]  # 0-22번까지만
        body_scores = scores[:23]
        
        # 신뢰도 threshold 이상인 Body keypoint만 선택
        valid = body_kpts[body_scores > self.config.kpt_threshold]
        
        # Fallback: 유효 keypoint가 없으면 전체 이미지 사용
        if len(valid) == 0: 
            return BboxInfo(
                (0, 0, w, h), 
                (w/2, h/2), 
                max(w, h), 
                "fallback"
            )
        
        # Min/Max 좌표로 bounding box 계산
        mn, mx = valid.min(0), valid.max(0)
        
        # 🔍 DEBUG: Keypoint bbox가 왜 큰지 확인
        print(f"\n🔍 [DEBUG] _kpt_to_person bbox calculation:")
        print(f"   Image size: {w}x{h}")
        print(f"   Valid keypoints: {len(valid)}/{len(body_kpts)}")
        print(f"   Keypoint range: x=[{mn[0]:.1f}, {mx[0]:.1f}], y=[{mn[1]:.1f}, {mx[1]:.1f}]")
        
        # Margin 적용 (설정 파일에서 지정, 예: 0.15 = 15%)
        margin = self.config.person_bbox_margin
        wd, ht = mx - mn
        mx_pad, my_pad = wd * margin, ht * margin
        
        print(f"   Margin: {margin*100:.1f}% → padding=({mx_pad:.1f}, {my_pad:.1f})")
        
        # 이미지 범위 내로 클리핑
        x1, y1 = max(0, int(mn[0] - mx_pad)), max(0, int(mn[1] - my_pad))
        x2, y2 = min(w, int(mx[0] + mx_pad)), min(h, int(mx[1] + my_pad))
        
        print(f"   Final bbox: ({x1}, {y1}, {x2}, {y2})")
        print(f"   Bbox size: {x2-x1}x{y2-y1}")
        
        return BboxInfo(
            (x1, y1, x2, y2), 
            ((x1+x2)/2, (y1+y2)/2), 
            max(x2-x1, y2-y1), 
            "keypoint"
        )

    def _kpt_to_face(self, kpts, scores, size=None, person_bbox_info=None):
        """
        [Final Adjusted] Face BBox 생성
        - Gaze-Aware Shift: 뒤통수 확보 (유지)
        - Person Constraint: Person BBox 범위를 넘지 않도록 절삭 (추가)
        - Reduced Padding: 상단 패딩 비율 축소 (수정)
        """
        # 1. 얼굴 관련 keypoint 인덱스 수집
        idx = (
            [BODY_INDICES['nose']] +      # 코
            BODY_INDICES['eyes'] +         # 눈
            BODY_INDICES['ears'] +         # 귀
            JAW_INDICES +                  # 턱선
            BROW_INDICES                   # 눈썹
        )
        
        valid = [
            kpts[i] for i in idx 
            if i < len(scores) and scores[i] > self.config.kpt_threshold
        ]
        
        if len(valid) < 1: 
            return BboxInfo((0, 0, 100, 100), (50.0, 50.0), 100.0, "fallback")
        
        # 2. 기초 BBox (이목구비 기준)
        v = np.array(valid)
        mn, mx = v.min(0), v.max(0)
        x1, y1 = mn[0], mn[1]
        x2, y2 = mx[0], mx[1]
        
        # -------------------------------------------------------------------
        # 🧠 Skull Estimation & Constraints
        # -------------------------------------------------------------------
        LS, RS = 5, 6
        NOSE = 0
        
        if LS < len(scores) and RS < len(scores) and \
           scores[LS] > 0.1 and scores[RS] > 0.1:
            
            shoulder_width = np.linalg.norm(kpts[RS] - kpts[LS])
            min_skull_size = shoulder_width * 0.5
            
            # (A) Gaze-Aware Shift (뒤통수 확보 - 기존 유지)
            shift_x = 0.0
            if scores[NOSE] > 0.1:
                nose_x = kpts[NOSE][0]
                neck_x = (kpts[LS][0] + kpts[RS][0]) / 2.0
                look_vec = nose_x - neck_x
                if abs(look_vec) > shoulder_width * 0.1:
                    shift_x = -look_vec * 0.8
            
            current_cx = (x1 + x2) / 2
            target_cx = current_cx + shift_x
            
            half_size = min_skull_size / 2
            new_x1 = target_cx - half_size
            new_x2 = target_cx + half_size
            
            x1 = min(x1, new_x1)
            x2 = max(x2, new_x2)
            
            # (B) Top Padding 조정 (과도한 상측 확장 방지)
            # 기존 0.6 -> 0.35로 축소 (이마+머리카락 정도만 확보)
            y1 = y1 - (min_skull_size * 0.35)
            
            # 하단은 턱 아래 약간만
            y2 = max(y2, y1 + min_skull_size * 1.1)

        # -------------------------------------------------------------------
        # ✂️ Person BBox Constraint (Person 영역 밖으로 나가지 않게 절삭)
        # -------------------------------------------------------------------
        if person_bbox_info is not None:
            px1, py1, px2, py2 = person_bbox_info.bbox
            
            # Face BBox가 Person BBox를 벗어나면 잘라냄
            # 단, Person BBox가 너무 타이트할 수 있으므로 아주 약간의 여유(slack)는 허용 가능
            # 여기서는 엄격하게 자르는 방식을 적용
            
            # 상단(머리 위)이 Person BBox보다 높으면 자름
            if y1 < py1:
                # print(f"   ✂️ Clipping Top: {y1:.1f} -> {py1}")
                y1 = py1
            
            # 좌우/하단도 Person BBox 내부로 제한
            x1 = max(x1, px1)
            x2 = min(x2, px2)
            y2 = min(y2, py2)

        # 4. Margin 적용 (기본 설정값)
        margin = self.config.face_bbox_margin
        wd, ht = x2 - x1, y2 - y1
        mx_pad, my_pad = wd * margin, ht * margin
        
        x1 -= mx_pad
        y1 -= my_pad
        x2 += mx_pad
        y2 += my_pad
        
        # 5. 좌표 클리핑
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = max(x1 + 1, int(x2))
        y2 = max(y1 + 1, int(y2))
        
        if size:
            h, w = size
            x2 = min(w, x2)
            y2 = min(h, y2)
        
        final_size = float(max(x2-x1, y2-y1))
        
        return BboxInfo(
            bbox=(x1, y1, x2, y2), 
            center=((x1+x2)/2.0, (y1+y2)/2.0), 
            size=final_size, 
            source="keypoint"
        )
    
    # ============================================================
    # Public Helper (Pipeline에서 호출)
    # ============================================================
    def _kpt_to_face_public(self, kpts, scores):
        """
        외부(Pipeline)에서 호출 가능한 Face Bbox 생성 메서드
        
        Note: _kpt_to_face()의 래퍼 메서드
        """
        return self._kpt_to_face(kpts, scores)

    def _run_yolo(self, img, kp_p, kp_f, debug):
        """
        YOLO 모델로 Person 및 Face를 감지하고 Hybrid Bbox 생성
        
        처리 흐름:
        
        [Person Detection]
        1. YOLO Person 모델로 전체 이미지 검사
        2. 가장 큰 person bbox 선택
        3. IoU(keypoint_bbox, yolo_bbox) 계산
        4. IoU > 0.3 → Union(keypoint, yolo)
        5. IoU <= 0.3 → keypoint_bbox 유지
        
        [Face Detection]
        1. Person bbox 영역만 crop하여 Face 검사 (효율성)
        2. YOLO Face 모델로 crop 영역 검사
        3. IoU(keypoint_face, yolo_face) 계산
        4. IoU > 0.1 → Union(keypoint, yolo)
        5. IoU <= 0.1 → keypoint가 fallback이면 yolo 사용, 아니면 keypoint 유지
        
        Args:
            img: 입력 이미지
            kp_p: Keypoint 기반 Person Bbox
            kp_f: Keypoint 기반 Face Bbox
            debug: DebugBboxData (디버그 정보 저장용)
        
        Returns:
            Tuple[DebugBboxData, None]:
                - 업데이트된 debug 데이터 (hybrid bbox 포함)
                - None (미래 확장용 placeholder)
        """
        # ============================================================
        # [1] Person Detection & Hybrid
        # ============================================================
        res = self._yolo_person.predict(
            img, 
            conf=self.config.yolo_person_conf,  # 신뢰도 threshold
            verbose=False
        )[0].boxes
        
        # YOLO class 0 = person
        mask = res.cls == 0
        h_p = kp_p  # 기본값: keypoint bbox
        
        if mask.sum() > 0:
            # 감지된 person들 중 가장 큰 bbox 선택
            b = res.xyxy[mask].cpu().numpy()
            yb = b[np.argmax((b[:,2]-b[:,0]) * (b[:,3]-b[:,1]))].astype(int)
            
            y_info = BboxInfo(
                (yb[0], yb[1], yb[2], yb[3]), 
                ((yb[0]+yb[2])/2, (yb[1]+yb[3])/2), 
                max(yb[2]-yb[0], yb[3]-yb[1]), 
                "yolo"
            )
            debug.yolo_person = y_info
            
            # Hybrid Bbox 생성: 항상 Union(합집합) 사용
            # YOLO가 더 큰 영역을 잡으므로 keypoint bbox와 합쳐서 확장
            kb = kp_p.bbox
            u_box = (
                min(kb[0], yb[0]),  # 더 왼쪽
                min(kb[1], yb[1]),  # 더 위쪽
                max(kb[2], yb[2]),  # 더 오른쪽
                max(kb[3], yb[3])   # 더 아래쪽
            )
            h_p = BboxInfo(
                u_box, 
                ((u_box[0]+u_box[2])/2, (u_box[1]+u_box[3])/2), 
                max(u_box[2]-u_box[0], u_box[3]-u_box[1]), 
                "hybrid"
            )
        
        debug.hybrid_person = h_p

        # ============================================================
        # [2] Face Detection & Hybrid (Person bbox 영역 내에서만 검색)
        # ============================================================
        px1, py1, px2, py2 = h_p.bbox
        h, w = img.shape[:2]
        
        # Person bbox를 이미지 범위 내로 클리핑
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        
        # Person 영역만 crop
        crop = img[py1:py2, px1:px2]
        h_f = kp_f  # 기본값: keypoint bbox
        
        if crop.size > 0:
            # YOLO Face 모델로 crop 영역 검사
            f_res = self._yolo_face.predict(
                crop, 
                conf=self.config.yolo_face_conf, 
                verbose=False
            )[0].boxes
            
            if len(f_res) > 0:
                # 첫 번째 face bbox 선택 (confidence가 가장 높음)
                fb = f_res[0].xyxy[0].cpu().numpy().astype(int)
                
                # Crop 좌표를 원본 이미지 좌표로 변환
                fx1, fy1 = fb[0] + px1, fb[1] + py1
                fx2, fy2 = fb[2] + px1, fb[3] + py1
                
                y_info = BboxInfo(
                    (fx1, fy1, fx2, fy2), 
                    ((fx1+fx2)/2, (fy1+fy2)/2), 
                    max(fx2-fx1, fy2-fy1), 
                    "yolo"
                )
                debug.yolo_face = y_info
                
                # Hybrid Bbox 생성: 항상 Union(합집합) 사용
                # YOLO Face가 더 정확하므로 keypoint와 합쳐서 확장
                kb = kp_f.bbox
                u_box = (
                    min(kb[0], fx1),  # 더 왼쪽
                    min(kb[1], fy1),  # 더 위쪽
                    max(kb[2], fx2),  # 더 오른쪽
                    max(kb[3], fy2)   # 더 아래쪽
                )
                h_f = BboxInfo(
                    u_box, 
                    ((u_box[0]+u_box[2])/2, (u_box[1]+u_box[3])/2), 
                    max(u_box[2]-u_box[0], u_box[3]-u_box[1]), 
                    "hybrid"
                )
        
        debug.hybrid_face = h_f
        return debug, None

    def _calc_iou(self, b1, b2):
        """
        두 Bounding Box 간의 IoU (Intersection over Union) 계산
        
        IoU = Intersection Area / Union Area
        
        IoU 값의 의미:
        - 1.0: 완전히 겹침
        - 0.5: 절반 정도 겹침
        - 0.0: 전혀 겹치지 않음
        
        Args:
            b1: Bbox 1 (x1, y1, x2, y2)
            b2: Bbox 2 (x1, y1, x2, y2)
        
        Returns:
            float: IoU 값 (0.0 ~ 1.0)
        """
        # 교집합(Intersection) 영역 계산
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 각 bbox의 면적 계산
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        
        # 합집합(Union) = 면적1 + 면적2 - 교집합
        union = a1 + a2 - inter
        
        return inter / union if union > 0 else 0

    def draw_debug(self, img, data: DebugBboxData):
        """
        디버그용 Bbox 시각화
        
        설정에 따라 다음 bbox들을 그릴 수 있음:
        - viz_kpt_bbox: Keypoint 기반 bbox (녹색)
        - viz_yolo_bbox: YOLO 감지 bbox (파란색)
        - viz_hybrid_bbox: 최종 Hybrid bbox (자주색/청록색)
        
        Args:
            img: 입력 이미지
            data: DebugBboxData (모든 bbox 정보 포함)
        
        Returns:
            np.ndarray: Bbox가 그려진 이미지
        """
        # 시각화가 모두 비활성화되어 있으면 원본 반환
        if not (self.config.viz_kpt_bbox or self.config.viz_yolo_bbox or self.config.viz_hybrid_bbox):
            return img
        
        out = img.copy()
        thick = max(1, self.config.line_thickness // 2)
        
        # Keypoint Bbox 그리기 (녹색)
        if self.config.viz_kpt_bbox:
            if data.kpt_person: 
                self._draw(out, data.kpt_person, COLOR_KPT_BBOX, "KPT-P", thick)
            if data.kpt_face: 
                self._draw(out, data.kpt_face, COLOR_KPT_BBOX, "KPT-F", thick)
        
        # YOLO Bbox 그리기 (파란색)
        if self.config.viz_yolo_bbox:
            if data.yolo_person: 
                self._draw(out, data.yolo_person, COLOR_YOLO_BBOX, "YOLO-P", thick)
            if data.yolo_face: 
                self._draw(out, data.yolo_face, COLOR_YOLO_BBOX, "YOLO-F", thick)
        
        # Hybrid Bbox 그리기 (자주색/청록색, 더 두껍게)
        if self.config.viz_hybrid_bbox:
            if data.hybrid_person: 
                self._draw(out, data.hybrid_person, COLOR_HYBRID_PERSON, "Hybrid-P", thick+1)
            if data.hybrid_face: 
                self._draw(out, data.hybrid_face, COLOR_HYBRID_FACE, "Hybrid-F", thick+1)
        
        return out

    def _draw(self, img, info, color, label, thick):
        x1, y1, x2, y2 = info.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)