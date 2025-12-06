"""
포즈 전이 파이프라인 v6.3
- Bbox 마진 설정(bbox.person_margin, bbox.face_margin) 적용
- Bbox 계산 시 턱선(Jawline) 포함 로직 유지
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Set, List
from dataclasses import dataclass, field
from enum import Enum

from .extractors import (
    DWPoseExtractor, DWPoseExtractorFactory, PersonFilter, RTMLIB_AVAILABLE
)
from .extractors.keypoint_constants import BODY_KEYPOINTS, FEET_KEYPOINTS
from .transfer import PoseTransferEngine, TransferConfig, FallbackStrategy
from .refiners import HandRefiner
from .renderers import SkeletonRenderer
from .utils import load_config, convert_to_openpose_format, load_image

# Bbox 색상 정의 (BGR)
COLOR_KPT_BBOX = (0, 255, 0)
COLOR_YOLO_BBOX = (255, 0, 0)
COLOR_HYBRID_PERSON = (127, 0, 255)
COLOR_HYBRID_FACE = (128, 128, 0)

class BodyType(Enum):
    FULL = "full"
    UPPER = "upper"

class AlignmentCase(Enum):
    A = "A"; B = "B"; C = "C"; D = "D"

LOWER_BODY_INDICES = [11, 12, 13, 14, 15, 16]

@dataclass
class PipelineConfig:
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    filter_enabled: bool = True
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    hand_refinement_enabled: bool = True
    min_hand_size: int = 48
    fallback_enabled: bool = True
    transfer_confidence_threshold: float = 0.3
    line_thickness: int = 4
    face_line_thickness: int = 2
    hand_line_thickness: int = 2
    point_radius: int = 4
    kpt_threshold: float = 0.3
    ghost_legs_clipping_enabled: bool = True
    lower_body_confidence_threshold: float = 2.0
    lower_body_margin_ratio: float = 0.10
    auto_crop_enabled: bool = True
    crop_padding_px: int = 50
    head_padding_ratio: float = 1.0
    full_body_min_valid_lower: int = 4
    yolo_verification_enabled: bool = True
    yolo_person_conf: float = 0.5
    yolo_face_conf: float = 0.3
    face_scale_enabled: bool = True
    
    # [NEW] Bbox Margin 설정
    person_bbox_margin: float = 0.0
    face_bbox_margin: float = 0.0
    
    # 디버그 설정
    debug_bbox_visualization: bool = False
    viz_kpt_bbox: bool = False
    viz_yolo_bbox: bool = False
    viz_hybrid_bbox: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        config = load_config(yaml_path)
        rendering = config.get('rendering', {})
        transfer = config.get('transfer', {})
        output = config.get('output', {})
        alignment = config.get('alignment', {})
        debug = config.get('debug', {})
        bbox = config.get('bbox', {}) # [NEW] bbox 섹션 로드
        
        return cls(
            backend=config.get('model', {}).get('backend', 'onnxruntime'),
            device=config.get('model', {}).get('device', 'cuda'),
            mode=config.get('model', {}).get('mode', 'performance'),
            to_openpose=config.get('model', {}).get('to_openpose', False),
            filter_enabled=config.get('person_filter', {}).get('enabled', True),
            area_weight=config.get('person_filter', {}).get('area_weight', 0.6),
            center_weight=config.get('person_filter', {}).get('center_weight', 0.4),
            filter_confidence_threshold=config.get('person_filter', {}).get('confidence_threshold', 0.3),
            hand_refinement_enabled=config.get('hand_refinement', {}).get('enabled', True),
            min_hand_size=config.get('hand_refinement', {}).get('min_hand_size', 48),
            fallback_enabled=config.get('fallback', {}).get('symmetric_mirror', True),
            transfer_confidence_threshold=transfer.get('confidence_threshold', 0.3),
            line_thickness=rendering.get('line_thickness', 4),
            face_line_thickness=rendering.get('face_line_thickness', 2),
            hand_line_thickness=rendering.get('hand_line_thickness', 2),
            point_radius=rendering.get('point_radius', 4),
            kpt_threshold=rendering.get('kpt_threshold', 0.3),
            ghost_legs_clipping_enabled=transfer.get('ghost_legs_clipping_enabled', True),
            lower_body_confidence_threshold=transfer.get('lower_body_confidence_threshold', 2.0),
            lower_body_margin_ratio=transfer.get('lower_body_margin_ratio', 0.10),
            auto_crop_enabled=output.get('auto_crop_enabled', True),
            crop_padding_px=output.get('crop_padding_px', 50),
            head_padding_ratio=output.get('head_padding_ratio', 1.0),
            full_body_min_valid_lower=alignment.get('full_body_min_valid_lower', 4),
            yolo_verification_enabled=alignment.get('yolo_verification_enabled', True),
            yolo_person_conf=alignment.get('yolo_person_conf', 0.5),
            yolo_face_conf=alignment.get('yolo_face_conf', 0.3),
            face_scale_enabled=alignment.get('face_scale_enabled', True),
            
            # [NEW] Bbox Margin 매핑
            person_bbox_margin=bbox.get('person_margin', 0.0),
            face_bbox_margin=bbox.get('face_margin', 0.0),
            
            debug_bbox_visualization=debug.get('bbox_visualization', False),
            viz_kpt_bbox=debug.get('visualize_keypoint_bbox', True),
            viz_yolo_bbox=debug.get('visualize_yolo_bbox', True),
            viz_hybrid_bbox=debug.get('visualize_hybrid_bbox', True),
        )

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

@dataclass
class AlignmentInfo:
    case: AlignmentCase
    src_body_type: BodyType
    ref_body_type: BodyType
    src_person_bbox: BboxInfo
    src_face_bbox: BboxInfo
    ref_face_bbox: BboxInfo
    face_scale_ratio: float
    alignment_method: str
    yolo_log: Dict[str, bool]

@dataclass
class PipelineResult:
    transferred_keypoints: np.ndarray
    transferred_scores: np.ndarray
    source_keypoints: np.ndarray
    source_scores: np.ndarray
    source_bone_lengths: Dict[str, float]
    reference_keypoints: np.ndarray
    reference_scores: np.ndarray
    skeleton_image: np.ndarray
    image_size: Tuple[int, int]
    selected_person_idx: Dict[str, int] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    alignment_info: Optional[AlignmentInfo] = None
    src_debug_image: Optional[np.ndarray] = None
    ref_debug_image: Optional[np.ndarray] = None
    
    def to_json(self) -> Dict[str, Any]:
        return convert_to_openpose_format(
            self.transferred_keypoints[np.newaxis, ...],
            self.transferred_scores[np.newaxis, ...],
            self.image_size
        )

class PoseTransferPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, yaml_config: Optional[dict] = None):
        self.config = config or PipelineConfig()
        self.yaml_config = yaml_config
        self._yolo_person = None
        self._yolo_face = None
        self._init_modules()
    
    def _init_modules(self):
        if not RTMLIB_AVAILABLE: raise RuntimeError("rtmlib not installed.")
        
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend, device=self.config.device,
            mode=self.config.mode, to_openpose=self.config.to_openpose, force_new=True
        )
        self.person_filter = PersonFilter(
            area_weight=self.config.area_weight, center_weight=self.config.center_weight,
            confidence_threshold=self.config.filter_confidence_threshold
        )
        transfer_config = TransferConfig(confidence_threshold=self.config.transfer_confidence_threshold)
        self.transfer_engine = PoseTransferEngine(config=transfer_config, yaml_config=self.yaml_config)
        self.fallback_strategy = FallbackStrategy(confidence_threshold=self.config.transfer_confidence_threshold)
        self.hand_refiner = HandRefiner(min_hand_size=self.config.min_hand_size, confidence_threshold=self.config.transfer_confidence_threshold)
        self.renderer = SkeletonRenderer(
            line_thickness=self.config.line_thickness, point_radius=self.config.point_radius,
            kpt_threshold=self.config.kpt_threshold, face_line_thickness=self.config.face_line_thickness,
            hand_line_thickness=self.config.hand_line_thickness
        )
        if self.config.yolo_verification_enabled:
            self._init_yolo_models()

    def _init_yolo_models(self):
        try:
            from ultralytics import YOLO
            base_dir = Path(__file__).parent.parent
            models_dir = base_dir / "models"
            person_local = models_dir / "yolo11n.pt"
            face_local = models_dir / "yolo11n-face.pt"
            
            if person_local.exists(): self._yolo_person = YOLO(str(person_local))
            else: self._yolo_person = YOLO('yolo11n.pt')
            
            if face_local.exists(): self._yolo_face = YOLO(str(face_local))
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
                self._yolo_face = YOLO(path)
        except Exception as e:
            print(f"   ⚠️ YOLO Init Failed: {e}")
            self._yolo_person = None; self._yolo_face = None

    def extract_pose(self, image: Union[np.ndarray, str, Path], filter_person: bool = True) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int,int]]:
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        
        image_size = img.shape[:2]
        all_kpts, all_scores = self.extractor.extract(img)
        
        if len(all_kpts) == 0:
            return np.zeros((133, 2)), np.zeros(133), -1, image_size
        
        if filter_person and self.config.filter_enabled and len(all_kpts) > 1:
            kpts, scores, idx, _ = self.person_filter.select_main_person(
                all_kpts, all_scores, image_size
            )
        else:
            kpts, scores, idx = all_kpts[0], all_scores[0], 0
        
        if self.config.hand_refinement_enabled:
            kpts, scores, _ = self.hand_refiner.refine_both_hands(
                img, kpts, scores, self.extractor
            )
        
        return kpts, scores, idx, image_size

    def _draw_bbox_debug_layers(self, image: np.ndarray, data: DebugBboxData) -> np.ndarray:
        if not (self.config.viz_kpt_bbox or self.config.viz_yolo_bbox or self.config.viz_hybrid_bbox):
            return image
        debug_img = image.copy()
        thick = max(1, self.config.line_thickness // 2)
        
        if self.config.viz_kpt_bbox:
            if data.kpt_person: self._draw_box(debug_img, data.kpt_person, COLOR_KPT_BBOX, "KPT-P", thick)
            if data.kpt_face: self._draw_box(debug_img, data.kpt_face, COLOR_KPT_BBOX, "KPT-F", thick)
        if self.config.viz_yolo_bbox:
            if data.yolo_person: self._draw_box(debug_img, data.yolo_person, COLOR_YOLO_BBOX, "YOLO-P", thick)
            if data.yolo_face: self._draw_box(debug_img, data.yolo_face, COLOR_YOLO_BBOX, "YOLO-F", thick)
        if self.config.viz_hybrid_bbox:
            if data.hybrid_person: self._draw_box(debug_img, data.hybrid_person, COLOR_HYBRID_PERSON, "Final-P", thick+1)
            if data.hybrid_face: self._draw_box(debug_img, data.hybrid_face, COLOR_HYBRID_FACE, "Final-F", thick+1)
        return debug_img

    def _draw_box(self, img, info: BboxInfo, color, label, thick):
        x1, y1, x2, y2 = info.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _determine_body_type(self, kpts, scores):
        valid = sum(1 for idx in LOWER_BODY_INDICES if idx < len(scores) and scores[idx] > self.config.kpt_threshold)
        return BodyType.FULL if valid >= self.config.full_body_min_valid_lower else BodyType.UPPER
    
    def _determine_case(self, src, ref):
        if src == BodyType.FULL and ref == BodyType.FULL: return AlignmentCase.A
        elif src == BodyType.UPPER and ref == BodyType.UPPER: return AlignmentCase.B
        elif src == BodyType.FULL and ref == BodyType.UPPER: return AlignmentCase.C
        else: return AlignmentCase.D

    # ============================================================
    # [FIX] Person Bbox Calculation (Margin applied from config)
    # ============================================================
    def _keypoints_to_person_bbox(self, kpts, scores, size):
        margin = self.config.person_bbox_margin # YAML 설정값 사용
        h, w = size
        valid = kpts[scores > self.config.kpt_threshold]
        if len(valid) == 0: return BboxInfo((0,0,w,h), (w/2,h/2), max(w,h), "fallback")
        mn, mx = valid.min(0), valid.max(0)
        wd, ht = mx - mn
        mx_pad, my_pad = wd*margin, ht*margin
        x1, y1 = max(0, int(mn[0]-mx_pad)), max(0, int(mn[1]-my_pad))
        x2, y2 = min(w, int(mx[0]+mx_pad)), min(h, int(mx[1]+my_pad))
        return BboxInfo((x1,y1,x2,y2), ((x1+x2)/2, (y1+y2)/2), max(x2-x1, y2-y1), "keypoint")

    # ============================================================
    # [FIX] Face Bbox Calculation (Margin applied from config + Jawline included)
    # ============================================================
# [FIX] Face Bbox Calculation (Include Eyebrows)
    def _keypoints_to_face_bbox(self, kpts, scores):
        margin = self.config.face_bbox_margin
        
        # 1. Body face (nose, eyes, ears) - 기존
        body_face_idx = [BODY_KEYPOINTS.get(n, i) for i, n in enumerate(['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'])]
        
        # 2. Detail face landmarks (Jawline + Eyebrows) - 수정됨
        # Face 0~16: Jawline (턱선) -> 전체 인덱스 23~39
        # Face 17~21: Left Eyebrow (왼쪽 눈썹) -> 전체 인덱스 40~44
        # Face 22~26: Right Eyebrow (오른쪽 눈썹) -> 전체 인덱스 45~49
        
        # COCO-WholeBody 전체 인덱스 기준 (Face Start = 23)
        jaw_idx = list(range(23, 40))       # 턱선
        l_brow_idx = list(range(40, 45))    # 왼쪽 눈썹 (추가됨)
        r_brow_idx = list(range(45, 50))    # 오른쪽 눈썹 (추가됨)
        
        target_indices = body_face_idx + jaw_idx + l_brow_idx + r_brow_idx
        
        valid = [kpts[i] for i in target_indices if i < len(scores) and scores[i] > self.config.kpt_threshold]
        
        if len(valid) < 2: return BboxInfo((0,0,100,100), (50,50), 100, "fallback")
        
        v = np.array(valid)
        mn, mx = v.min(0), v.max(0)
        
        width = mx[0] - mn[0]
        height = mx[1] - mn[1]
        size = max(width, height)
        
        mx_pad = width * margin
        my_pad = height * margin
        
        x1 = int(mn[0] - mx_pad)
        y1 = int(mn[1] - my_pad)
        x2 = int(mx[0] + mx_pad)
        y2 = int(mx[1] + my_pad)
        
        return BboxInfo((x1, y1, x2, y2), 
                        ((x1+x2)/2, (y1+y2)/2), size, "keypoint")

    def _run_yolo_and_merge(self, image: np.ndarray, kpt_person: BboxInfo, kpt_face: BboxInfo) -> Tuple[DebugBboxData, Dict]:
        debug_data = DebugBboxData(kpt_person=kpt_person, kpt_face=kpt_face)
        yolo_log = {"person": False, "face": False}
        
        if self._yolo_person is None:
            debug_data.hybrid_person = kpt_person
            debug_data.hybrid_face = kpt_face
            return debug_data, yolo_log

        try:
            # Person
            res = self._yolo_person.predict(image, conf=self.config.yolo_person_conf, verbose=False)[0].boxes
            mask = (res.cls == 0)
            hybrid_person = kpt_person
            
            if mask.sum() > 0:
                yolo_log["person"] = True
                boxes = res.xyxy[mask].cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                yb = boxes[np.argmax(areas)].astype(int)
                yolo_info = BboxInfo((yb[0], yb[1], yb[2], yb[3]), 
                                     ((yb[0]+yb[2])/2, (yb[1]+yb[3])/2), 
                                     max(yb[2]-yb[0], yb[3]-yb[1]), "yolo")
                debug_data.yolo_person = yolo_info
                
                iou = self._calc_iou(kpt_person.bbox, yb)
                if iou > 0.5: hybrid_person = yolo_info
                else: hybrid_person = kpt_person
            debug_data.hybrid_person = hybrid_person

            # Face
            px1, py1, px2, py2 = hybrid_person.bbox
            crop = image[py1:py2, px1:px2]
            hybrid_face = kpt_face
            
            if crop.size > 0:
                f_res = self._yolo_face.predict(crop, conf=self.config.yolo_face_conf, verbose=False)[0].boxes
                if len(f_res) > 0:
                    yolo_log["face"] = True
                    fb = f_res[0].xyxy[0].cpu().numpy().astype(int)
                    fx1, fy1, fx2, fy2 = fb[0]+px1, fb[1]+py1, fb[2]+px1, fb[3]+py1
                    yolo_info = BboxInfo((fx1, fy1, fx2, fy2), 
                                         ((fx1+fx2)/2, (fy1+fy2)/2), 
                                         max(fx2-fx1, fy2-fy1), "yolo")
                    debug_data.yolo_face = yolo_info
                    hybrid_face = yolo_info
            debug_data.hybrid_face = hybrid_face

        except Exception as e:
            print(f"   ⚠️ YOLO Error: {e}")
            debug_data.hybrid_person = kpt_person
            debug_data.hybrid_face = kpt_face
            
        return debug_data, yolo_log

    def _calc_iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

    def transfer(self, source_image, reference_image, output_image_size=None):
        if isinstance(source_image, (str, Path)): src_img = load_image(source_image)
        else: src_img = source_image
        if isinstance(reference_image, (str, Path)): ref_img = load_image(reference_image)
        else: ref_img = reference_image
        
        src_h, src_w = src_img.shape[:2]; ref_h, ref_w = ref_img.shape[:2]
        
        print("\n[STEP 1] Extracting poses...")
        src_kpts, src_scores, src_idx, src_size = self.extract_pose(src_img)
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        src_type = self._determine_body_type(src_kpts, src_scores)
        ref_type = self._determine_body_type(ref_kpts, ref_scores)
        case = self._determine_case(src_type, ref_type)
        print(f"   Case {case.value} ({src_type.value} -> {ref_type.value})")
        
        print("\n[STEP 3] Bbox Calculation...")
        src_kpt_p = self._keypoints_to_person_bbox(src_kpts, src_scores, src_size)
        src_kpt_f = self._keypoints_to_face_bbox(src_kpts, src_scores)
        ref_kpt_p = self._keypoints_to_person_bbox(ref_kpts, ref_scores, ref_size)
        ref_kpt_f = self._keypoints_to_face_bbox(ref_kpts, ref_scores)
        
        if self.config.yolo_verification_enabled:
            src_debug_data, src_yolo_log = self._run_yolo_and_merge(src_img, src_kpt_p, src_kpt_f)
            ref_debug_data, ref_yolo_log = self._run_yolo_and_merge(ref_img, ref_kpt_p, ref_kpt_f)
        else:
            src_debug_data = DebugBboxData(kpt_person=src_kpt_p, kpt_face=src_kpt_f, hybrid_person=src_kpt_p, hybrid_face=src_kpt_f)
            ref_debug_data = DebugBboxData(kpt_person=ref_kpt_p, kpt_face=ref_kpt_f, hybrid_person=ref_kpt_p, hybrid_face=ref_kpt_f)
            src_yolo_log = {"person": False, "face": False}

        src_person_final = src_debug_data.hybrid_person
        src_face_final = src_debug_data.hybrid_face
        ref_face_final = ref_debug_data.hybrid_face
        
        src_debug_img = None; ref_debug_img = None
        if self.config.debug_bbox_visualization:
            src_ov = self.renderer.render(src_img, src_kpts, src_scores)
            src_debug_img = self._draw_bbox_debug_layers(src_ov, src_debug_data)
            ref_ov = self.renderer.render(ref_img, ref_kpts, ref_scores)
            ref_debug_img = self._draw_bbox_debug_layers(ref_ov, ref_debug_data)

        result = self.transfer_engine.transfer(
            src_kpts, src_scores, ref_kpts, ref_scores,
            source_image_size=(src_h, src_w), reference_image_size=(ref_h, ref_w)
        )
        trans_kpts, trans_scores = result.keypoints, result.scores
        trans_kpts, trans_scores = self._postprocess_for_case(trans_kpts, trans_scores, case, src_scores)
        
        render_h, render_w = max(src_h, ref_h)*2, max(src_w, ref_w)*2
        offset_x, offset_y = render_w//4, render_h//4
        r_kpts = trans_kpts.copy(); r_kpts[:,0]+=offset_x; r_kpts[:,1]+=offset_y
        skel = self.renderer.render_skeleton_only((render_h, render_w, 3), r_kpts, trans_scores)
        skel_crop, _ = self._crop_skeleton_tight(skel)
        
        scale = 1.0
        if self.config.face_scale_enabled:
            scale_img, scale = self._scale_skeleton(skel_crop, src_face_final.size, ref_face_final.size)
        else:
            scale_img = skel_crop
        scaled_kpts = trans_kpts.copy() * scale
        
        aligned, aligned_kpts = self._align_skeleton(
            scale_img, (src_h, src_w), src_person_final, src_face_final, case, scaled_kpts, trans_scores
        )
        padded, padded_kpts = self._add_head_padding(aligned, aligned_kpts, trans_scores)
        final, final_kpts, final_size = self._final_crop(padded, padded_kpts, self.config.crop_padding_px)
        
        align_info = AlignmentInfo(
            case=case, src_body_type=src_type, ref_body_type=ref_type,
            src_person_bbox=src_person_final, src_face_bbox=src_face_final, ref_face_bbox=ref_face_final,
            face_scale_ratio=scale, alignment_method="feet" if case==AlignmentCase.A else "face",
            yolo_log=src_yolo_log
        )
        
        return PipelineResult(
            transferred_keypoints=final_kpts, transferred_scores=trans_scores,
            source_keypoints=src_kpts, source_scores=src_scores,
            source_bone_lengths=result.source_bone_lengths,
            reference_keypoints=ref_kpts, reference_scores=ref_scores,
            skeleton_image=final, image_size=final_size,
            selected_person_idx={'source': src_idx, 'reference': ref_idx},
            processing_info={'transfer_log': result.transfer_log},
            alignment_info=align_info,
            src_debug_image=src_debug_img, ref_debug_image=ref_debug_img
        )

    def _crop_skeleton_tight(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        mask = np.any(img != 0, axis=2)
        if not np.any(mask): return img, (0, 0, img.shape[1], img.shape[0])
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return img[y1:y2+1, x1:x2+1], (x1, y1, x2+1, y2+1)
    
    def _scale_skeleton(self, img: np.ndarray, src_size: float, ref_size: float) -> Tuple[np.ndarray, float]:
        if ref_size < 1: return img, 1.0
        ratio = np.clip(src_size / ref_size, 0.3, 3.0)
        if abs(ratio - 1.0) < 0.05: return img, 1.0
        new_h, new_w = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return scaled, ratio

    def _align_skeleton(self, skeleton, src_size, src_person, src_face, case, trans_kpts, trans_scores):
        src_h, src_w = src_size
        sk_h, sk_w = skeleton.shape[:2]
        if case == AlignmentCase.A:
            x1, y1, x2, y2 = src_person.bbox
            padding_h, padding_w = max(src_h, sk_h + (src_h - y2)), max(src_w, sk_w + x1)
            canvas = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
            pose_x1, pose_y2 = x1, padding_h - (src_h - y2)
            pose_y1 = max(0, pose_y2 - sk_h)
            sk_y_start = max(0, sk_h - (pose_y2 - pose_y1))
            canvas[pose_y1:pose_y2, pose_x1:pose_x1+sk_w] = skeleton[sk_y_start:, :min(sk_w, padding_w-pose_x1)]
            adj_kpts = trans_kpts.copy()
            adj_kpts[:, 0] += pose_x1; adj_kpts[:, 1] += pose_y1 - sk_y_start
            return canvas, adj_kpts
        else:
            trans_face = self._keypoints_to_face_bbox(trans_kpts, trans_scores)
            offset_x, offset_y = src_face.center[0] - trans_face.center[0], src_face.center[1] - trans_face.center[1]
            new_x1, new_y1, new_x2, new_y2 = offset_x, offset_y, offset_x + sk_w, offset_y + sk_h
            exp_l, exp_t = max(0, -new_x1), max(0, -new_y1)
            exp_r, exp_b = max(0, new_x2 - src_w), max(0, new_y2 - src_h)
            padding_w, padding_h = int(src_w + exp_l + exp_r), int(src_h + exp_t + exp_b)
            canvas = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
            pose_x1, pose_y1 = int(new_x1 + exp_l), int(new_y1 + exp_t)
            canvas[pose_y1:pose_y1+sk_h, pose_x1:pose_x1+sk_w] = skeleton
            adj_kpts = trans_kpts.copy()
            adj_kpts[:, 0] += (offset_x + exp_l); adj_kpts[:, 1] += (offset_y + exp_t)
            return canvas, adj_kpts

    def _add_head_padding(self, img, kpts, scores):
        pad = self._calc_head_padding(kpts, scores)
        pl, pt, pr, pb = [int(p) for p in pad]
        if pl == 0 and pt == 0 and pr == 0 and pb == 0: return img, kpts
        h, w = img.shape[:2]
        padded = np.zeros((h + pt + pb, w + pl + pr, 3), dtype=np.uint8)
        padded[pt:pt+h, pl:pl+w] = img
        adj = kpts.copy(); adj[:, 0] += pl; adj[:, 1] += pt
        return padded, adj

    def _calc_head_padding(self, kpts, scores):
        nose_idx = BODY_KEYPOINTS.get('nose', 0)
        l_sh, r_sh = BODY_KEYPOINTS.get('left_shoulder', 5), BODY_KEYPOINTS.get('right_shoulder', 6)
        l_eye, r_eye = BODY_KEYPOINTS.get('left_eye', 1), BODY_KEYPOINTS.get('right_eye', 2)
        if scores[nose_idx] <= 0.1 or not (scores[l_sh] > 0.1 or scores[r_sh] > 0.1): return (0, 0, 0, 0)
        neck = (kpts[l_sh] + kpts[r_sh]) / 2 if scores[l_sh] > 0.1 and scores[r_sh] > 0.1 else (kpts[l_sh] if scores[l_sh] > 0.1 else kpts[r_sh])
        nose = kpts[nose_idx]; head_vec = nose - neck; length = np.linalg.norm(head_vec)
        if length < 1: return (0, 0, 0, 0)
        direction = head_vec / length
        head_size = np.linalg.norm(kpts[l_eye] - kpts[r_eye]) * 2.5 if scores[l_eye] > 0.1 and scores[r_eye] > 0.1 else length * 1.5
        extend = head_size * self.config.head_padding_ratio
        crown = nose + direction * extend
        offset_x, offset_y = crown[0] - nose[0], crown[1] - nose[1]
        return (max(0, -offset_x), max(0, -offset_y), max(0, offset_x), max(0, offset_y))

    def _final_crop(self, canvas, kpts, padding):
        h, w = canvas.shape[:2]
        valid_mask = np.any(kpts > 0, axis=1)
        if not np.any(valid_mask): return canvas, kpts, canvas.shape[:2]
        valid = kpts[valid_mask]
        x1 = max(0, int(np.min(valid[:, 0])) - padding)
        y1 = max(0, int(np.min(valid[:, 1])) - padding)
        x2 = min(w, int(np.max(valid[:, 0])) + padding)
        y2 = min(h, int(np.max(valid[:, 1])) + padding)
        cropped = canvas[y1:y2, x1:x2]
        adj = kpts.copy(); adj[:, 0] -= x1; adj[:, 1] -= y1
        return cropped, adj, cropped.shape[:2]

    def _postprocess_for_case(self, kpts, scores, case, src_scores):
        if case == AlignmentCase.D:
            new_scores = scores.copy()
            for idx in LOWER_BODY_INDICES:
                if idx < len(src_scores) and src_scores[idx] < self.config.kpt_threshold:
                    if idx < len(new_scores): new_scores[idx] = 0.0
            if FEET_KEYPOINTS:
                for name, idx in FEET_KEYPOINTS.items():
                    if idx < len(src_scores) and src_scores[idx] < self.config.kpt_threshold:
                        if idx < len(new_scores): new_scores[idx] = 0.0
            return kpts, new_scores
        return kpts, scores

    def extract_and_render(self, image):
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        image_size = img.shape[:2]
        kpts, scores, _, _ = self.extract_pose(img)
        json_data = convert_to_openpose_format(kpts[np.newaxis, ...], scores[np.newaxis, ...], image_size)
        skel_img = self.renderer.render_skeleton_only((image_size[0], image_size[1], 3), kpts, scores)
        overlay_img = self.renderer.render(img, kpts, scores)
        return json_data, skel_img, overlay_img