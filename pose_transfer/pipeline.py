"""
Pose Transfer Pipeline (Refactored v7.2 - Full Depth Report)

위치: pose_transfer/pipeline.py
변경사항:
- [Fix] Depth Analysis 리포트 범위를 5개 -> 17개(전신)로 확장
- [Fix] 리포트 포맷을 다른 섹션과 통일성 있게 맞춤
"""
import cv2
import numpy as np
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .extractors import DWPoseExtractorFactory, PersonFilter, RTMLIB_AVAILABLE
from .extractors.body_extractor import BodyExtractor
# [Added] BoneCalculator Import (for Report)
from .transfer.logic.body import BoneCalculator 
from .transfer import PoseTransferEngine, TransferConfig
from .refiners import HandRefiner
from .renderers import SkeletonRenderer
from .utils import load_image, convert_to_openpose_format

from .logic.bbox_manager import BboxManager
from .logic.align_manager import AlignManager
from .logic.canvas_manager import CanvasManager
from .logic.cross_filter import CrossFilter, CrossFilterConfig

@dataclass
class PipelineConfig:
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    filter_enabled: bool = True
    hand_refinement_enabled: bool = True
    cross_filter_enabled: bool = True
    depth_enabled: bool = False
    depth_model_type: str = 'depth_anything_v2_vitl'
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    min_hand_size: int = 48
    line_thickness: int = 4
    point_radius: int = 4
    kpt_threshold: float = 0.3
    auto_crop_enabled: bool = False
    canvas_padding_ratio: float = 0.0
    yolo_verification_enabled: bool = True
    person_bbox_margin: float = 0.0
    face_bbox_margin: float = 0.0
    debug_bbox_visualization: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data: return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

@dataclass
class PipelineResult:
    transferred_keypoints: np.ndarray
    transferred_scores: np.ndarray
    source_keypoints: np.ndarray
    source_scores: np.ndarray
    reference_keypoints: np.ndarray
    reference_scores: np.ndarray
    skeleton_image: np.ndarray
    image_size: Tuple[int, int]
    modified_source_image: Optional[np.ndarray] = None
    processing_info: Dict[str, Any] = None
    alignment_info: Any = None
    src_debug_image: Optional[np.ndarray] = None
    ref_debug_image: Optional[np.ndarray] = None
    src_debug_text: str = ""
    ref_debug_text: str = ""
    src_depth_map: Optional[np.ndarray] = None
    ref_depth_map: Optional[np.ndarray] = None
    
    def to_json(self) -> Dict[str, Any]:
        return convert_to_openpose_format(
            self.transferred_keypoints[np.newaxis, ...],
            self.transferred_scores[np.newaxis, ...],
            self.image_size
        )

class PoseTransferPipeline:
    def __init__(self, config: PipelineConfig, transfer_config: TransferConfig = None):
        self.config = config
        self.transfer_config = transfer_config or TransferConfig()
        
        self.bbox_mgr = BboxManager(self.config)
        self.align_mgr = AlignManager(self.config)
        self.canvas_mgr = CanvasManager(self.config)
        
        self.bone_calculator = BoneCalculator(self.config.kpt_threshold)
        self.depth_model = None
        self._init_modules()
    
    def _init_modules(self):
        if not RTMLIB_AVAILABLE: raise RuntimeError("rtmlib not installed.")
        self.extractor = DWPoseExtractorFactory.get_instance(
            backend=self.config.backend, device=self.config.device, mode=self.config.mode
        )
        self.body_extractor = None
        self.cross_filter = None
        if self.config.cross_filter_enabled:
            print("🛡️ Cross Filter Enabled")
            self.body_extractor = BodyExtractor(backend='onnxruntime', device='cpu')
            self.cross_filter = CrossFilter(CrossFilterConfig()) 
        if self.config.depth_enabled:
            self._load_depth_model()
        self.person_filter = PersonFilter(self.config.area_weight, self.config.center_weight, self.config.filter_confidence_threshold)
        self.transfer_engine = PoseTransferEngine(config=self.transfer_config)
        self.hand_refiner = HandRefiner(self.config.min_hand_size)
        self.renderer = SkeletonRenderer(line_thickness=self.config.line_thickness, point_radius=self.config.point_radius)

    def _load_depth_model(self):
        print(f"🧭 Initializing Depth Model: {self.config.depth_model_type}")
        project_root = Path(__file__).parent.parent
        repo_path = project_root / "vendor" / "Depth-Anything-V2"
        ckpt_path = repo_path / "checkpoints" / f"{self.config.depth_model_type}.pth"
        
        if not repo_path.exists():
            print(f"   ⚠️ Depth Repo missing at {repo_path}. Skipping.")
            return

        try:
            if str(repo_path) not in sys.path: sys.path.append(str(repo_path))
            from depth_anything_v2.dpt import DepthAnythingV2
            configs = { 'depth_anything_v2_vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}, 'depth_anything_v2_vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}, 'depth_anything_v2_vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}, 'depth_anything_v2_vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]} }
            conf = configs.get(self.config.depth_model_type, configs['depth_anything_v2_vitl'])
            self.depth_model = DepthAnythingV2(**conf)
            if ckpt_path.exists():
                self.depth_model.load_state_dict(torch.load(str(ckpt_path), map_location='cpu'))
                self.depth_model = self.depth_model.to(self.config.device).eval()
                print(f"   ✅ Loaded Depth Weights: {ckpt_path.name}")
            else:
                self.depth_model = None
        except Exception as e:
            print(f"   ❌ Depth Init Error: {e}")
            self.depth_model = None

    def _estimate_depth(self, image):
        if self.depth_model is None: return None, None
        try:
            raw_depth = self.depth_model.infer_image(image)
            depth_norm = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
            depth_vis = depth_norm.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            return raw_depth, depth_vis
        except: return None, None

    def _sample_depths(self, kpts, scores, depth_map):
        if depth_map is None: return None
        h, w = depth_map.shape[:2]
        z_values = np.zeros(len(kpts))
        for i, (x, y) in enumerate(kpts):
            if scores[i] > 0.1:
                ix, iy = int(x), int(y)
                if 0 <= ix < w and 0 <= iy < h:
                    z_values[i] = depth_map[iy, ix]
        return z_values

    def extract_pose(self, image, tag="Image"):
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        h, w = img.shape[:2]
        all_kpts, all_scores = self.extractor.extract(img)
        if len(all_kpts) == 0: return np.zeros((133, 2)), np.zeros(133), -1, (h, w), "No Person"
        kpts, scores, idx = all_kpts[0], all_scores[0], 0
        if len(all_kpts) > 1 and self.config.filter_enabled:
            kpts, scores, idx, _ = self.person_filter.select_main_person(all_kpts, all_scores, (h, w))
            
        cross_filter_rpt = ""
        body_kpts, body_scores = None, None
        if self.config.cross_filter_enabled and self.body_extractor:
            body_res = self.body_extractor.extract(img)
            if len(body_res[0]) > 0:
                body_kpts, body_scores = body_res[0][0], body_res[1][0]
                cross_filter_rpt = self._generate_cross_filter_table(body_kpts, body_scores, kpts, scores)
                kpts, scores, _ = self.cross_filter.filter(body_kpts, body_scores, kpts, scores)
        
        if self.config.hand_refinement_enabled:
            kpts, scores, _ = self.hand_refiner.refine_both_hands(img, kpts, scores, self.extractor)

        depth_rpt = ""
        raw_depth, _ = self._estimate_depth(img)
        z_vals = None
        if raw_depth is not None:
            z_vals = self._sample_depths(kpts, scores, raw_depth)
            depth_rpt = self._generate_depth_report(z_vals, scores)
            
        full_report = self._generate_full_report(tag, (h, w), kpts, scores, cross_filter_rpt, depth_rpt)

        return kpts, scores, idx, (h, w), full_report, raw_depth

    def _generate_full_report(self, tag, size, kpts, scores, cf_rpt, depth_rpt):
        h, w = size
        lines = []
        lines.append("="*80)
        lines.append(f"Pose Debug Report [{tag}]")
        lines.append("="*80)
        lines.append(f"\n[1] Image Information")
        lines.append(f"{'-'*80}")
        lines.append(f"Size: {w}x{h} (WxH)")
        
        valid_cnt = np.sum(scores > self.config.kpt_threshold)
        avg_score = np.mean(scores[scores > 0]) if valid_cnt > 0 else 0
        lines.append(f"\n[2] Keypoint Statistics")
        lines.append(f"{'-'*80}")
        lines.append(f"Detected: {valid_cnt}/133")
        lines.append(f"Avg Conf: {avg_score:.4f}")
        
        if cf_rpt:
            lines.append(f"\n[3] Cross-Filter Verification")
            lines.append(cf_rpt)
            
        lines.append(f"\n[4] Bone Lengths (Pixel)")
        lines.append(f"{'-'*80}")
        is_src = (tag == "SRC")
        props = self.bone_calculator.calculate(kpts, scores, is_source=is_src)
        if props and props.bone_lengths:
            for name, info in sorted(props.bone_lengths.items()):
                valid_mark = "" if info.is_valid else "(Invalid)"
                lines.append(f"{name:<30}: {info.length:8.2f} {valid_mark}")
        else:
            lines.append("No valid bones detected.")
            
        if depth_rpt:
            lines.append(f"\n[5] Depth Analysis (Z-Value)")
            lines.append(depth_rpt)
            
        return "\n".join(lines)

    def _generate_cross_filter_table(self, body_kpts, body_scores, dw_kpts, dw_scores):
        lines = [f"{'-'*80}", f"{'No':<4} {'Name':<18} {'Body':<8} {'DW':<8} {'Status'}", f"{'-'*80}"]
        names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        for i, name in enumerate(names):
            b, d = (body_scores[i], dw_scores[i]) if i < len(body_scores) else (0, dw_scores[i])
            st = "OK" if b>0.3 and d>0.3 else ("BodyLow" if d>0.3 else "BothLow")
            lines.append(f"{i:<4} {name:<18} {b:<8.3f} {d:<8.3f} {st}")
        return "\n".join(lines)

    def _generate_depth_report(self, z_vals, scores):
        lines = [f"{'-'*80}"]
        valid_z = z_vals[scores > 0.1]
        if len(valid_z) > 0:
            lines.append(f"Global Stats - Min: {np.min(valid_z):.2f}, Max: {np.max(valid_z):.2f}, Mean: {np.mean(valid_z):.2f}")
        
        lines.append(f"{'-'*80}")
        lines.append(f"{'No':<4} {'Name':<18} {'Z-Value':<10} {'Conf':<10}")
        
        # [Fix] 5개 -> 17개(전신)로 확장
        coco_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip", 
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        for idx, name in enumerate(coco_names):
            if idx < len(z_vals):
                z = z_vals[idx]
                conf = scores[idx]
                lines.append(f"{idx:<4} {name:<18} {z:<10.2f} {conf:<10.2f}")
                
        return "\n".join(lines)

    def _get_ground_y(self, kpts, scores):
        """발바닥 Y 좌표 반환 (가장 아래쪽 발 키포인트)"""
        foot_indices = [15, 16, 17, 18, 19, 20, 21, 22]
        valid_ys = []
        for idx in foot_indices:
            if idx < len(kpts) and scores[idx] > 0.1:
                valid_ys.append(kpts[idx][1])
        return max(valid_ys) if valid_ys else 0

    def _sync_scale_to_source_face(self, trans_kpts, trans_scores, src_face, align_by_feet=True):
        """
        Source 얼굴 크기에 맞춰 전이된 키포인트 스케일 동기화
        
        정책:
        - align_by_feet=True (발 정렬): 얼굴 관련 키포인트만 Pivot 기준 스케일링
        - align_by_feet=False (얼굴 정렬): 전체 키포인트 동일 배율 스케일링
        """
        # Trans 얼굴 bbox 계산
        current_trans_face = self.bbox_mgr._kpt_to_face_public(trans_kpts, trans_scores)
        
        # size = max(width, height) 로 계산
        trans_face_size = max(current_trans_face.width, current_trans_face.height)
        src_face_size = max(src_face.width, src_face.height)
        
        if trans_face_size <= 1 or src_face_size <= 1:
            return trans_kpts, 1.0

        scale_factor = float(np.clip(src_face_size / trans_face_size, 0.5, 2.0))
        if abs(scale_factor - 1.0) < 0.01:
            return trans_kpts, 1.0

        scaled = trans_kpts.copy()

        # 발 정렬 모드: 어깨(Neck) 기준 Pivot으로 전체 스케일링
        if align_by_feet:
            LS, RS = 5, 6
            if trans_scores[LS] > 0.1 and trans_scores[RS] > 0.1:
                pivot = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
            else:
                pivot = np.array(current_trans_face.center, dtype=np.float32)

            # 전체 키포인트를 Pivot 기준으로 스케일링
            for idx in range(len(trans_scores)):
                if trans_scores[idx] > 0.1:
                    scaled[idx] = pivot + (scaled[idx] - pivot) * scale_factor

            return scaled, scale_factor

        # 얼굴 정렬 모드: 전체 스켈레톤 스케일링
        scaled *= scale_factor
        return scaled, scale_factor

    def _sync_scale_to_source_face_feet_pivot(self, trans_kpts, trans_scores, src_face):
        """
        발을 Pivot으로 전체 스켈레톤 스케일링
        - 발 위치 고정, 위쪽으로 확대/축소
        - scale < 1: 스켈레톤 축소 (머리가 내려감)
        - scale > 1: 스켈레톤 확대 (머리가 올라감)
        """
        # Trans 얼굴 bbox 계산
        current_trans_face = self.bbox_mgr._kpt_to_face_public(trans_kpts, trans_scores)
        
        # size = max(width, height)
        trans_face_size = max(current_trans_face.width, current_trans_face.height)
        src_face_size = max(src_face.width, src_face.height)
        
        if trans_face_size <= 1 or src_face_size <= 1:
            return trans_kpts, 1.0

        scale_factor = float(np.clip(src_face_size / trans_face_size, 0.3, 3.0))
        if abs(scale_factor - 1.0) < 0.01:
            return trans_kpts, 1.0

        scaled = trans_kpts.copy()
        
        # Pivot: 발바닥 위치 (가장 아래쪽 Y)
        foot_y = self._get_ground_y(trans_kpts, trans_scores)
        if foot_y <= 0:
            # 발이 없으면 발목 사용
            LA, RA = 15, 16
            if trans_scores[LA] > 0.1:
                foot_y = trans_kpts[LA][1]
            elif trans_scores[RA] > 0.1:
                foot_y = trans_kpts[RA][1]
            else:
                return trans_kpts, 1.0  # 발 정보 없음
        
        # X축 중심
        valid_xs = trans_kpts[trans_scores > 0][:, 0]
        pivot_x = np.mean(valid_xs) if len(valid_xs) > 0 else 0
        pivot = np.array([pivot_x, foot_y])
        
        # 전체 키포인트를 발 Pivot 기준으로 스케일링
        for idx in range(len(trans_scores)):
            if trans_scores[idx] > 0.1:
                scaled[idx] = pivot + (scaled[idx] - pivot) * scale_factor

        return scaled, scale_factor

    def _get_body_height(self, kpts, scores):
        """어깨 중심 ~ 발바닥 거리 (Body Height)"""
        LS, RS = 5, 6
        if scores[LS] < 0.1 or scores[RS] < 0.1:
            return 0
        shoulder_y = (kpts[LS][1] + kpts[RS][1]) / 2
        foot_y = self._get_ground_y(kpts, scores)
        if foot_y <= 0:
            # 발바닥 없으면 발목 사용
            LA, RA = 15, 16
            if scores[LA] > 0.1:
                foot_y = kpts[LA][1]
            elif scores[RA] > 0.1:
                foot_y = kpts[RA][1]
        if foot_y <= shoulder_y:
            return 0
        return foot_y - shoulder_y

    def _sync_body_height(self, trans_kpts, trans_scores, src_kpts, src_scores):
        """
        Src 사람 높이에 맞춰 Trans 스켈레톤 스케일링 (어깨 Pivot)
        - 어깨 고정, 아래쪽으로 확장하여 발 위치를 Src에 맞춤
        """
        src_height = self._get_body_height(src_kpts, src_scores)
        trans_height = self._get_body_height(trans_kpts, trans_scores)
        
        if src_height <= 0 or trans_height <= 0:
            return trans_kpts, 1.0
        
        scale_factor = src_height / trans_height
        scale_factor = float(np.clip(scale_factor, 0.5, 3.0))
        
        if abs(scale_factor - 1.0) < 0.01:
            return trans_kpts, 1.0
        
        scaled = trans_kpts.copy()
        
        # Pivot: 어깨 중심 (어깨 고정, 아래로 확장)
        LS, RS = 5, 6
        if trans_scores[LS] > 0.1 and trans_scores[RS] > 0.1:
            shoulder_y = (trans_kpts[LS][1] + trans_kpts[RS][1]) / 2
            pivot_x = (trans_kpts[LS][0] + trans_kpts[RS][0]) / 2
        else:
            return trans_kpts, 1.0
        
        pivot = np.array([pivot_x, shoulder_y])
        
        # 전체 키포인트를 어깨 Pivot 기준으로 스케일링
        for idx in range(len(trans_scores)):
            if trans_scores[idx] > 0.1:
                scaled[idx] = pivot + (scaled[idx] - pivot) * scale_factor
        
        return scaled, scale_factor

    def transfer(self, source_image, reference_image):
        if isinstance(source_image, (str, Path)): src_img = load_image(source_image)
        else: src_img = source_image
        if isinstance(reference_image, (str, Path)): ref_img = load_image(reference_image)
        else: ref_img = reference_image
        
        src_kpts, src_scores, src_idx, src_size, src_rpt, src_raw = self.extract_pose(src_img, "SRC")
        ref_kpts, ref_scores, ref_idx, ref_size, ref_rpt, ref_raw = self.extract_pose(ref_img, "REF")
        
        _, src_vis = self._estimate_depth(src_img)
        _, ref_vis = self._estimate_depth(ref_img)
        
        src_depths = self._sample_depths(src_kpts, src_scores, src_raw)
        ref_depths = self._sample_depths(ref_kpts, ref_scores, ref_raw)
        
        src_bbox, src_face, src_dbg = self.bbox_mgr.get_bboxes(src_img, src_kpts, src_scores)
        ref_bbox, ref_face, ref_dbg = self.bbox_mgr.get_bboxes(ref_img, ref_kpts, ref_scores)
        layout = self.align_mgr.analyze_layout(src_bbox, ref_bbox, src_kpts, src_scores, ref_kpts, ref_scores)
        
        result = self.transfer_engine.transfer(
            src_kpts, src_scores, ref_kpts, ref_scores,
            src_size, ref_size,
            layout=layout,
            source_depths=src_depths,
            reference_depths=ref_depths
        )
        
        trans_kpts, trans_scores = result.keypoints, result.scores
        
        # 디버그: Engine 출력 직후 어깨/발 위치
        LS, RS = 5, 6
        LA, RA = 15, 16  # Ankle
        if trans_scores[LS] > 0.1 and trans_scores[RS] > 0.1:
            print(f"   [DEBUG] After Engine - Shoulder Y: {(trans_kpts[LS][1] + trans_kpts[RS][1])/2:.0f}")
        if trans_scores[LA] > 0.1 and trans_scores[RA] > 0.1:
            print(f"   [DEBUG] After Engine - Ankle Y: {(trans_kpts[LA][1] + trans_kpts[RA][1])/2:.0f}")
        
        # [Post-Processing 1] Layout Offset 적용 (Ref → Src 위치 이동)
        # Engine에서 Ref 좌표계로 생성되었으므로, 먼저 Src 위치로 이동
        if hasattr(layout, 'offset_vector') and layout.offset_vector is not None:
            offset_vec = np.array(layout.offset_vector, dtype=np.float32)
            mask = (trans_scores > 0)
            trans_kpts[mask] += offset_vec
            print(f"   🚚 [Pipeline] Layout Offset Applied: {offset_vec.astype(int)}")
            # 디버그: Offset 적용 후 어깨 위치
            if trans_scores[LS] > 0.1 and trans_scores[RS] > 0.1:
                print(f"   [DEBUG] After Offset - Shoulder Y: {(trans_kpts[LS][1] + trans_kpts[RS][1])/2:.0f}")
        
        # [Post-Processing 2] Body Height 기반 스케일링
        # Src 사람 높이에 맞춰 Trans 스켈레톤 스케일링 (어깨 Pivot)
        trans_kpts, body_scale = self._sync_body_height(trans_kpts, trans_scores, src_kpts, src_scores)
        if abs(body_scale - 1.0) > 0.01:
            print(f"   📏 [Pipeline] Body Scale Applied: {body_scale:.3f}")
        
        # [Post-Processing 3] 발 정렬 미세 조정
        src_foot_y = self._get_ground_y(src_kpts, src_scores)
        trans_foot_y = self._get_ground_y(trans_kpts, trans_scores)
        # 스케일링 후 발 위치가 Src와 어긋날 수 있음
        src_foot_y = self._get_ground_y(src_kpts, src_scores)
        trans_foot_y = self._get_ground_y(trans_kpts, trans_scores)
        
        if src_foot_y > 0 and trans_foot_y > 0:
            offset_y = src_foot_y - trans_foot_y
            # X축은 Src bbox 중앙과 Trans 중앙 맞춤
            src_center_x = (src_bbox.x1 + src_bbox.x2) / 2
            trans_xs = trans_kpts[trans_scores > 0][:, 0]
            trans_center_x = (np.min(trans_xs) + np.max(trans_xs)) / 2 if len(trans_xs) > 0 else src_center_x
            offset_x = src_center_x - trans_center_x
            
            feet_align_offset = np.array([offset_x, offset_y])
            mask = (trans_scores > 0)
            trans_kpts[mask] += feet_align_offset
            print(f"   🦶 [Pipeline] Feet Align Offset: {feet_align_offset.astype(int)}")
        
        final_img, final_kpts, final_size = self.canvas_mgr.expand_canvas_to_fit(src_img, trans_kpts, trans_scores, padding_ratio=self.config.canvas_padding_ratio)
        skeleton_img = self.renderer.render_skeleton_only((final_size[0], final_size[1], 3), final_kpts, trans_scores)
        
        src_debug_img = None
        ref_debug_img = None
        if self.config.debug_bbox_visualization:
            src_debug_img = self.bbox_mgr.draw_debug(src_img.copy(), src_dbg)
            ref_debug_img = self.bbox_mgr.draw_debug(ref_img.copy(), ref_dbg)

        return PipelineResult(
            transferred_keypoints=final_kpts,
            transferred_scores=trans_scores,
            source_keypoints=src_kpts, source_scores=src_scores,
            reference_keypoints=ref_kpts, reference_scores=ref_scores,
            skeleton_image=skeleton_img,
            image_size=final_size,
            modified_source_image=final_img,
            processing_info={'transfer_log': result.transfer_log},
            alignment_info=layout,
            src_debug_image=src_debug_img,
            ref_debug_image=ref_debug_img,
            src_debug_text=src_rpt,
            ref_debug_text=ref_rpt,
            src_depth_map=src_vis,
            ref_depth_map=ref_vis
        )