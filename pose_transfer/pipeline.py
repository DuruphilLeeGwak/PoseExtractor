"""
포즈 전이 파이프라인 v15 (Source Scale Priority)
- F_F 이외의 케이스(H_F, H_H 등)에서 Source 얼굴 크기를 기준으로 강력하게 스케일링
- 캔버스 밖으로 나가는 포즈는 CanvasManager를 통해 패딩 처리
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
from .extractors.keypoint_constants import BODY_KEYPOINTS
from .transfer import PoseTransferEngine, TransferConfig, FallbackStrategy
from .refiners import HandRefiner
from .renderers import SkeletonRenderer
from .utils import load_config, convert_to_openpose_format, load_image

# Logic Modules
from .logic import (
    BboxManager, AlignManager, PostProcessor, CanvasManager,
    AlignmentCase, BodyType, DebugBboxData, BboxInfo,
    COLOR_KPT_BBOX, COLOR_YOLO_BBOX, COLOR_HYBRID_PERSON, COLOR_HYBRID_FACE
)

# Ghost Filter
from .logic.ghost_filter import GhostFilter, GhostFilterConfig, filter_ghost_keypoints


@dataclass
class PipelineConfig:
    """파이프라인 통합 설정"""
    backend: str = 'onnxruntime'
    device: str = 'cuda'
    mode: str = 'performance'
    to_openpose: bool = False
    
    # Filter
    filter_enabled: bool = True
    area_weight: float = 0.6
    center_weight: float = 0.4
    filter_confidence_threshold: float = 0.3
    
    # Hand
    hand_refinement_enabled: bool = True
    min_hand_size: int = 48
    
    # Fallback
    fallback_enabled: bool = True
    
    # Transfer
    transfer_confidence_threshold: float = 0.3
    ghost_legs_clipping_enabled: bool = True
    lower_body_confidence_threshold: float = 2.0
    lower_body_margin_ratio: float = 0.10
    visibility_margin: float = 0.2
    
    # Rendering
    line_thickness: int = 4
    face_line_thickness: int = 2
    hand_line_thickness: int = 2
    point_radius: int = 4
    kpt_threshold: float = 0.3
    
    # Output / Crop
    auto_crop_enabled: bool = True
    crop_padding_px: int = 50
    head_padding_ratio: float = 1.0
    canvas_padding_ratio: float = 0.1
    
    # Alignment / Logic
    full_body_min_valid_lower: int = 4
    ghost_score_threshold: float = 2.0
    yolo_verification_enabled: bool = True
    yolo_person_conf: float = 0.5
    yolo_face_conf: float = 0.3
    face_scale_enabled: bool = True
    
    # Ghost Filter (NEW)
    ghost_filter_enabled: bool = True
    ghost_body_score_threshold: float = 2.0
    ghost_hand_score_threshold: float = 1.5
    ghost_wrist_score_threshold: float = 2.0
    
    # Bbox Margin
    person_bbox_margin: float = 0.0
    face_bbox_margin: float = 0.0
    
    # Debug
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
        bbox = config.get('bbox', {})
        ghost = config.get('ghost_filter', {})
        
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
            ghost_legs_clipping_enabled=transfer.get('ghost_legs_clipping_enabled', True),
            lower_body_confidence_threshold=transfer.get('lower_body_confidence_threshold', 2.0),
            lower_body_margin_ratio=transfer.get('lower_body_margin_ratio', 0.10),
            visibility_margin=transfer.get('visibility_margin', 0.2),
            line_thickness=rendering.get('line_thickness', 4),
            face_line_thickness=rendering.get('face_line_thickness', 2),
            hand_line_thickness=rendering.get('hand_line_thickness', 2),
            point_radius=rendering.get('point_radius', 4),
            kpt_threshold=rendering.get('kpt_threshold', 0.3),
            auto_crop_enabled=output.get('auto_crop_enabled', True),
            crop_padding_px=output.get('crop_padding_px', 50),
            head_padding_ratio=output.get('head_padding_ratio', 1.0),
            canvas_padding_ratio=output.get('canvas_padding_ratio', 0.1),
            full_body_min_valid_lower=alignment.get('full_body_min_valid_lower', 4),
            ghost_score_threshold=alignment.get('ghost_score_threshold', 2.0),
            yolo_verification_enabled=alignment.get('yolo_verification_enabled', True),
            yolo_person_conf=alignment.get('yolo_person_conf', 0.5),
            yolo_face_conf=alignment.get('yolo_face_conf', 0.3),
            face_scale_enabled=alignment.get('face_scale_enabled', True),
            ghost_filter_enabled=ghost.get('enabled', True),
            ghost_body_score_threshold=ghost.get('body_score_threshold', 2.0),
            ghost_hand_score_threshold=ghost.get('hand_score_threshold', 1.5),
            ghost_wrist_score_threshold=ghost.get('wrist_score_threshold', 2.0),
            person_bbox_margin=bbox.get('person_margin', 0.0),
            face_bbox_margin=bbox.get('face_margin', 0.0),
            debug_bbox_visualization=debug.get('bbox_visualization', False),
            viz_kpt_bbox=debug.get('visualize_keypoint_bbox', True),
            viz_yolo_bbox=debug.get('visualize_yolo_bbox', True),
            viz_hybrid_bbox=debug.get('visualize_hybrid_bbox', True),
        )

@dataclass
class AlignmentInfo:
    case: AlignmentCase
    src_body_type: BodyType
    ref_body_type: BodyType
    src_person_bbox: Any
    src_face_bbox: Any
    ref_face_bbox: Any
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
    modified_source_image: Optional[np.ndarray] = None
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
        
        self.bbox_mgr = BboxManager(self.config)
        self.align_mgr = AlignManager(self.config)
        self.post_proc = PostProcessor(self.config)
        self.canvas_mgr = CanvasManager(self.config)
        
        # Ghost Filter 초기화
        self.ghost_filter = GhostFilter(GhostFilterConfig(
            body_score_threshold=self.config.ghost_body_score_threshold,
            hand_score_threshold=self.config.ghost_hand_score_threshold,
            wrist_score_threshold=self.config.ghost_wrist_score_threshold,
            ghost_score_threshold=self.config.ghost_score_threshold,
        ))
        
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
        transfer_config = TransferConfig(
            confidence_threshold=self.config.transfer_confidence_threshold,
            visibility_margin=self.config.visibility_margin
        )
        self.transfer_engine = PoseTransferEngine(config=transfer_config, yaml_config=self.yaml_config)
        self.fallback_strategy = FallbackStrategy(confidence_threshold=self.config.transfer_confidence_threshold)
        self.hand_refiner = HandRefiner(min_hand_size=self.config.min_hand_size, confidence_threshold=self.config.transfer_confidence_threshold)
        self.renderer = SkeletonRenderer(
            line_thickness=self.config.line_thickness, point_radius=self.config.point_radius,
            kpt_threshold=self.config.kpt_threshold, face_line_thickness=self.config.face_line_thickness,
            hand_line_thickness=self.config.hand_line_thickness
        )

    def extract_pose(self, image: Union[np.ndarray, str, Path], filter_person: bool = True) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int,int]]:
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        image_size = img.shape[:2]
        all_kpts, all_scores = self.extractor.extract(img)
        if len(all_kpts) == 0: return np.zeros((133, 2)), np.zeros(133), -1, image_size
        if filter_person and self.config.filter_enabled and len(all_kpts) > 1:
            kpts, scores, idx, _ = self.person_filter.select_main_person(all_kpts, all_scores, image_size)
        else: kpts, scores, idx = all_kpts[0], all_scores[0], 0
        if self.config.hand_refinement_enabled:
            kpts, scores, _ = self.hand_refiner.refine_both_hands(img, kpts, scores, self.extractor)
        return kpts, scores, idx, image_size

    def _apply_ghost_filter(self, kpts: np.ndarray, scores: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        if not self.config.ghost_filter_enabled:
            return scores
        return self.ghost_filter.filter(kpts, scores, image_size)

    def transfer(self, source_image, reference_image, output_image_size=None):
        print("\n" + "#"*70)
        print("# 🔍 [DEBUG] PoseTransferPipeline.transfer() START")
        print("#"*70)
        
        if isinstance(source_image, (str, Path)): src_img = load_image(source_image)
        else: src_img = source_image
        if isinstance(reference_image, (str, Path)): ref_img = load_image(reference_image)
        else: ref_img = reference_image
        
        src_h, src_w = src_img.shape[:2]; ref_h, ref_w = ref_img.shape[:2]
        print(f"\n📐 Image Sizes: src={src_w}x{src_h}, ref={ref_w}x{ref_h}")
        
        print("\n" + "-"*50)
        print("[STEP 1] Extracting poses...")
        print("-"*50)
        src_kpts, src_scores, src_idx, src_size = self.extract_pose(src_img)
        ref_kpts, ref_scores, ref_idx, ref_size = self.extract_pose(ref_img)
        
        # 추출 직후 하반신 점수 확인
        print("\n📊 Extracted Keypoints - Lower Body Scores:")
        lower_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        for name in lower_names:
            idx = BODY_KEYPOINTS.get(name, -1)
            if idx >= 0:
                src_score = src_scores[idx]
                ref_score = ref_scores[idx]
                print(f"   {name:15}: src={src_score:.3f}, ref={ref_score:.3f}")
        
        print("\n" + "-"*50)
        print("[STEP 2] Determining Body Type...")
        print("-"*50)
        src_type, ref_type, case = self.align_mgr.determine_case(src_kpts, src_scores, ref_kpts, ref_scores)
        print(f"   Result: Case {case.value} ({src_type.value} → {ref_type.value})")
        
        print("\n" + "-"*50)
        print("[STEP 3] Bbox Calculation...")
        print("-"*50)
        src_person, src_face, src_debug = self.bbox_mgr.get_bboxes(src_img, src_kpts, src_scores)
        ref_person, ref_face, ref_debug = self.bbox_mgr.get_bboxes(ref_img, ref_kpts, ref_scores)
        print(f"   src_person: {src_person.bbox}")
        print(f"   src_face: {src_face.bbox}")
        
        # Debug 이미지 생성 (Ghost Filter 적용)
        src_debug_img = None; ref_debug_img = None
        if self.config.debug_bbox_visualization:
            src_filtered = self._apply_ghost_filter(src_kpts, src_scores, src_size)
            ref_filtered = self._apply_ghost_filter(ref_kpts, ref_scores, ref_size)
            
            src_ov = self.renderer.render(src_img, src_kpts, src_filtered)
            src_debug_img = self.bbox_mgr.draw_debug(src_ov, src_debug)
            ref_ov = self.renderer.render(ref_img, ref_kpts, ref_filtered)
            ref_debug_img = self.bbox_mgr.draw_debug(ref_ov, ref_debug)

        print("\n" + "-"*50)
        print("[STEP 4] Transferring...")
        print("-"*50)
        result = self.transfer_engine.transfer(
            src_kpts, src_scores, ref_kpts, ref_scores,
            source_image_size=(src_h, src_w), reference_image_size=(ref_h, ref_w), 
            alignment_case=case.value
        )
        trans_kpts, trans_scores = result.keypoints, result.scores
        
        print("\n" + "-"*50)
        print("[STEP 5] Post-processing Keys...")
        print("-"*50)
        trans_kpts, trans_scores = self.post_proc.process_by_case(trans_kpts, trans_scores, case, src_scores)
        
        print("\n" + "-"*50)
        print("[STEP 7] Scaling (Size Matching)...")
        print("-"*50)
        
        # [수정] F_F가 아닌 경우(H_F, H_H, F_H) Source 얼굴 크기에 강제 동기화
        scale_factor = 1.0
        
        if case != AlignmentCase.F_F:
            # 1. 현재 전이된 결과물(Engine 출력)의 얼굴 BBox 계산
            #    (Engine 내부에서 이미 Global Scale이 적용된 상태임)
            current_trans_face = self.bbox_mgr._kpt_to_face_public(trans_kpts, trans_scores)
            
            # 2. 크기 검증 (0으로 나누기 방지)
            if current_trans_face.size > 1 and src_face.size > 1:
                # 3. 보정 비율 계산: (목표 Src 크기) / (현재 Trans 크기)
                #    Src/Ref를 쓰는 게 아니라, 현재 결과물을 Src에 맞춤
                scale_factor = src_face.size / current_trans_face.size
                
                # 안전 장치: 너무 극단적인 스케일링 방지 (0.5배 ~ 2.0배 사이로만 보정)
                # Engine이 이미 얼추 맞췄을 것이므로 보정치는 1.0 근처여야 함
                scale_factor = np.clip(scale_factor, 0.5, 2.0)
                
                print(f"   Case {case.value}: Adjusting Scale to Match Src Face")
                print(f"   Src Face Size: {src_face.size:.1f}")
                print(f"   Cur Trans Face Size: {current_trans_face.size:.1f}")
                print(f"   >>> Adjustment Scale: {scale_factor:.4f}")
                
                # 4. 스케일 적용 (0,0 기준 확대 -> Step 8에서 정렬됨)
                trans_kpts *= scale_factor
            else:
                print("   ⚠️ Face size too small for scaling, skipping.")
        else:
            print(f"   Case {case.value}: Using Global Scale (Shoulder/Body based)")

        # AlignInfo에 기록하기 위해 scale 변수 업데이트
        scale = scale_factor

        print("\n" + "-"*50)
        print("[STEP 8] Aligning (Anchoring)...")
        print("-"*50)
        
        # [수정] 정렬 로직 (스케일링 된 좌표를 Source 기준점(발 or 얼굴)으로 이동)
        trans_kpts = self.align_mgr.align_coordinates(
            trans_kpts, trans_scores, case, src_person, src_face,
            lambda k, s: self.bbox_mgr._kpt_to_face_public(k, s) 
        )
        
        print("\n" + "-"*50)
        print("[STEP 9] Head Padding...")
        print("-"*50)
        head_pad = self.post_proc.apply_head_padding(trans_kpts, trans_scores)
        print(f"   head_pad: {head_pad:.1f}")
        
        print("\n" + "-"*50)
        print("[STEP 10] Canvas Expansion...")
        print("-"*50)
        # 캔버스 확장: Source 이미지 리사이징 없이, 포즈가 튀어나가면 패딩 추가
        final_src_img, final_kpts, final_size = self.canvas_mgr.expand_canvas_to_fit(
            src_img, trans_kpts, trans_scores, head_pad_px=head_pad
        )
        final_h, final_w = final_size

        print("\n" + "-"*50)
        print("[RENDER] Skeleton...")
        print("-"*50)
        skeleton_image = self.renderer.render_skeleton_only(
            (final_h, final_w, 3), final_kpts, trans_scores
        )
        print(f"   skeleton_image size: {skeleton_image.shape}")
        
        align_info = AlignmentInfo(
            case=case, src_body_type=src_type, ref_body_type=ref_type,
            src_person_bbox=src_person, src_face_bbox=src_face, ref_face_bbox=ref_face,
            face_scale_ratio=scale, alignment_method="feet" if case==AlignmentCase.F_F else "face",
            yolo_log=src_debug.yolo_person is not None
        )
        
        print("\n" + "#"*70)
        print("# 🔍 [DEBUG] PoseTransferPipeline.transfer() END")
        print("#"*70 + "\n")
        
        return PipelineResult(
            transferred_keypoints=final_kpts, transferred_scores=trans_scores,
            source_keypoints=src_kpts, source_scores=src_scores,
            reference_keypoints=ref_kpts, reference_scores=ref_scores,
            source_bone_lengths=result.source_bone_lengths,
            skeleton_image=skeleton_image, image_size=final_size,
            modified_source_image=final_src_img,
            selected_person_idx={'source': src_idx, 'reference': ref_idx},
            processing_info={'transfer_log': result.transfer_log},
            alignment_info=align_info,
            src_debug_image=src_debug_img, ref_debug_image=ref_debug_img
        )

    def extract_and_render(self, image):
        """
        이미지에서 포즈 추출 및 렌더링 (범용 Ghost Filter 적용!)
        """
        if isinstance(image, (str, Path)): img = load_image(image)
        else: img = image
        image_size = img.shape[:2]
        kpts, scores, _, _ = self.extract_pose(img)
        
        # [NEW] 범용 Ghost Filter 적용
        print("\n🔍 [Ghost Filter] Applying to extracted pose...")
        filtered_scores = self._apply_ghost_filter(kpts, scores, image_size)
        
        # 필터링된 scores로 JSON 및 렌더링
        json_data = convert_to_openpose_format(kpts[np.newaxis, ...], filtered_scores[np.newaxis, ...], image_size)
        skel_img = self.renderer.render_skeleton_only((image_size[0], image_size[1], 3), kpts, filtered_scores)
        overlay_img = self.renderer.render(img, kpts, filtered_scores)
        
        return json_data, skel_img, overlay_img