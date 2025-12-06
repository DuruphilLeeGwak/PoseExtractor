"""
Pose Transfer Engine Module (Final Fix)
- Force generate lower body if reference has valid legs (even if source is half-body)
- Remove invisible leg clipping (CanvasManager handles expansion)
"""
import numpy as np
from typing import Dict, Tuple, Optional

from ..analyzers.bone_calculator import BoneCalculator
from ..analyzers.direction_extractor import DirectionExtractor
from ..utils.geometry import calculate_distance
from ..extractors.keypoint_constants import BODY_KEYPOINTS

from .config import TransferConfig, TransferResult, FaceRenderingConfig
from .logic import BodyTransfer, FaceTransfer, HandTransfer

class PoseTransferEngine:
    def __init__(self, config: Optional[TransferConfig] = None, yaml_config: Optional[dict] = None):
        self.config = config or TransferConfig()
        
        # Config 로드 (Face Rendering)
        if yaml_config:
            if 'face_rendering' in yaml_config:
                self.config.face_rendering = FaceRenderingConfig.from_dict(yaml_config['face_rendering'])
        
        # 모듈 초기화
        self.bone_calculator = BoneCalculator(confidence_threshold=self.config.confidence_threshold)
        self.direction_extractor = DirectionExtractor(confidence_threshold=self.config.confidence_threshold)
        
        # 로직 분리 (Body, Face, Hand)
        self.body_logic = BodyTransfer()
        self.face_logic = FaceTransfer(self.config)
        self.hand_logic = HandTransfer()

    def transfer(
        self,
        source_keypoints: np.ndarray, source_scores: np.ndarray,
        reference_keypoints: np.ndarray, reference_scores: np.ndarray,
        source_image_size: Optional[Tuple[int, int]] = None,
        reference_image_size: Optional[Tuple[int, int]] = None,
        target_image_size: Optional[Tuple[int, int]] = None
    ) -> TransferResult:
        
        # 1. 이미지 크기 추정
        if source_image_size is None:
            max_y = np.max(source_keypoints[:, 1])
            source_image_size = (int(max_y * 1.1), int(np.max(source_keypoints[:, 0])))
        src_h, src_w = source_image_size

        # 2. 하반신 유효성 체크 (Reference만 체크)
        # Source의 다리 유무는 전이 여부 결정에 영향을 주지 않음 (강제 생성)
        ref_lower_valid = True
        if reference_image_size:
            ref_lower_valid = self._check_lower_body_valid(reference_keypoints, reference_scores, reference_image_size[0])
        
        # Reference의 무릎 점수가 너무 낮으면 그리지 않음 (진짜 없는 경우)
        ref_knee_score = min(reference_scores[BODY_KEYPOINTS['left_knee']], reference_scores[BODY_KEYPOINTS['right_knee']])
        if ref_knee_score < 0.1:
            ref_lower_valid = False

        # 3. 데이터 추출
        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        reference_directions = self.direction_extractor.extract(reference_keypoints, reference_scores)
        
        global_scale = self._calculate_global_scale(source_proportions, reference_keypoints, reference_scores)
        corrected_lengths = self._correct_bone_lengths(source_proportions, global_scale, reference_keypoints)

        # 4. 결과 배열 초기화
        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2))
        trans_scores = np.zeros(num_kpts)
        transfer_log = {}
        processed = set()

        # 5. 실행 (Body -> Face -> Hands)
        # [Body: Upper]
        self.body_logic.transfer_shoulders(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, processed, transfer_log
        )
        self.body_logic.transfer_torso(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, global_scale, processed, transfer_log
        )
        self.body_logic.transfer_chain(
            trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, global_scale, processed, transfer_log, is_lower=False
        )
        
        # [Body: Lower] (조건부 강제 생성)
        # Reference에 다리가 있다면, Source 상태와 무관하게 무조건 생성 시도
        if ref_lower_valid:
            print("   🦵 [Transfer] Generating Lower Body (Forced by Reference)")
            self.body_logic.transfer_chain(
                trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, global_scale, processed, transfer_log, is_lower=True
            )
        else:
            print("   🚫 [Transfer] Skipping Lower Body (Reference invalid)")

        # [Face]
        if self.config.use_face:
            self.face_logic.transfer(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, transfer_log)

        # [Hands]
        if self.config.use_hands:
            self.hand_logic.transfer(trans_kpts, trans_scores, reference_keypoints, reference_scores, global_scale, transfer_log)

        # [REMOVED] self._clip_invisible_legs(...)
        # 화면 밖 클리핑 로직 제거 -> CanvasManager가 처리함

        return TransferResult(trans_kpts, trans_scores, corrected_lengths, {}, transfer_log)

    def _check_lower_body_valid(self, kpts, scores, img_h):
        # 하반신이 믿을만한지 체크 (입력 데이터 검증용)
        indices = [BODY_KEYPOINTS['left_knee'], BODY_KEYPOINTS['right_knee']]
        max_score = max([scores[i] for i in indices])
        
        # 점수가 너무 낮으면 무효
        if max_score < self.config.lower_body_confidence_threshold: return False
        
        # 이미지 바닥에 너무 붙어있으면(Ghost Leg) 무효
        margin = img_h * self.config.lower_body_margin_ratio
        limit = img_h - margin
        l_y = kpts[BODY_KEYPOINTS['left_knee']][1]
        r_y = kpts[BODY_KEYPOINTS['right_knee']][1]
        
        if (l_y > limit and scores[BODY_KEYPOINTS['left_knee']] > 0.1) or \
           (r_y > limit and scores[BODY_KEYPOINTS['right_knee']] > 0.1):
            return False
            
        return True
    
    def _calculate_global_scale(self, src_props, ref_kpts, ref_scores):
        src_w = src_props.shoulder_width
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        if src_w > 0 and ref_scores[l_sh] > 0.3:
            ref_w = calculate_distance(ref_kpts[l_sh], ref_kpts[r_sh])
            return src_w / ref_w if ref_w > 0 else 1.0
        return 1.0
    
    def _correct_bone_lengths(self, props, scale, ref_kpts):
        lengths = {}
        for n, i in props.bone_lengths.items():
            if i.is_valid: lengths[n] = i.length
        return lengths