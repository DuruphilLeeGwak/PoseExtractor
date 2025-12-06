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
        
        # Config 로드
        if yaml_config:
            if 'face_rendering' in yaml_config:
                self.config.face_rendering = FaceRenderingConfig.from_dict(yaml_config['face_rendering'])
        
        # 모듈 초기화
        self.bone_calculator = BoneCalculator(confidence_threshold=self.config.confidence_threshold)
        self.direction_extractor = DirectionExtractor(confidence_threshold=self.config.confidence_threshold)
        
        # 로직 분리
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
        src_h, _ = source_image_size

        # 2. 하반신 유효성 체크 (Input Validation Only)
        src_lower_valid = self._check_lower_body_valid(source_keypoints, source_scores, src_h)
        ref_lower_valid = True
        if reference_image_size:
            ref_lower_valid = self._check_lower_body_valid(reference_keypoints, reference_scores, reference_image_size[0])

        # 3. 데이터 추출
        src_props = self.bone_calculator.calculate(source_keypoints, source_scores)
        ref_dirs = self.direction_extractor.extract(reference_keypoints, reference_scores)
        
        global_scale = self._calculate_global_scale(src_props, reference_keypoints, reference_scores)
        lengths = self._correct_bone_lengths(src_props, global_scale)

        # 4. 결과 배열 초기화
        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2))
        trans_scores = np.zeros(num_kpts)
        log = {}
        processed = set()

        # 5. 실행 (Body -> Face -> Hands)
        # [Body]
        self.body_logic.transfer_shoulders(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, processed, log
        )
        self.body_logic.transfer_torso(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, global_scale, processed, log
        )
        self.body_logic.transfer_chain(
            trans_kpts, trans_scores, lengths, reference_keypoints, reference_scores, global_scale, processed, log, is_lower=False
        )
        
        # [Lower Body] (조건부)
        if src_lower_valid and ref_lower_valid:
            self.body_logic.transfer_chain(
                trans_kpts, trans_scores, lengths, reference_keypoints, reference_scores, global_scale, processed, log, is_lower=True
            )

        # [Face]
        if self.config.use_face:
            self.face_logic.transfer(trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, log)

        # [Hands]
        if self.config.use_hands:
            self.hand_logic.transfer(trans_kpts, trans_scores, reference_keypoints, reference_scores, global_scale, log)

        # [삭제됨] self._clip_invisible_legs(...) 호출 제거!
        # 이제 화면 밖으로 나간 다리도 그대로 살아서 나갑니다.

        return TransferResult(trans_kpts, trans_scores, lengths, {}, log)

    def _check_lower_body_valid(self, kpts, scores, img_h):
        # 하반신이 믿을만한지 체크
        indices = [BODY_KEYPOINTS['left_knee'], BODY_KEYPOINTS['right_knee']]
        max_score = max([scores[i] for i in indices])
        if max_score < self.config.lower_body_confidence_threshold: return False
        
        margin = img_h * self.config.lower_body_margin_ratio
        limit = img_h - margin
        l_y, r_y = kpts[BODY_KEYPOINTS['left_knee']][1], kpts[BODY_KEYPOINTS['right_knee']][1]
        
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
    
    def _correct_bone_lengths(self, props, scale):
        lengths = {}
        for n, i in props.bone_lengths.items():
            if i.is_valid: lengths[n] = i.length
        return lengths