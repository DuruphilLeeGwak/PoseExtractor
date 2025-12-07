"""
Pose Transfer Engine Module (v12 - Case-aware Lower Body Control)
Case naming: F_F, F_H, H_F, H_H (Full/Half × src/ref)
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
        
        if yaml_config:
            if 'face_rendering' in yaml_config:
                self.config.face_rendering = FaceRenderingConfig.from_dict(yaml_config['face_rendering'])
        
        self.bone_calculator = BoneCalculator(confidence_threshold=self.config.confidence_threshold)
        self.direction_extractor = DirectionExtractor(confidence_threshold=self.config.confidence_threshold)
        
        self.body_logic = BodyTransfer()
        self.face_logic = FaceTransfer(self.config)
        self.hand_logic = HandTransfer()

    def transfer(
        self,
        source_keypoints: np.ndarray, source_scores: np.ndarray,
        reference_keypoints: np.ndarray, reference_scores: np.ndarray,
        source_image_size: Optional[Tuple[int, int]] = None,
        reference_image_size: Optional[Tuple[int, int]] = None,
        target_image_size: Optional[Tuple[int, int]] = None,
        alignment_case: Optional[str] = None,
    ) -> TransferResult:
        """
        포즈 전이 메인 로직
        
        Args:
            alignment_case: 'F_F', 'F_H', 'H_F', 'H_H' 중 하나
                - F_F: Full → Full (전신→전신, 하반신 전이 O)
                - F_H: Full → Half (전신→상반신, 하반신 전이 X)
                - H_F: Half → Full (상반신→전신, SRC 하반신 없음)
                - H_H: Half → Half (상반신→상반신, 하반신 전이 X)
        """
        
        print("\n" + "="*70)
        print("🔍 [DEBUG] PoseTransferEngine.transfer() START")
        print("="*70)
        
        if alignment_case:
            print(f"\n📋 Alignment Case: {alignment_case}")
        
        # 1. 이미지 크기 추정
        if source_image_size is None:
            max_y = np.max(source_keypoints[:, 1])
            source_image_size = (int(max_y * 1.1), int(np.max(source_keypoints[:, 0])))
        src_h, src_w = source_image_size
        
        print(f"\n📐 Image Sizes:")
        print(f"   Source: {src_w}x{src_h}")
        print(f"   Reference: {reference_image_size}")

        # 2. 하반신 키포인트 점수 출력
        print(f"\n📊 Lower Body Keypoint Scores:")
        lower_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        for name in lower_names:
            idx = BODY_KEYPOINTS.get(name, -1)
            if idx >= 0:
                src_score = source_scores[idx] if idx < len(source_scores) else 0
                ref_score = reference_scores[idx] if idx < len(reference_scores) else 0
                src_pos = source_keypoints[idx] if idx < len(source_keypoints) else [0, 0]
                ref_pos = reference_keypoints[idx] if idx < len(reference_keypoints) else [0, 0]
                print(f"   {name:15} (idx={idx:2d}): src_score={src_score:.3f} ref_score={ref_score:.3f}")
                print(f"                          src_pos={src_pos}, ref_pos={ref_pos}")

        # =====================================================================
        # 3. 하반신 전이 여부 결정 (Case 기반 - 핵심!)
        # =====================================================================
        # REF가 상반신(Half)인 경우 → 하반신 전이 스킵
        # F_H: Full → Half (REF가 상반신)
        # H_H: Half → Half (REF가 상반신)
        if alignment_case in ['F_H', 'H_H']:
            ref_lower_valid = False
            print(f"\n⏭️ [Skip] Lower Body (REF is HALF, Case {alignment_case})")
        else:
            # F_F, H_F는 REF가 전신이므로 기존 로직으로 검증
            ref_lower_valid = True
            if reference_image_size:
                ref_lower_valid = self._check_lower_body_valid(reference_keypoints, reference_scores, reference_image_size[0])
            
            ref_knee_score = min(
                reference_scores[BODY_KEYPOINTS['left_knee']], 
                reference_scores[BODY_KEYPOINTS['right_knee']]
            )
            print(f"   ref_knee_score (min): {ref_knee_score:.3f}")
            
            if ref_knee_score < 0.1:
                ref_lower_valid = False
                print(f"   ❌ ref_lower_valid = False (knee score < 0.1)")
        
        print(f"\n🦵 ref_lower_valid = {ref_lower_valid}")

        # 4. 데이터 추출
        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        reference_directions = self.direction_extractor.extract(reference_keypoints, reference_scores)
        
        global_scale = self._calculate_global_scale(source_proportions, reference_keypoints, reference_scores)
        corrected_lengths = self._correct_bone_lengths(source_proportions, global_scale, reference_keypoints)
        
        print(f"\n📏 Global Scale: {global_scale:.3f}")
        print(f"   Corrected Bone Lengths: {list(corrected_lengths.keys())}")

        # 5. 결과 배열 초기화
        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2))
        trans_scores = np.zeros(num_kpts)
        transfer_log = {}
        processed = set()

        # 6. 실행 (Body -> Face -> Hands)
        print("\n" + "-"*50)
        print("🏃 Body Transfer START")
        print("-"*50)
        
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
        
        # [Body: Lower] - Case 기반 제어
        if ref_lower_valid:
            print("\n   🦵 [Transfer] Generating Lower Body (REF is FULL)")
            self.body_logic.transfer_chain(
                trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, global_scale, processed, transfer_log, is_lower=True
            )
            
            # Feet 전이
            print("\n   🦶 [Transfer] Generating Feet")
            self.body_logic.transfer_feet(
                trans_kpts, trans_scores, corrected_lengths, reference_keypoints, reference_scores, global_scale, processed, transfer_log
            )
        else:
            print(f"\n   🚫 [Skip] Lower Body & Feet (Case {alignment_case} - REF is HALF)")
            # 하반신 키포인트 점수 0으로 설정 (렌더링 제외)
            lower_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            for idx in lower_indices:
                if idx < len(trans_scores):
                    trans_scores[idx] = 0.0
            print(f"      → Lower body scores set to 0")

        # 전이 후 하반신 점수 확인
        print("\n" + "-"*50)
        print("📊 After Body Transfer - Lower Body Scores:")
        print("-"*50)
        for name in lower_names:
            idx = BODY_KEYPOINTS.get(name, -1)
            if idx >= 0:
                score = trans_scores[idx]
                pos = trans_kpts[idx]
                status = "✅" if score > 0 else "❌"
                print(f"   {status} {name:15} (idx={idx:2d}): score={score:.3f}, pos={pos}")

        # [Face] - r_scores 전달
        if self.config.use_face:
            self.face_logic.transfer(
                trans_kpts, trans_scores, 
                source_keypoints, source_scores, 
                reference_keypoints, reference_scores,
                transfer_log
            )

        # [Hands]
        if self.config.use_hands:
            self.hand_logic.transfer(trans_kpts, trans_scores, reference_keypoints, reference_scores, global_scale, transfer_log)

        print("\n" + "="*70)
        print("🔍 [DEBUG] PoseTransferEngine.transfer() END")
        print("="*70 + "\n")

        return TransferResult(trans_kpts, trans_scores, corrected_lengths, {}, transfer_log)

    def _check_lower_body_valid(self, kpts, scores, img_h):
        print(f"\n   🔍 [DEBUG] _check_lower_body_valid(img_h={img_h})")
        
        indices = [BODY_KEYPOINTS['left_knee'], BODY_KEYPOINTS['right_knee']]
        max_score = max([scores[i] for i in indices])
        
        print(f"      knee indices: {indices}")
        print(f"      knee scores: {[scores[i] for i in indices]}")
        print(f"      max_score: {max_score:.3f}")
        print(f"      threshold: {self.config.lower_body_confidence_threshold}")
        
        if max_score < self.config.lower_body_confidence_threshold:
            print(f"      ❌ INVALID: max_score < threshold")
            return False
        
        margin = img_h * self.config.lower_body_margin_ratio
        limit = img_h - margin
        l_y = kpts[BODY_KEYPOINTS['left_knee']][1]
        r_y = kpts[BODY_KEYPOINTS['right_knee']][1]
        
        print(f"      margin_ratio: {self.config.lower_body_margin_ratio}")
        print(f"      limit (img_h - margin): {limit:.1f}")
        print(f"      left_knee_y: {l_y:.1f}, right_knee_y: {r_y:.1f}")
        
        if (l_y > limit and scores[BODY_KEYPOINTS['left_knee']] > 0.1) or \
           (r_y > limit and scores[BODY_KEYPOINTS['right_knee']] > 0.1):
            print(f"      ❌ INVALID: knee too close to bottom (Ghost Leg)")
            return False
        
        print(f"      ✅ VALID")
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