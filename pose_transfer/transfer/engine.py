"""
Pose Transfer Engine (Refactored for Post-Alignment)

위치: pose_transfer/transfer/engine.py
변경사항:
- [Critical] 본 생성 과정 중에는 Offset 적용 금지
- [Critical] 모든 생성 완료 후, 마지막에 Offset을 일괄 적용 (Post-Alignment)
"""
import numpy as np
import logging
from typing import Optional, Dict
from .config import TransferConfig
from .logic.body import BodyTransfer
from .logic.face import FaceTransfer
from .logic.hands import HandTransfer
from .logic.feets import FeetTransfer

class PoseTransferEngine:
    def __init__(self, config: TransferConfig = None, yaml_config: Optional[dict] = None):
        self.config = config or TransferConfig()
        if yaml_config:
            self.config.update_from_yaml(yaml_config)
            
        self.body_logic = BodyTransfer(self.config)
        self.face_logic = FaceTransfer(self.config)
        self.hand_logic = HandTransfer(self.config)
        self.feet_logic = FeetTransfer(self.config)

    def transfer(self, source_keypoints, source_scores, reference_keypoints, reference_scores,
                  source_size=None, reference_size=None, layout=None, 
                  source_depths=None, reference_depths=None, log_callback=None):
        # source_size, reference_size, depths are optional - reserved for future use
        
        transfer_log = {}

        # 0. 초기화 (Canvas)
        trans_kpts = np.zeros_like(source_keypoints)
        trans_scores = np.zeros_like(source_scores)
        processed = set()
        
        # 1. Body Proportion Analysis
        # Src 신체 비율 계산 (Src 길이 + Ref 벡터 방식)
        src_props = self.body_logic.calculator.calculate(source_keypoints, source_scores, is_source=True)
        corrected_lengths = {k: v.length for k, v in src_props.bone_lengths.items()}
        corrected_lengths['shoulder_width'] = src_props.shoulder_width
        corrected_lengths['hip_width'] = src_props.hip_width
        # Torso 길이도 추가 (BodyTransfer.transfer_torso에서 사용)
        if 'left_torso' in src_props.bone_lengths:
            corrected_lengths['left_torso'] = src_props.bone_lengths['left_torso'].length
        if 'right_torso' in src_props.bone_lengths:
            corrected_lengths['right_torso'] = src_props.bone_lengths['right_torso'].length
        
        # DEBUG: 본 길이 출력
        lt = corrected_lengths.get('left_torso', 0)
        rt = corrected_lengths.get('right_torso', 0)
        lul = corrected_lengths.get('left_upper_leg', 0)
        lll = corrected_lengths.get('left_lower_leg', 0)
        print(f"   [DEBUG Engine] Bone Lengths: torso={lt:.0f}/{rt:.0f}, leg={lul:.0f}+{lll:.0f}")

        # =====================================================================
        # Phase 1: Skeleton Generation (In Reference Coordinate Space)
        # =====================================================================
        # 모든 부위를 Ref 위치 기준으로 생성합니다. Offset 적용 안 함.
        
        # A. Shoulders
        self.body_logic.transfer_shoulders(
            trans_kpts, trans_scores, source_keypoints, source_scores, 
            reference_keypoints, corrected_lengths, processed, transfer_log, reference_scores, layout
        )
        
        # B. Torso
        self.body_logic.transfer_torso(
            trans_kpts, trans_scores, source_keypoints, source_scores, 
            reference_keypoints, corrected_lengths, processed, transfer_log, reference_scores, layout
        )
        
        # C. Limbs (Arms, Legs) - Excludes Feet
        self.body_logic.transfer_limbs_chain(
            trans_kpts, trans_scores, source_keypoints, source_scores, 
            reference_keypoints, reference_scores, corrected_lengths, processed, transfer_log, layout=layout
        )
        
        # D. Feet (독립 모듈)
        self.feet_logic.transfer_feet(
            trans_kpts, trans_scores, source_keypoints, source_scores,
            reference_keypoints, reference_scores, 
            global_scale=layout.global_scale, log=transfer_log
        )
        
        # E. Face & Hands (독립 모듈)
        self.face_logic.transfer_structure(
            trans_kpts, trans_scores, source_keypoints, source_scores,
            reference_keypoints, reference_scores, processed, transfer_log,
            body_scale=layout.global_scale if layout else 1.0
        )
        self.face_logic.transfer_landmarks(
            trans_kpts, trans_scores,
            reference_keypoints, reference_scores, processed
        )
        self.hand_logic.transfer_hands(
            trans_kpts, trans_scores, source_keypoints, source_scores,
            reference_keypoints, reference_scores, 
            hand_scale_ratio=layout.global_scale if layout else 1.0, 
            log=transfer_log
        )

        # =====================================================================
        # Phase 2: Return Results (Offset will be applied in Pipeline after Face Scale)
        # =====================================================================

        # 결과 반환 (namespace object for attribute access)
        class TransferOutput:
            def __init__(self, kpts, scores, log):
                self.keypoints = kpts
                self.scores = scores
                self.transfer_log = log
        return TransferOutput(trans_kpts, trans_scores, transfer_log)