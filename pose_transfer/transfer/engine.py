"""
PoseTransferEngine Module (Refactored v2.2 - Tree Structure Adapted)

위치: pose_transfer/transfer/engine.py
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

# [FIX 1] Worker 로직들은 현재 폴더(transfer) 안의 logic에 있음 -> .logic
from .logic.body import BodyTransfer, BoneCalculator, BodyProportions
from .logic.face import FaceTransfer
from .logic.hands import HandTransfer

# [FIX 2] Utils는 상위(pose_transfer)의 utils에 있음 -> ..utils
from ..utils.geometry import calculate_distance

if TYPE_CHECKING:
    # [FIX 3] AlignManager는 상위(pose_transfer)의 logic에 있음 -> ..logic
    from ..logic.align_manager import TransferLayout

@dataclass
class TransferConfig:
    confidence_threshold: float = 0.3
    use_face: bool = True
    use_hands: bool = True
    enable_symmetric_fallback: bool = True
    visibility_margin: float = 0.2
    enable_upper_ratio_tuning: bool = True
    enable_lower_ratio_tuning: bool = True

@dataclass
class TransferResult:
    keypoints: np.ndarray
    scores: np.ndarray
    source_bone_lengths: Dict[str, float] = None
    corrected_bone_lengths: Dict[str, float] = None
    transfer_log: Dict[str, Any] = None


class PoseTransferEngine:
    def __init__(self, config: TransferConfig = None, yaml_config: Optional[dict] = None):
        self.config = config or TransferConfig()
        self.yaml_config = yaml_config or {}
        
        self.bone_calculator = BoneCalculator(self.config.confidence_threshold)
        self.body_logic = BodyTransfer(self.config)
        self.face_logic = FaceTransfer(self.config)
        self.hand_logic = HandTransfer(self.config)

    def _correct_bone_lengths(
        self,
        source_proportions: BodyProportions,
        target_scale: float
    ) -> Dict[str, float]:
        corrected = {}
        for bone_name, info in source_proportions.bone_lengths.items():
            if info.is_valid:
                corrected[bone_name] = info.length * target_scale
            else:
                corrected[bone_name] = 0.0
        return corrected

    def transfer(
        self,
        source_keypoints: np.ndarray, source_scores: np.ndarray,
        reference_keypoints: np.ndarray, reference_scores: np.ndarray,
        source_image_size: Tuple[int, int],
        reference_image_size: Tuple[int, int],
        layout: Optional['TransferLayout'] = None,
        source_depths: Optional[np.ndarray] = None,
        reference_depths: Optional[np.ndarray] = None,
        depth_z_scale: float = 1000.0
    ) -> TransferResult:
        
        print("\n" + "="*70)
        print("⚙️ [Engine] Executing Transfer (Tree Adapted)")
        print("="*70)

        num_kpts = len(source_keypoints)
        trans_kpts = np.zeros((num_kpts, 2), dtype=np.float32)
        trans_scores = np.zeros(num_kpts, dtype=np.float32)
        transfer_log = {}
        processed = set()

        if layout:
            global_scale = layout.global_scale
            print(f"   📋 Layout Applied: Scale={global_scale:.3f}, Anchor={layout.anchor_type}")
        else:
            global_scale = 1.0
            print("   ⚠️ No Layout provided. Using default scale 1.0")

        source_proportions = self.bone_calculator.calculate(source_keypoints, source_scores)
        corrected_lengths = self._correct_bone_lengths(source_proportions, global_scale)

        # [Phase 1] Body
        self.body_logic.transfer_shoulders(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints,
            corrected_lengths=corrected_lengths, processed=processed, log=transfer_log, r_scores=reference_scores
        )
        self.body_logic.transfer_torso(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints,
            corrected_lengths=corrected_lengths, processed=processed, log=transfer_log
        )
        self.body_logic.transfer_limbs_chain(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores,
            corrected_lengths=corrected_lengths, processed=processed, log=transfer_log,
            src_depths=source_depths, ref_depths=reference_depths, depth_z_scale=depth_z_scale
        )

        # [Phase 2] Face
        self.face_logic.transfer_structure(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores,
            processed=processed, log=transfer_log, body_scale=global_scale
        )
        if self.config.use_face:
            self.face_logic.transfer_landmarks(
                trans_kpts, trans_scores, reference_keypoints, reference_scores, processed=processed
            )
        self.face_logic.transfer_ears_fallback(
            trans_kpts, trans_scores, reference_keypoints, reference_scores, processed=processed
        )

        # [Phase 3] Hands
        if self.config.use_hands:
            self.hand_logic.transfer_hands(
                trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores,
                hand_scale_ratio=global_scale, log=transfer_log
            )

        # [Phase 4] Repair
        self._fill_missing_from_reference(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, reference_scores,
            global_scale=global_scale, processed=processed, log=transfer_log
        )

        # [Phase 5] Alignment Offset
        if layout and layout.offset_vector is not None:
            offset = layout.offset_vector
            valid_mask = trans_scores > 0
            trans_kpts[valid_mask] += offset
            print(f"   🚚 Final Alignment Offset: {offset.astype(int)}")

        return TransferResult(
            keypoints=trans_kpts,
            scores=trans_scores,
            source_bone_lengths={k: v.length for k, v in source_proportions.bone_lengths.items()},
            corrected_bone_lengths=corrected_lengths,
            transfer_log=transfer_log
        )

    def _fill_missing_from_reference(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, global_scale, processed, log):
        parent_map = {
            7: 5, 9: 7, 8: 6, 10: 8,       
            13: 11, 15: 13, 14: 12, 16: 14,
            17: 15, 18: 15, 19: 15,        
            20: 16, 21: 16, 22: 16         
        }
        filled_cnt = 0
        for idx in range(len(trans_scores)):
            if trans_scores[idx] < 0.01 and ref_scores[idx] > 0.3:
                parent = parent_map.get(idx)
                if parent is not None and trans_scores[parent] > 0.1:
                    parent_pos = trans_kpts[parent]
                    ref_vec = ref_kpts[idx] - ref_kpts[parent]
                    trans_kpts[idx] = parent_pos + ref_vec * global_scale
                    trans_scores[idx] = ref_scores[idx] * 0.6
                    filled_cnt += 1
        if filled_cnt > 0:
            print(f"   🔧 Repaired {filled_cnt} missing keypoints.")