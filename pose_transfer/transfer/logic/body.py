"""
Body Logic (Refactored v2.2 - Tree Structure Adapted)

위치: pose_transfer/transfer/logic/body.py
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Set, Optional, Any

# [FIX] Import Depth 수정 (logic -> transfer -> pose_transfer -> extractors)
from ...extractors.keypoint_constants import BODY_KEYPOINTS
from ...utils.geometry import calculate_distance, normalize_vector

@dataclass
class BoneInfo:
    length: float
    is_valid: bool

@dataclass
class BodyProportions:
    bone_lengths: Dict[str, BoneInfo] = field(default_factory=dict)

class BoneCalculator:
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold

    def calculate(self, keypoints, scores) -> BodyProportions:
        bones = {}
        definitions = [
            (5, 6, 'shoulder_width'), (11, 12, 'hip_width'),
            (5, 7, 'left_upper_arm'), (7, 9, 'left_lower_arm'),
            (6, 8, 'right_upper_arm'), (8, 10, 'right_lower_arm'),
            (11, 13, 'left_upper_leg'), (13, 15, 'left_lower_leg'),
            (12, 14, 'right_upper_leg'), (14, 16, 'right_lower_leg')
        ]
        if (scores[5]>0.1 and scores[6]>0.1 and scores[11]>0.1 and scores[12]>0.1):
            neck = (keypoints[5] + keypoints[6]) / 2
            root = (keypoints[11] + keypoints[12]) / 2
            bones['torso_length'] = BoneInfo(np.linalg.norm(root - neck), True)
        else:
            bones['torso_length'] = BoneInfo(0.0, False)

        for i1, i2, name in definitions:
            if scores[i1] > 0.1 and scores[i2] > 0.1:
                dist = np.linalg.norm(keypoints[i1] - keypoints[i2])
                bones[name] = BoneInfo(dist, True)
            else:
                bones[name] = BoneInfo(0.0, False)
        return BodyProportions(bone_lengths=bones)

class BodyTransfer:
    def __init__(self, config=None):
        self.config = config

    def transfer_shoulders(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, corrected_lengths, processed, log, r_scores=None):
        LS, RS = 5, 6
        shoulder_width = corrected_lengths.get('shoulder_width', 0.0)
        if shoulder_width <= 0: shoulder_width = 1.0
        
        ref_angle = 0.0
        if r_scores is not None and r_scores[LS] > 0.1 and r_scores[RS] > 0.1:
            ref_vec = ref_kpts[RS] - ref_kpts[LS]
            ref_angle = np.arctan2(ref_vec[1], ref_vec[0])
            
        center = np.array([0.0, 0.0]) 
        dx = np.cos(ref_angle) * (shoulder_width / 2.0)
        dy = np.sin(ref_angle) * (shoulder_width / 2.0)
        
        trans_kpts[LS] = center - np.array([dx, dy])
        trans_scores[LS] = 1.0; processed.add(LS)
        trans_kpts[RS] = center + np.array([dx, dy])
        trans_scores[RS] = 1.0; processed.add(RS)

    def transfer_torso(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, corrected_lengths, processed, log):
        LS, RS, LH, RH = 5, 6, 11, 12
        torso_len = corrected_lengths.get('torso_length', 0.0)
        hip_width = corrected_lengths.get('hip_width', 0.0)
        neck = (trans_kpts[LS] + trans_kpts[RS]) / 2.0
        
        ref_dir = np.array([0.0, 1.0])
        r_ls, r_rs, r_lh, r_rh = 5, 6, 11, 12
        if np.any(ref_kpts[r_ls]!=0) and np.any(ref_kpts[r_lh]!=0):
            ref_neck = (ref_kpts[r_ls]+ref_kpts[r_rs])/2
            ref_root = (ref_kpts[r_lh]+ref_kpts[r_rh])/2
            vec = ref_root - ref_neck
            if np.linalg.norm(vec) > 1e-6: ref_dir = vec / np.linalg.norm(vec)
                
        root = neck + ref_dir * torso_len
        hip_angle = 0.0 
        if np.any(ref_kpts[r_rh]!=0) and np.any(ref_kpts[r_lh]!=0):
            v = ref_kpts[r_rh]-ref_kpts[r_lh]
            hip_angle = np.arctan2(v[1], v[0])
            
        h_dx = np.cos(hip_angle)*(hip_width/2)
        h_dy = np.sin(hip_angle)*(hip_width/2)
        trans_kpts[LH] = root - np.array([h_dx, h_dy])
        trans_scores[LH] = 1.0; processed.add(LH)
        trans_kpts[RH] = root + np.array([h_dx, h_dy])
        trans_scores[RH] = 1.0; processed.add(RH)

    def transfer_limbs_chain(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, corrected_lengths, processed, log, src_depths=None, ref_depths=None, depth_z_scale=1000.0):
        chains = [
            (5, 7, 'left_upper_arm'), (7, 9, 'left_lower_arm'),
            (6, 8, 'right_upper_arm'), (8, 10, 'right_lower_arm'),
            (11, 13, 'left_upper_leg'), (13, 15, 'left_lower_leg'),
            (12, 14, 'right_upper_leg'), (14, 16, 'right_lower_leg')
        ]
        for p, c, name in chains:
            if trans_scores[p] < 0.1: continue
            length = corrected_lengths.get(name, 0.0)
            if length <= 0 or ref_scores[p] < 0.1 or ref_scores[c] < 0.1: continue
            
            direction = normalize_vector(ref_kpts[c] - ref_kpts[p])
            trans_kpts[c] = trans_kpts[p] + direction * length
            trans_scores[c] = 1.0; processed.add(c)