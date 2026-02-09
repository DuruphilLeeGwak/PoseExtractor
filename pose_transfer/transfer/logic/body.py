"""
Body Transfer Logic Module (Fixed v5.0 - Pure Generation / No Alignment)

위치: pose_transfer/transfer/logic/body.py
변경사항:
- [Critical] Layout/Offset 적용 로직 제거 (Engine에서 후처리로 일괄 적용)
- Ref의 좌표계 위에서 뼈대를 생성하는 것에만 집중
"""
import numpy as np
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from ..config import TransferConfig

@dataclass
class BoneInfo:
    length: float
    vector: np.ndarray
    angle: float
    is_valid: bool

@dataclass
class BodyProportions:
    bone_lengths: Dict[str, BoneInfo]
    shoulder_width: float
    hip_width: float
    torso_length: float

class BoneCalculator:
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        # 대칭 보정 (발 부위 제외)
        self.symmetry_pairs = [
            ("left_upper_arm", "right_upper_arm"),
            ("left_lower_arm", "right_lower_arm"),
            ("left_upper_leg", "right_upper_leg"),
            ("left_lower_leg", "right_lower_leg"),
            ("left_torso", "right_torso"),
        ]

    def calculate(self, kpts: np.ndarray, scores: np.ndarray, is_source: bool = False) -> BodyProportions:
        bones = {}
        # 발가락 제외한 정의
        definitions = [
            (5, 7, "left_upper_arm"), (7, 9, "left_lower_arm"),
            (6, 8, "right_upper_arm"), (8, 10, "right_lower_arm"),
            (11, 13, "left_upper_leg"), (13, 15, "left_lower_leg"),
            (12, 14, "right_upper_leg"), (14, 16, "right_lower_leg"),
            (5, 6, "shoulder_width"), (11, 12, "hip_width"),
            (5, 11, "left_torso"), (6, 12, "right_torso"),
        ]

        for start, end, name in definitions:
            if start < len(scores) and end < len(scores):
                if scores[start] > self.confidence_threshold and scores[end] > self.confidence_threshold:
                    vec = kpts[end] - kpts[start]
                    length = np.linalg.norm(vec)
                    angle = np.degrees(np.arctan2(vec[1], vec[0]))
                    bones[name] = BoneInfo(length, vec, angle, True)
                    continue
            bones[name] = BoneInfo(0.0, np.zeros(2), 0.0, False)

        if is_source:
            self._apply_symmetry(bones)

        t_len = 0
        if bones["left_torso"].is_valid: t_len = bones["left_torso"].length
        elif bones["right_torso"].is_valid: t_len = bones["right_torso"].length
        if bones["left_torso"].is_valid and bones["right_torso"].is_valid:
             t_len = (bones["left_torso"].length + bones["right_torso"].length) / 2

        s_width = bones["shoulder_width"].length / 2 if bones["shoulder_width"].is_valid else 0
        h_width = bones["hip_width"].length / 2 if bones["hip_width"].is_valid else 0

        return BodyProportions(bones, s_width, h_width, t_len)

    def _apply_symmetry(self, bones: Dict[str, BoneInfo]):
        for l_name, r_name in self.symmetry_pairs:
            l_bone = bones.get(l_name)
            r_bone = bones.get(r_name)
            if l_bone and r_bone and l_bone.is_valid and r_bone.is_valid:
                avg = (l_bone.length + r_bone.length) / 2.0
                l_bone.length = avg
                r_bone.length = avg

class BodyTransfer:
    def __init__(self, config: TransferConfig):
        self.config = config
        self.calculator = BoneCalculator(config.confidence_threshold)

    def transfer_shoulders(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, corrected_lengths, processed, log, r_scores, layout=None):
        """ 어깨 전이: Ref 위치 기준 + Src 길이 + Ref 각도 """
        
        # 1. Ref Neck Center (기준 위치) - Ref 좌표계 사용
        if r_scores[5] > 0.1 and r_scores[6] > 0.1:
            ref_neck = (ref_kpts[5] + ref_kpts[6]) / 2.0
        else:
            ref_neck = np.array([0.0, 0.0])
        
        trans_neck = ref_neck  # Ref 위치 사용
        
        # 2. Ref 각도 참조 (Ref 포즈 방향)
        if r_scores[5] > 0.1 and r_scores[6] > 0.1:
            ref_vec = ref_kpts[6] - ref_kpts[5]
            angle_rad = np.arctan2(ref_vec[1], ref_vec[0])
        else:
            angle_rad = 0.0

        # 3. Src Shoulder Width (Src 비율 그대로, Scale 미적용)
        s_width = corrected_lengths.get('shoulder_width', 40.0)
        
        # 4. 배치 (Ref 위치 + Src 길이 + Ref 각도)
        trans_kpts[5] = trans_neck - np.array([s_width * np.cos(angle_rad), s_width * np.sin(angle_rad)])
        trans_kpts[6] = trans_neck + np.array([s_width * np.cos(angle_rad), s_width * np.sin(angle_rad)])
        
        trans_scores[5] = max(src_scores[5], r_scores[5])
        trans_scores[6] = max(src_scores[6], r_scores[6])
        processed.add(5)
        processed.add(6)

    def transfer_torso(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, corrected_lengths, processed, log, ref_scores, layout=None):
        """ 몸통 전이: 어깨 기준 + Src torso 길이 + Ref 각도 """
        scale = layout.global_scale if layout else 1.0

        # 1. Trans Neck Center (이미 transfer_shoulders에서 생성됨)
        if trans_scores[5] > 0.1 and trans_scores[6] > 0.1:
            trans_neck = (trans_kpts[5] + trans_kpts[6]) / 2.0
        else:
            return  # 어깨가 없으면 골반 생성 불가
        
        # 2. Ref Torso 각도 계산 (어깨중심 -> 골반중심 방향)
        if ref_scores[5] > 0.1 and ref_scores[6] > 0.1 and ref_scores[11] > 0.1 and ref_scores[12] > 0.1:
            ref_neck = (ref_kpts[5] + ref_kpts[6]) / 2.0
            ref_hip_center = (ref_kpts[11] + ref_kpts[12]) / 2.0
            torso_vec = ref_hip_center - ref_neck
            torso_angle = np.arctan2(torso_vec[1], torso_vec[0])
        else:
            torso_angle = np.pi / 2  # 수직 하향
        
        # 3. Src Torso 길이 사용 (corrected_lengths에서 가져옴)
        left_torso = corrected_lengths.get('left_torso', 0)
        right_torso = corrected_lengths.get('right_torso', 0)
        if left_torso > 0 and right_torso > 0:
            torso_length = (left_torso + right_torso) / 2.0
        elif left_torso > 0:
            torso_length = left_torso
        elif right_torso > 0:
            torso_length = right_torso
        else:
            torso_length = 300.0  # fallback
        
        torso_length *= scale
        
        # 4. Trans Hip Center = Trans Neck + (Src Length * Ref Direction)
        trans_hip = trans_neck + np.array([
            torso_length * np.cos(torso_angle),
            torso_length * np.sin(torso_angle)
        ])

        # 5. Hip Angle (좌우 골반 방향)
        if ref_scores[11] > 0.1 and ref_scores[12] > 0.1:
            ref_vec = ref_kpts[12] - ref_kpts[11]
            hip_angle = np.arctan2(ref_vec[1], ref_vec[0])
        else:
            hip_angle = 0.0
            
        # 6. Hip Width (이미 절반 값이므로 /2 제거)
        h_width = corrected_lengths.get('hip_width', 30.0) * scale
        
        # 7. 배치
        trans_kpts[11] = trans_hip - np.array([h_width * np.cos(hip_angle), h_width * np.sin(hip_angle)])
        trans_kpts[12] = trans_hip + np.array([h_width * np.cos(hip_angle), h_width * np.sin(hip_angle)])
        
        trans_scores[11] = ref_scores[11]
        trans_scores[12] = ref_scores[12]
        processed.add(11)
        processed.add(12)

    def transfer_limbs_chain(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, corrected_lengths, processed, log, src_depths=None, ref_depths=None, depth_z_scale=1000.0, layout=None):
        """ 팔다리 전이 (Offset 적용 X) """
        scale = layout.global_scale if layout else 1.0
        
        chains = [
            (7, 5, 'left_upper_arm'), (9, 7, 'left_lower_arm'),
            (8, 6, 'right_upper_arm'), (10, 8, 'right_lower_arm'),
            (13, 11, 'left_upper_leg'), (15, 13, 'left_lower_leg'),
            (14, 12, 'right_upper_leg'), (16, 14, 'right_lower_leg'),
        ]
        
        for joint, parent, bone_name in chains:
            if joint >= len(ref_scores) or parent >= len(ref_scores): continue
            
            length = corrected_lengths.get(bone_name, 0) * scale
            
            if ref_scores[joint] > 0.1 and ref_scores[parent] > 0.1:
                ref_vec = ref_kpts[joint] - ref_kpts[parent]
                angle_rad = np.arctan2(ref_vec[1], ref_vec[0])
            else:
                angle_rad = np.pi / 2 
            
            if trans_kpts[parent][0] == 0 and trans_kpts[parent][1] == 0:
                continue 
                
            parent_pos = trans_kpts[parent]
            new_x = parent_pos[0] + length * np.cos(angle_rad)
            new_y = parent_pos[1] + length * np.sin(angle_rad)
            
            trans_kpts[joint] = [new_x, new_y]
            trans_scores[joint] = ref_scores[joint]
            processed.add(joint)