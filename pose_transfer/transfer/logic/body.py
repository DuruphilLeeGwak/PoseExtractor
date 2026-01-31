"""
Body Transfer Logic Module (Refactored v8.3 - Symmetry List Fixed)

위치: pose_transfer/transfer/logic/body.py
변경사항:
- [Fix] symmetry_pairs 리스트 정비 (존재하지 않는 키 제거, torso 추가)
- [Improve] 어깨/골반은 단일 너비(Full Width)를 사용하므로 목록에서 제외 (2로 나누어 사용하므로 자동 대칭임)
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
        
        # [Fix] 대칭 보정이 필요한 쌍 (실제 definitions에 있는 키만 포함)
        self.symmetry_pairs = [
            ("left_upper_arm", "right_upper_arm"),
            ("left_lower_arm", "right_lower_arm"),
            ("left_upper_leg", "right_upper_leg"),
            ("left_lower_leg", "right_lower_leg"),
            # [Added] 몸통(Torso) 대칭 추가
            ("left_torso", "right_torso"),
            # [Added] 발 부위
            ("left_ankle_heel", "right_ankle_heel"),
            ("left_ankle_toe", "right_ankle_toe"),
        ]
        # 참고: shoulder_width, hip_width는 단일 값(5->6, 11->12)이므로 대칭 평균 대상이 아님.
        # 사용 시 /2를 하므로 구조적으로 대칭이 됨.

    def calculate(self, kpts: np.ndarray, scores: np.ndarray, is_source: bool = False) -> BodyProportions:
        bones = {}
        # (Start, End, Name)
        definitions = [
            (5, 7, "left_upper_arm"), (7, 9, "left_lower_arm"),
            (6, 8, "right_upper_arm"), (8, 10, "right_lower_arm"),
            (11, 13, "left_upper_leg"), (13, 15, "left_lower_leg"),
            (12, 14, "right_upper_leg"), (14, 16, "right_lower_leg"),
            (5, 6, "shoulder_width"), (11, 12, "hip_width"),
            (5, 11, "left_torso"), (6, 12, "right_torso"),
            # Feet
            (15, 17, "left_ankle_toe"), (15, 19, "left_ankle_heel"),
            (16, 18, "right_ankle_toe"), (16, 20, "right_ankle_heel")
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

        # 소스 이미지일 경우 좌우 대칭 평균화 적용
        if is_source:
            self._apply_symmetry(bones)

        # Torso Length (이제 bones 값이 평균화되었으므로 단순 조회 가능)
        t_len = 0
        if bones["left_torso"].is_valid: t_len = bones["left_torso"].length
        elif bones["right_torso"].is_valid: t_len = bones["right_torso"].length
        
        # Fallback if averaging logic missed something or both invalid
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

    def transfer_shoulders(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, corrected_lengths, processed, log, r_scores):
        neck_pos = np.array([0.0, 0.0]) 
        s_len = corrected_lengths.get('shoulder_width', 0) / 2.0
        if s_len == 0: s_len = 20.0 
        
        if r_scores[5] > 0.1 and r_scores[6] > 0.1:
            ref_vec = ref_kpts[6] - ref_kpts[5]
            angle_rad = np.arctan2(ref_vec[1], ref_vec[0])
        else:
            angle_rad = 0.0 
            
        l_x = neck_pos[0] - s_len * np.cos(angle_rad)
        l_y = neck_pos[1] - s_len * np.sin(angle_rad)
        r_x = neck_pos[0] + s_len * np.cos(angle_rad)
        r_y = neck_pos[1] + s_len * np.sin(angle_rad)
        
        trans_kpts[5] = [l_x, l_y]; trans_scores[5] = r_scores[5]
        trans_kpts[6] = [r_x, r_y]; trans_scores[6] = r_scores[6]
        processed.add(5); processed.add(6)

    def transfer_torso(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, corrected_lengths, processed, log, ref_scores=None):
        neck_pos = np.array([0.0, 0.0])
        
        # Torso Length
        t_len = 0
        l_t = corrected_lengths.get('left_torso', 0)
        r_t = corrected_lengths.get('right_torso', 0)
        if l_t > 0: t_len = l_t # 이미 평균화됨
        elif r_t > 0: t_len = r_t
        if t_len == 0: t_len = 100.0 
        
        # Ref Spine Angle
        ref_neck = (ref_kpts[5] + ref_kpts[6]) / 2
        ref_hip = (ref_kpts[11] + ref_kpts[12]) / 2
        spine_vec = ref_hip - ref_neck
        spine_angle = np.arctan2(spine_vec[1], spine_vec[0])
        
        deg = np.degrees(spine_angle)
        if -150 < deg < -30: 
            print(f"   ⚠️ Detected inverted spine ({deg:.1f}deg). Forcing downward.")
            spine_angle = np.pi / 2 
            
        mid_hip_x = neck_pos[0] + t_len * np.cos(spine_angle)
        mid_hip_y = neck_pos[1] + t_len * np.sin(spine_angle)
        
        # Hip Line Angle from Reference
        if ref_scores is not None and ref_scores[11] > 0.1 and ref_scores[12] > 0.1:
            ref_hip_vec = ref_kpts[12] - ref_kpts[11]
            hip_angle = np.arctan2(ref_hip_vec[1], ref_hip_vec[0])
        else:
            hip_angle = spine_angle - np.pi/2
            
        h_width = corrected_lengths.get('hip_width', 0) / 2.0
        if h_width == 0: h_width = 15.0
        
        trans_kpts[11] = [
            mid_hip_x - h_width * np.cos(hip_angle),
            mid_hip_y - h_width * np.sin(hip_angle)
        ]
        trans_kpts[12] = [
            mid_hip_x + h_width * np.cos(hip_angle),
            mid_hip_y + h_width * np.sin(hip_angle)
        ]
        
        if ref_scores is not None:
            trans_scores[11] = ref_scores[11]
            trans_scores[12] = ref_scores[12]
        
        processed.add(11); processed.add(12)

    def transfer_limbs_chain(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, corrected_lengths, processed, log, src_depths=None, ref_depths=None, depth_z_scale=1000.0):
        chains = [
            (7, 5, 'left_upper_arm'), (9, 7, 'left_lower_arm'),
            (8, 6, 'right_upper_arm'), (10, 8, 'right_lower_arm'),
            (13, 11, 'left_upper_leg'), (15, 13, 'left_lower_leg'),
            (14, 12, 'right_upper_leg'), (16, 14, 'right_lower_leg'),
            #(17, 15, 'left_ankle_toe'), (19, 15, 'left_ankle_heel'),
            (18, 16, 'right_ankle_toe'), (20, 16, 'right_ankle_heel')
        ]
        
        for joint, parent, bone_name in chains:
            if joint >= len(ref_scores) or parent >= len(ref_scores): continue
            length = corrected_lengths.get(bone_name, 0)
            
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