import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, get_keypoint_index
from ...utils.geometry import normalize_vector, calculate_distance

class BodyTransfer:
    def __init__(self):
        self.upper_body_order = [
            ('left_shoulder', 'left_shoulder', ['left_elbow']),
            ('left_elbow', 'left_elbow', ['left_wrist']),
            ('right_shoulder', 'right_shoulder', ['right_elbow']),
            ('right_elbow', 'right_elbow', ['right_wrist']),
        ]
        self.lower_body_order = [
            ('left_hip', 'left_hip', ['left_knee']),
            ('left_knee', 'left_knee', ['left_ankle']),
            ('right_hip', 'right_hip', ['right_knee']),
            ('right_knee', 'right_knee', ['right_ankle']),
        ]

    def transfer_shoulders(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, processed, log):
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        # Source 어깨 중심 및 너비
        src_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        src_width = calculate_distance(s_kpts[l_sh], s_kpts[r_sh])
        
        # Ref 어깨 각도
        ref_vec = r_kpts[r_sh] - r_kpts[l_sh]
        ref_dir = normalize_vector(ref_vec)
        
        # 전이
        t_kpts[l_sh] = src_center - ref_dir * (src_width / 2)
        t_kpts[r_sh] = src_center + ref_dir * (src_width / 2)
        t_scores[l_sh] = t_scores[r_sh] = 0.9
        
        processed.add(l_sh); processed.add(r_sh)
        log['shoulder'] = 'anchor'

    def transfer_torso(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, scale, processed, log):
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
        
        t_neck = (t_kpts[l_sh] + t_kpts[r_sh]) / 2
        r_neck = (r_kpts[l_sh] + r_kpts[r_sh]) / 2
        r_root = (r_kpts[l_hip] + r_kpts[r_hip]) / 2
        
        # 척추 방향
        spine_vec = r_root - r_neck
        spine_dir = normalize_vector(spine_vec)
        
        # 척추 길이 (Source 우선, 없으면 Ref * Scale)
        if s_scores[l_hip] > 0.3:
            s_root = (s_kpts[l_hip] + s_kpts[r_hip]) / 2
            s_neck = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
            spine_len = calculate_distance(s_root, s_neck)
        else:
            spine_len = calculate_distance(r_root, r_neck) * scale
            
        new_root = t_neck + spine_dir * spine_len
        
        # 골반 배치
        r_hip_vec = r_kpts[r_hip] - r_kpts[l_hip]
        r_hip_dir = normalize_vector(r_hip_vec)
        
        if s_scores[l_hip] > 0.3:
            hip_width = calculate_distance(s_kpts[l_hip], s_kpts[r_hip])
        else:
            hip_width = calculate_distance(r_kpts[l_hip], r_kpts[r_hip]) * scale
            
        t_kpts[l_hip] = new_root - r_hip_dir * (hip_width / 2)
        t_kpts[r_hip] = new_root + r_hip_dir * (hip_width / 2)
        t_scores[l_hip] = t_scores[r_hip] = 0.9
        
        processed.add(l_hip); processed.add(r_hip)
        log['torso'] = 'spine_calc'

    def transfer_chain(self, t_kpts, t_scores, lengths, r_kpts, r_scores, scale, processed, log, is_lower=False):
        order = self.lower_body_order if is_lower else self.upper_body_order
        
        for _, p_name, children in order:
            p_idx = get_keypoint_index(p_name)
            if p_idx not in processed: continue
            
            p_pos = t_kpts[p_idx]
            for c_name in children:
                c_idx = get_keypoint_index(c_name)
                
                # Ref 점수가 너무 낮으면 스킵 (Ghost Leg 방지 1차)
                if r_scores[c_idx] < 0.1: continue
                
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                
                length = lengths.get(bone) or lengths.get(alt) or \
                         calculate_distance(r_kpts[p_idx], r_kpts[c_idx]) * scale
                
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direct = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direct * length
                t_scores[c_idx] = 0.8
                
                processed.add(c_idx)
                log[c_name] = 'chain'