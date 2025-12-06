import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, FEET_KEYPOINTS, get_keypoint_index
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
        # [NEW] Feet 전이 순서 (ankle -> heel, big_toe, small_toe)
        self.feet_order = [
            ('left_ankle', 'left_ankle', ['left_heel', 'left_big_toe', 'left_small_toe']),
            ('right_ankle', 'right_ankle', ['right_heel', 'right_big_toe', 'right_small_toe']),
        ]

    def transfer_shoulders(self, t_kpts, t_scores, s_kpts, s_scores, r_kpts, processed, log):
        l_sh = BODY_KEYPOINTS['left_shoulder']
        r_sh = BODY_KEYPOINTS['right_shoulder']
        
        src_center = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
        src_width = calculate_distance(s_kpts[l_sh], s_kpts[r_sh])
        
        ref_vec = r_kpts[r_sh] - r_kpts[l_sh]
        ref_dir = normalize_vector(ref_vec)
        
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
        
        spine_vec = r_root - r_neck
        spine_dir = normalize_vector(spine_vec)
        
        if s_scores[l_hip] > 0.3:
            s_root = (s_kpts[l_hip] + s_kpts[r_hip]) / 2
            s_neck = (s_kpts[l_sh] + s_kpts[r_sh]) / 2
            spine_len = calculate_distance(s_root, s_neck)
        else:
            spine_len = calculate_distance(r_root, r_neck) * scale
            
        new_root = t_neck + spine_dir * spine_len
        
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
        chain_type = "LOWER" if is_lower else "UPPER"
        
        print(f"\n   🔍 [DEBUG] transfer_chain({chain_type}) START")
        print(f"      processed indices: {sorted(processed)}")
        
        for _, p_name, children in order:
            p_idx = get_keypoint_index(p_name)
            
            if p_idx not in processed:
                continue
            
            p_pos = t_kpts[p_idx]
            
            for c_name in children:
                c_idx = get_keypoint_index(c_name)
                if c_idx is None:
                    continue
                    
                r_score = r_scores[c_idx] if c_idx < len(r_scores) else 0
                
                # Ghost Leg 방지
                if r_score < 0.1:
                    print(f"      ❌ {c_name} (idx={c_idx}): ref_score={r_score:.3f} < 0.1, SKIP")
                    continue
                
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
                
                print(f"      ✅ {c_name} (idx={c_idx}): transferred, pos={t_kpts[c_idx]}")
        
        print(f"   🔍 [DEBUG] transfer_chain({chain_type}) END")
        print(f"      final processed: {sorted(processed)}")

    def transfer_feet(self, t_kpts, t_scores, lengths, r_kpts, r_scores, scale, processed, log):
        """
        [NEW] Feet 키포인트 전이 (ankle -> heel, big_toe, small_toe)
        """
        print(f"\n   🦶 [DEBUG] transfer_feet() START")
        print(f"      processed indices: {sorted(processed)}")
        
        # FEET_KEYPOINTS가 없으면 스킵
        if not FEET_KEYPOINTS:
            print(f"      ⚠️ FEET_KEYPOINTS not defined, skipping")
            return
        
        for _, p_name, children in self.feet_order:
            p_idx = get_keypoint_index(p_name)
            
            if p_idx is None or p_idx not in processed:
                print(f"      ❌ {p_name} (idx={p_idx}): not in processed, SKIP")
                continue
            
            p_pos = t_kpts[p_idx]
            print(f"\n      [Parent] {p_name} (idx={p_idx}), pos={p_pos}")
            
            for c_name in children:
                # FEET_KEYPOINTS에서 인덱스 가져오기
                c_idx = FEET_KEYPOINTS.get(c_name)
                if c_idx is None:
                    print(f"         ❌ {c_name}: not in FEET_KEYPOINTS")
                    continue
                
                r_score = r_scores[c_idx] if c_idx < len(r_scores) else 0
                
                print(f"         [Child] {c_name} (idx={c_idx}): ref_score={r_score:.3f}")
                
                # 점수가 너무 낮으면 스킵
                if r_score < 0.1:
                    print(f"         ❌ SKIP: ref_score < 0.1")
                    continue
                
                # 뼈 길이 계산
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                
                if bone in lengths:
                    length = lengths[bone]
                elif alt in lengths:
                    length = lengths[alt]
                else:
                    # lengths에 없으면 ref에서 직접 계산
                    length = calculate_distance(r_kpts[p_idx], r_kpts[c_idx]) * scale
                
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direct = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direct * length
                t_scores[c_idx] = 0.8
                
                processed.add(c_idx)
                log[c_name] = 'feet_chain'
                
                print(f"         ✅ TRANSFERRED: pos={t_kpts[c_idx]}, length={length:.1f}")
        
        print(f"\n   🦶 [DEBUG] transfer_feet() END")
        print(f"      final processed: {sorted(processed)}")