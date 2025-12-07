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
                
                if r_score < 0.1:
                    print(f"      ❌ {c_name} (idx={c_idx}): ref_score={r_score:.3f} < 0.1, SKIP")
                    continue
                
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                
                # 뼈 길이 결정
                src_length = lengths.get(bone) or lengths.get(alt)
                ref_length = calculate_distance(r_kpts[p_idx], r_kpts[c_idx])
                
                if src_length:
                    length = src_length
                    source = "SRC"
                else:
                    length = ref_length * scale
                    source = f"REF*scale"
                
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direct = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direct * length
                t_scores[c_idx] = 0.8
                
                processed.add(c_idx)
                log[c_name] = 'chain'
                
                print(f"      ✅ {c_name} (idx={c_idx}): length={length:.1f} ({source})")
        
        print(f"   🔍 [DEBUG] transfer_chain({chain_type}) END")
        print(f"      final processed: {sorted(processed)}")

    def transfer_feet(self, t_kpts, t_scores, lengths, r_kpts, r_scores, scale, processed, log):
        """
        Feet 키포인트 전이 (DEBUG VERSION)
        """
        print(f"\n" + "="*60)
        print(f"🦶 [DEBUG] transfer_feet()")
        print("="*60)
        print(f"   global_scale (어깨비율): {scale:.3f}")
        
        # 발 관련 뼈 길이 확인
        feet_bones = [k for k in lengths.keys() if any(x in k for x in ['ankle', 'toe', 'heel'])]
        print(f"   src에서 계산된 발 뼈 길이: {feet_bones if feet_bones else 'NONE!'}")
        
        if not FEET_KEYPOINTS:
            print(f"   ⚠️ FEET_KEYPOINTS not defined, skipping")
            return
        
        for _, p_name, children in self.feet_order:
            p_idx = get_keypoint_index(p_name)
            
            if p_idx is None or p_idx not in processed:
                print(f"\n   ❌ {p_name} not in processed, SKIP")
                continue
            
            p_pos = t_kpts[p_idx]
            side = "LEFT" if "left" in p_name else "RIGHT"
            print(f"\n   [{side}] Foot from {p_name} ({p_idx})")
            print(f"      parent_pos: ({p_pos[0]:.1f}, {p_pos[1]:.1f})")
            
            for c_name in children:
                c_idx = FEET_KEYPOINTS.get(c_name)
                if c_idx is None:
                    print(f"      ❌ {c_name}: not in FEET_KEYPOINTS")
                    continue
                
                r_score = r_scores[c_idx] if c_idx < len(r_scores) else 0
                
                if r_score < 0.1:
                    print(f"      ❌ {c_name} (idx={c_idx}): ref_score={r_score:.3f} < 0.1, SKIP")
                    continue
                
                # 뼈 길이 결정
                bone = f"{p_name}_{c_name}"
                alt = f"{c_name}_{p_name}"
                
                src_length = lengths.get(bone) or lengths.get(alt)
                ref_length = calculate_distance(r_kpts[p_idx], r_kpts[c_idx])
                
                if src_length:
                    length = src_length
                    source = "SRC"
                else:
                    length = ref_length * scale
                    source = f"REF*scale ({ref_length:.1f}*{scale:.3f})"
                
                vec = r_kpts[c_idx] - r_kpts[p_idx]
                direct = normalize_vector(vec)
                
                t_kpts[c_idx] = p_pos + direct * length
                t_scores[c_idx] = 0.8
                
                processed.add(c_idx)
                log[c_name] = 'feet_chain'
                
                print(f"      ✅ {c_name} (idx={c_idx}): length={length:.1f} ({source})")
        
        print(f"\n   final feet processed: {sorted([i for i in processed if i >= 17 and i <= 22])}")