import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, LEFT_HAND_START_IDX, RIGHT_HAND_START_IDX
from ...utils.geometry import calculate_distance

class HandTransfer:
    def transfer(self, t_kpts, t_scores, r_kpts, r_scores, scale, log):
        print("\n" + "="*60)
        print("🖐️ [DEBUG] HandTransfer.transfer()")
        print("="*60)
        print(f"   global_scale (어깨비율): {scale:.3f}")
        
        for is_left in [True, False]:
            side = "LEFT" if is_left else "RIGHT"
            w_name = 'left_wrist' if is_left else 'right_wrist'
            w_idx = BODY_KEYPOINTS[w_name]
            
            # 손목이 전이되지 않았으면 손도 스킵
            if t_scores[w_idx] == 0:
                print(f"\n   [{side}] wrist score=0, SKIP")
                continue
            
            start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
            ref_w = r_kpts[w_idx]
            wrist_pos = t_kpts[w_idx]
            
            print(f"\n   [{side}] Hand Transfer")
            print(f"      ref_wrist: ({ref_w[0]:.1f}, {ref_w[1]:.1f})")
            print(f"      trans_wrist: ({wrist_pos[0]:.1f}, {wrist_pos[1]:.1f})")
            
            # ref 손 크기 계산 (손목-중지 끝)
            middle_tip_idx = start + 12  # middle finger tip
            ref_hand_len = 0
            if r_scores[middle_tip_idx] > 0.2:
                ref_hand_len = calculate_distance(r_kpts[w_idx], r_kpts[middle_tip_idx])
                print(f"      ref_hand_length (wrist→middle_tip): {ref_hand_len:.1f}px")
            
            transferred_count = 0
            for i in range(21):
                idx = start + i
                if r_scores[idx] > 0.2:
                    rel = r_kpts[idx] - ref_w
                    t_kpts[idx] = wrist_pos + rel * scale
                    t_scores[idx] = 0.9
                    transferred_count += 1
            
            print(f"      transferred: {transferred_count}/21 keypoints")
            
            # 전이 후 손 크기 확인
            if t_scores[middle_tip_idx] > 0 and ref_hand_len > 0:
                trans_hand_len = calculate_distance(t_kpts[w_idx], t_kpts[middle_tip_idx])
                print(f"      trans_hand_length: {trans_hand_len:.1f}px")
                print(f"      actual_ratio (trans/ref): {trans_hand_len/ref_hand_len:.3f}")
                print(f"      expected_ratio (global_scale): {scale:.3f}")
                
                if abs(trans_hand_len/ref_hand_len - scale) > 0.01:
                    print(f"      ✅ Scale applied correctly")
                else:
                    print(f"      ⚠️ Check scale application")