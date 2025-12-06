from ...extractors.keypoint_constants import BODY_KEYPOINTS, LEFT_HAND_START_IDX, RIGHT_HAND_START_IDX

class HandTransfer:
    def transfer(self, t_kpts, t_scores, r_kpts, r_scores, scale, log):
        for is_left in [True, False]:
            w_name = 'left_wrist' if is_left else 'right_wrist'
            w_idx = BODY_KEYPOINTS[w_name]
            
            # 손목이 전이되지 않았으면 손도 스킵
            if t_scores[w_idx] == 0: continue
            
            start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
            ref_w = r_kpts[w_idx]
            wrist_pos = t_kpts[w_idx]
            
            for i in range(21):
                idx = start + i
                if r_scores[idx] > 0.2:
                    rel = r_kpts[idx] - ref_w
                    t_kpts[idx] = wrist_pos + rel * scale
                    t_scores[idx] = 0.9