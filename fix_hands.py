"""hands.py를 올바르게 작성"""

content = '''import numpy as np
from ...extractors.keypoint_constants import BODY_KEYPOINTS, LEFT_HAND_START_IDX, RIGHT_HAND_START_IDX
from ...utils.geometry import calculate_distance

class HandTransfer:
    def transfer(self, t_kpts, t_scores, s_kpts, s_scores, scale, log):
        """손 전이: Source 손 형태를 유지하되 전이된 손목 위치에 배치"""
        print("\\n" + "="*60)
        print("🖐️ HandTransfer - Using SOURCE hands")
        print("="*60)
        
        for is_left in [True, False]:
            side = "LEFT" if is_left else "RIGHT"
            w_name = 'left_wrist' if is_left else 'right_wrist'
            w_idx = BODY_KEYPOINTS[w_name]
            
            if t_scores[w_idx] == 0:
                print(f"\\n   [{side}] trans wrist score=0, SKIP")
                continue
            
            start = LEFT_HAND_START_IDX if is_left else RIGHT_HAND_START_IDX
            src_wrist = s_kpts[w_idx]
            trans_wrist = t_kpts[w_idx]
            
            print(f"\\n   [{side}] Hand Transfer")
            print(f"      src_wrist: ({src_wrist[0]:.1f}, {src_wrist[1]:.1f})")
            print(f"      trans_wrist: ({trans_wrist[0]:.1f}, {trans_wrist[1]:.1f})")
            
            middle_tip_idx = start + 12
            src_hand_len = 0
            if s_scores[middle_tip_idx] > 0.2:
                src_hand_len = calculate_distance(s_kpts[w_idx], s_kpts[middle_tip_idx])
                print(f"      src_hand_length: {src_hand_len:.1f}px")
            
            transferred_count = 0
            for i in range(21):
                idx = start + i
                if s_scores[idx] > 0.2:
                    rel = s_kpts[idx] - src_wrist
                    t_kpts[idx] = trans_wrist + rel
                    t_scores[idx] = s_scores[idx]
                    transferred_count += 1
            
            print(f"      transferred: {transferred_count}/21 keypoints from SOURCE")
            
            if t_scores[middle_tip_idx] > 0 and src_hand_len > 0:
                trans_hand_len = calculate_distance(t_kpts[w_idx], t_kpts[middle_tip_idx])
                print(f"      trans_hand_length: {trans_hand_len:.1f}px")
                print(f"      size ratio: {trans_hand_len/src_hand_len:.3f} (should be ~1.0)")
'''

with open(r'D:\2025\pose_extractor\pose_transfer\transfer\logic\hands.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ hands.py 재작성 완료")

# 검증
import subprocess
result = subprocess.run(
    [r'C:\Users\durup\anaconda3\envs\jyk\python.exe', '-c',
     "from pose_transfer.transfer.logic.hands import HandTransfer; import inspect; print('Signature:', inspect.signature(HandTransfer.transfer))"],
    capture_output=True, text=True, cwd=r'D:\2025\pose_extractor'
)
print(result.stdout)
if result.stderr:
    print("ERROR:", result.stderr)
