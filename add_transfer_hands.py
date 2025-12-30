"""_transfer_hands 메서드를 engine.py에 추가"""

with open(r'D:\2025\pose_extractor\pose_transfer\transfer\engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# _transfer_hands 메서드 정의
transfer_hands_method = '''
    def _transfer_hands(self, trans_kpts, trans_scores, src_kpts, src_scores, ref_kpts, ref_scores, global_scale):
        """손 전이: src 우선, src 없으면 ref 사용"""
        print("\\n" + "="*60)
        print("🖐️ _transfer_hands() START")
        print("="*60)
        print(f"   global_scale: {global_scale:.3f}")
        
        LW, RW = 9, 10
        
        hands = [
            (LW, 91, 112, "Left"),
            (RW, 112, 133, "Right")
        ]
        
        for wrist_idx, start, end, side in hands:
            print(f"\\n   [{side}] Checking wrist...")
            print(f"      wrist_idx={wrist_idx}")
            print(f"      trans_wrist_score={trans_scores[wrist_idx]:.3f}")
            
            if trans_scores[wrist_idx] < 0.1:
                print(f"      ❌ trans wrist score < 0.1, SKIP")
                continue
            
            trans_wrist = trans_kpts[wrist_idx]
            
            # 전략: src 손 우선, 없으면 ref 손 사용
            src_hand_count = sum(1 for i in range(start, min(end, len(src_scores))) if src_scores[i] > 0.2)
            ref_hand_count = sum(1 for i in range(start, min(end, len(ref_scores))) if ref_scores[i] > 0.2)
            
            print(f"      src_hand_count={src_hand_count}/21")
            print(f"      ref_hand_count={ref_hand_count}/21")
            
            if src_hand_count > 5:
                # Src 손 사용
                src_wrist = src_kpts[wrist_idx]
                transferred = 0
                for idx in range(start, end):
                    if idx < len(src_scores) and src_scores[idx] > 0.2:
                        rel = src_kpts[idx] - src_wrist
                        trans_kpts[idx] = trans_wrist + rel
                        trans_scores[idx] = src_scores[idx]
                        transferred += 1
                print(f"      ✅ Used SOURCE: {transferred}/21 keypoints")
            
            elif ref_hand_count > 5:
                # Ref 손 사용 (global_scale 적용)
                print(f"      → Using REFERENCE (src_hand_count={src_hand_count} <= 5)")
                ref_wrist = ref_kpts[wrist_idx]
                transferred = 0
                for idx in range(start, end):
                    if idx < len(ref_scores) and ref_scores[idx] > 0.2:
                        rel = ref_kpts[idx] - ref_wrist
                        scaled_rel = rel * global_scale
                        trans_kpts[idx] = trans_wrist + scaled_rel
                        trans_scores[idx] = ref_scores[idx] * 0.9
                        transferred += 1
                print(f"      ✅ Used REFERENCE (scaled): {transferred}/21 keypoints")
            
            else:
                print(f"      ⚠️ No valid hand in both src and ref (src={src_hand_count}, ref={ref_hand_count})")
'''

# _calculate_torso_ratio 앞에 삽입
marker = '    def _calculate_torso_ratio(self, src_kpts, src_scores, ref_kpts, ref_scores):'
if marker in content:
    content = content.replace(marker, transfer_hands_method + '\n' + marker)
    
    with open(r'D:\2025\pose_extractor\pose_transfer\transfer\engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ _transfer_hands 메서드 추가 완료")
else:
    print("❌ _calculate_torso_ratio를 찾을 수 없습니다")

# 검증
import subprocess
result = subprocess.run(
    [r'C:\Users\durup\anaconda3\envs\jyk\python.exe', '-c',
     "from pose_transfer.transfer.engine import PoseTransferEngine; print('Has _transfer_hands:', hasattr(PoseTransferEngine, '_transfer_hands'))"],
    capture_output=True, text=True, cwd=r'D:\2025\pose_extractor'
)
print(result.stdout)
if result.stderr:
    print("ERROR:", result.stderr)
