"""engine.py에 torso_ratio + 손 스케일 수정을 직접 적용하는 스크립트"""

# 1. _calculate_torso_ratio 메서드 추가
torso_ratio_method = '''
    def _calculate_torso_ratio(self, src_kpts, src_scores, ref_kpts, ref_scores):
        """몸통 길이 기반 비율 (시점 독립적)"""
        l_sh, r_sh = BODY_KEYPOINTS['left_shoulder'], BODY_KEYPOINTS['right_shoulder']
        l_hip, r_hip = BODY_KEYPOINTS['left_hip'], BODY_KEYPOINTS['right_hip']
        
        # Src torso length
        if (src_scores[l_sh] > 0.3 and src_scores[r_sh] > 0.3 and
            src_scores[l_hip] > 0.3 and src_scores[r_hip] > 0.3):
            src_neck = (src_kpts[l_sh] + src_kpts[r_sh]) / 2
            src_root = (src_kpts[l_hip] + src_kpts[r_hip]) / 2
            src_torso = calculate_distance(src_neck, src_root)
        else:
            return 1.0
        
        # Ref torso length
        if (ref_scores[l_sh] > 0.3 and ref_scores[r_sh] > 0.3 and
            ref_scores[l_hip] > 0.3 and ref_scores[r_hip] > 0.3):
            ref_neck = (ref_kpts[l_sh] + ref_kpts[r_sh]) / 2
            ref_root = (ref_kpts[l_hip] + ref_kpts[r_hip]) / 2
            ref_torso = calculate_distance(ref_neck, ref_root)
        else:
            return 1.0
        
        if ref_torso < 1e-6:
            return 1.0
        
        ratio = src_torso / ref_torso
        # Clamp to prevent extreme shoulder narrowing
        clamped = float(np.clip(ratio, 0.85, 1.25))
        print(f"   📏 Torso Ratio: src={src_torso:.1f}, ref={ref_torso:.1f} → {ratio:.3f} (clamped={clamped:.3f})")
        return clamped
'''

with open(r'D:\2025\pose_extractor\pose_transfer\transfer\engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 2. transfer_shoulders 호출 수정
old_shoulders = '''        # [Body: Upper]
        self.body_logic.transfer_shoulders(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints, processed, transfer_log
        )'''

new_shoulders = '''        # [Body: Upper]
        torso_ratio = self._calculate_torso_ratio(source_keypoints, source_scores, reference_keypoints, reference_scores)
        self.body_logic.transfer_shoulders(
            trans_kpts, trans_scores, source_keypoints, source_scores, reference_keypoints,
            torso_ratio=torso_ratio, processed=processed, log=transfer_log, r_scores=reference_scores
        )'''

# 3. 손 전이 수정
old_hands = '''        # [Hands]
        if self.config.use_hands:
            self.hand_logic.transfer(trans_kpts, trans_scores, reference_keypoints, reference_scores, global_scale, transfer_log)'''

new_hands = '''        # [Hands]
        if self.config.use_hands:
            print("\\n✋ Hands Transfer (scaled)...")
            self.hand_logic.transfer(trans_kpts, trans_scores, source_keypoints, source_scores, global_scale, transfer_log)'''

# 4. _calculate_global_scale 앞에 torso_ratio 메서드 추가
old_marker = '    def _calculate_global_scale(self, src_props, ref_kpts, ref_scores):'
if old_marker in content:
    content = content.replace(old_marker, torso_ratio_method + '\n' + old_marker)

# 5. 어깨/손 호출 교체
if old_shoulders in content:
    content = content.replace(old_shoulders, new_shoulders)
else:
    print("⚠️ shoulders 교체 실패")

if old_hands in content:
    content = content.replace(old_hands, new_hands)
else:
    print("⚠️ hands 교체 실패")

# 6. 저장
with open(r'D:\2025\pose_extractor\pose_transfer\transfer\engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ engine.py 수정 완료")

# 7. 검증
import subprocess
result = subprocess.run(
    [r'C:\Users\durup\anaconda3\envs\jyk\python.exe', '-c',
     "import pose_transfer.transfer.engine as e; print('torso_ratio exists:', hasattr(e.PoseTransferEngine, '_calculate_torso_ratio'))"],
    capture_output=True, text=True, cwd=r'D:\2025\pose_extractor'
)
print(result.stdout)
if result.stderr:
    print("ERROR:", result.stderr)
