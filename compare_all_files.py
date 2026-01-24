import json
import numpy as np
from pathlib import Path

# 수정 전 (20260123_235523)
folder1 = Path('test_io/outputs/20260123_235523')
# 수정 후 (20260124_000009)
folder2 = Path('test_io/outputs/20260124_000009')

files = sorted(folder1.glob('*_kp.json'))

print("="*80)
print("전체 파일 발 키포인트 비교")
print("="*80)

issues = []

for f in files:
    name = f.stem
    f2 = folder2 / f.name
    
    if not f2.exists():
        continue
    
    d1 = json.load(open(f))
    d2 = json.load(open(f2))
    
    kpts1 = np.array(d1['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    kpts2 = np.array(d2['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    
    # 발 키포인트 (17-22) 변화량 확인
    foot_change = 0
    for i in range(17, 23):
        if kpts1[i][2] > 0.1 or kpts2[i][2] > 0.1:
            diff = np.linalg.norm(kpts1[i][:2] - kpts2[i][:2])
            foot_change = max(foot_change, diff)
    
    # 좌상단 뭉침 (발 키포인트가 좌상단에 있는지)
    corner1 = [(i, kpts1[i]) for i in range(17, 23) if kpts1[i][0] < 500 and kpts1[i][1] < 500 and kpts1[i][2] > 0.1]
    corner2 = [(i, kpts2[i]) for i in range(17, 23) if kpts2[i][0] < 500 and kpts2[i][1] < 500 and kpts2[i][2] > 0.1]
    
    if foot_change > 10 or len(corner1) != len(corner2):
        issues.append({
            'file': name,
            'foot_change': foot_change,
            'corner_before': len(corner1),
            'corner_after': len(corner2)
        })
        print(f"{name}: 발 변화 {foot_change:.1f}px, 좌상단 {len(corner1)}→{len(corner2)}")

print(f"\n총 {len(issues)}개 파일에서 변화 발견")

if len(issues) == 0:
    print("\n✅ 모든 파일이 동일합니다. 수정 전후 차이 없음.")
