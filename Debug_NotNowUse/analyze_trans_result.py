import json
import numpy as np

# Trans 결과 로드
d = json.load(open('test_io/outputs/20260124_000009/14_kp.json'))
kpts = np.array(d['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

print("="*60)
print("Trans 결과 분석 (14.jpg)")
print("="*60)

# 발 키포인트 (17-22)
print("\n발 키포인트 (17-22):")
foot_names = ['L-BigToe', 'L-SmallToe', 'L-Heel', 'R-BigToe', 'R-SmallToe', 'R-Heel']
for i, name in enumerate(foot_names, start=17):
    print(f"  {i:2d} {name:12s}: x={kpts[i][0]:6.1f}, y={kpts[i][1]:6.1f}, score={kpts[i][2]:.2f}")

# 좌상단 키포인트 (x<200, y<200)
print(f"\n좌상단 키포인트 (x<200, y<200):")
corner = [(i, kpts[i]) for i in range(len(kpts)) if kpts[i][0] < 200 and kpts[i][1] < 200 and kpts[i][2] > 0.1]
if len(corner) == 0:
    print("  없음")
else:
    for i, k in corner[:15]:
        print(f"  idx {i:3d}: x={k[0]:6.1f}, y={k[1]:6.1f}, score={k[2]:.2f}")

# 이미지 크기 확인 (debug.txt에서)
print(f"\n총 키포인트 개수: {len(kpts)}")
print(f"유효 키포인트 (score > 0.1): {len([k for k in kpts if k[2] > 0.1])}")

# 발목-발가락 거리
ankle_r = kpts[16][:2]
toe_r = kpts[20][:2]
dist = np.linalg.norm(ankle_r - toe_r)
print(f"\nR-Ankle (16) → R-BigToe (20) 거리: {dist:.1f}px")
