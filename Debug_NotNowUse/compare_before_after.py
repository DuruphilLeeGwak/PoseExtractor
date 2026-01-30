import json
import numpy as np

# 수정 전 (20260123_235523)
d1 = json.load(open('test_io/outputs/20260123_235523/14_kp.json'))
kpts1 = np.array(d1['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

# 수정 후 (20260124_000009)
d2 = json.load(open('test_io/outputs/20260124_000009/14_kp.json'))
kpts2 = np.array(d2['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

print("="*70)
print("수정 전 vs 수정 후 비교 (14.jpg)")
print("="*70)

# 발 키포인트 비교
foot_names = ['L-BigToe', 'L-SmallToe', 'L-Heel', 'R-BigToe', 'R-SmallToe', 'R-Heel']
print("\n발 키포인트 (17-22) 비교:")
print(f"{'Idx':<3} {'Name':<12} {'수정 전 (X, Y)':<20} {'수정 후 (X, Y)':<20} {'변화'}")
print("-"*70)
for i, name in enumerate(foot_names, start=17):
    x1, y1, s1 = kpts1[i]
    x2, y2, s2 = kpts2[i]
    if s1 > 0.1 or s2 > 0.1:
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        print(f"{i:<3} {name:<12} ({x1:5.1f}, {y1:5.1f})  ({x2:5.1f}, {y2:5.1f})  Δ={dist:5.1f}px")
    else:
        print(f"{i:<3} {name:<12} (없음)               (없음)")

# R-Ankle ~ R-BigToe 거리 비교
print("\n발목-발가락 거리:")
if kpts1[20][2] > 0.1:
    dist1 = np.linalg.norm(kpts1[16][:2] - kpts1[20][:2])
    print(f"  수정 전: {dist1:.1f}px")
else:
    print(f"  수정 전: 없음")

if kpts2[20][2] > 0.1:
    dist2 = np.linalg.norm(kpts2[16][:2] - kpts2[20][:2])
    print(f"  수정 후: {dist2:.1f}px")
else:
    print(f"  수정 후: 없음")

# 좌상단 키포인트 비교
print("\n좌상단 키포인트 (x<200, y<200):")
corner1 = [(i, kpts1[i]) for i in range(len(kpts1)) if kpts1[i][0] < 200 and kpts1[i][1] < 200 and kpts1[i][2] > 0.1]
corner2 = [(i, kpts2[i]) for i in range(len(kpts2)) if kpts2[i][0] < 200 and kpts2[i][1] < 200 and kpts2[i][2] > 0.1]

print(f"  수정 전: {len(corner1)}개")
for i, k in corner1[:5]:
    print(f"    idx {i:3d}: x={k[0]:6.1f}, y={k[1]:6.1f}")

print(f"  수정 후: {len(corner2)}개")
for i, k in corner2[:5]:
    print(f"    idx {i:3d}: x={k[0]:6.1f}, y={k[1]:6.1f}")
