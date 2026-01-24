import json
import numpy as np
from pathlib import Path

# 최신 출력 폴더 찾기
latest = sorted(Path('io/outputs').glob('*_src_to_ref'))[-1]
print(f'📁 분석 폴더: {latest.name}\n')

# Trans 키포인트 로드
data = json.load(open(latest / 'trans' / 'trans_kp.json'))
kpts = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

names = [
    'Nose', 'L-Eye', 'R-Eye', 'L-Ear', 'R-Ear',  # 0-4
    'L-Shoulder', 'R-Shoulder',  # 5-6
    'L-Elbow', 'R-Elbow',  # 7-8
    'L-Wrist', 'R-Wrist',  # 9-10
    'L-Hip', 'R-Hip',  # 11-12
    'L-Knee', 'R-Knee',  # 13-14
    'L-Ankle', 'R-Ankle',  # 15-16
    'L-BigToe', 'L-SmallToe', 'L-Heel',  # 17-19
    'R-BigToe', 'R-SmallToe', 'R-Heel',  # 20-22
]

print('=== 전체 키포인트 범위 ===')
valid_kpts = kpts[kpts[:, 2] > 0.1]
print(f'X 범위: {valid_kpts[:, 0].min():.1f} ~ {valid_kpts[:, 0].max():.1f}')
print(f'Y 범위: {valid_kpts[:, 1].min():.1f} ~ {valid_kpts[:, 1].max():.1f}')

print('\n=== 좌상단 키포인트 (X<1500, Y<2000) ===')
for i in range(len(kpts)):
    if kpts[i, 2] > 0.1 and kpts[i, 0] < 1500 and kpts[i, 1] < 2000:
        name = names[i] if i < len(names) else f'Face/Hand-{i}'
        print(f'{name:20s} ({i:3d}): x={kpts[i,0]:7.1f}, y={kpts[i,1]:7.1f}, conf={kpts[i,2]:.2f}')

print('\n=== Body 17개 키포인트 위치 ===')
for i in range(17):
    if kpts[i, 2] > 0.1:
        print(f'{names[i]:15s} ({i:2d}): x={kpts[i,0]:7.1f}, y={kpts[i,1]:7.1f}')

# 이상치 탐지
print('\n=== 이상치 탐지 (평균에서 멀리 떨어진 점) ===')
valid = kpts[kpts[:, 2] > 0.1]
mean_x, mean_y = valid[:, 0].mean(), valid[:, 1].mean()
print(f'평균 위치: x={mean_x:.1f}, y={mean_y:.1f}')

outliers = []
for i in range(len(kpts)):
    if kpts[i, 2] > 0.1:
        dx = abs(kpts[i, 0] - mean_x)
        dy = abs(kpts[i, 1] - mean_y)
        if dx > 1500 or dy > 2000:
            name = names[i] if i < len(names) else f'Idx-{i}'
            outliers.append((i, name, kpts[i, 0], kpts[i, 1], dx, dy))

if outliers:
    print(f'발견된 이상치: {len(outliers)}개')
    for idx, name, x, y, dx, dy in outliers:
        print(f'  {name:20s} ({idx:3d}): x={x:7.1f}, y={y:7.1f} (평균서 Δx={dx:.1f}, Δy={dy:.1f})')
else:
    print('이상치 없음')
