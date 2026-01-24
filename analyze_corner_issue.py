import json
import numpy as np
from pathlib import Path

# 최신 결과 로드
result_dir = Path('test_io/outputs/20260124_000920')
json_file = result_dir / '14_kp.json'

d = json.load(open(json_file))
kpts = np.array(d['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

print("="*80)
print(f"📄 파일: {json_file.name}")
print(f"📊 총 키포인트 개수: {len(kpts)}")
print("="*80)

# 키포인트 이름 정의 (133개)
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

# Face 키포인트 (23-90, 68개)
for i in range(68):
    names.append(f'Face-{i}')

# Left Hand (91-111, 21개)
for i in range(21):
    names.append(f'LHand-{i}')

# Right Hand (112-132, 21개)
for i in range(21):
    names.append(f'RHand-{i}')

# 이미지 크기 확인
print(f"\n📐 이미지 크기: {d.get('image_width', 'N/A')} x {d.get('image_height', 'N/A')}")

# 좌상단 영역 키포인트 찾기 (노란 동그라미 영역 추정)
print("\n🔍 좌상단 영역 키포인트 (x < 200, y < 200):")
print(f"{'Index':<6} {'Name':<20} {'X':>8} {'Y':>8} {'Score':>8}")
print("-"*60)

corner_kpts = []
for i in range(len(kpts)):
    if kpts[i][0] < 200 and kpts[i][1] < 200 and kpts[i][2] > 0.1:
        name = names[i] if i < len(names) else f'Unknown-{i}'
        corner_kpts.append((i, kpts[i], name))
        print(f"{i:<6d} {name:<20s} {kpts[i][0]:8.1f} {kpts[i][1]:8.1f} {kpts[i][2]:8.3f}")

print(f"\n📌 총 {len(corner_kpts)}개 발견")

# 키포인트 그룹별 통계
print("\n📊 키포인트 그룹별 분포:")
print(f"  - Body (0-16): {sum(1 for i, k, n in corner_kpts if 0 <= i <= 16)}개")
print(f"  - Feet (17-22): {sum(1 for i, k, n in corner_kpts if 17 <= i <= 22)}개")
print(f"  - Face (23-90): {sum(1 for i, k, n in corner_kpts if 23 <= i <= 90)}개")
print(f"  - LHand (91-111): {sum(1 for i, k, n in corner_kpts if 91 <= i <= 111)}개")
print(f"  - RHand (112-132): {sum(1 for i, k, n in corner_kpts if 112 <= i <= 132)}개")

# 전체 키포인트 범위 확인
valid_kpts = kpts[kpts[:, 2] > 0.1]
if len(valid_kpts) > 0:
    print(f"\n📏 전체 키포인트 범위:")
    print(f"  X: {valid_kpts[:, 0].min():.1f} ~ {valid_kpts[:, 0].max():.1f}")
    print(f"  Y: {valid_kpts[:, 1].min():.1f} ~ {valid_kpts[:, 1].max():.1f}")
