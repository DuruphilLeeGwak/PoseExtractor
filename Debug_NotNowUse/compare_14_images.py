import json
import cv2
import matplotlib.pyplot as plt

# 14_kp.json 읽기
with open('test_io/outputs/20260124_000920/14_kp.json', 'r') as f:
    data = json.load(f)

# OpenPose 형식
pose_kpts = data['people'][0]['pose_keypoints_2d']
kpts = [[pose_kpts[i], pose_kpts[i+1], pose_kpts[i+2]] for i in range(0, len(pose_kpts), 3)]
print(f"총 키포인트 개수: {len(kpts)}")
print(f"\n좌상단 영역 키포인트 (x < 300, y < 250):")
print(f"{'Index':<8} {'Name':<25} {'X':>10} {'Y':>10} {'Score':>8}")
print("=" * 70)

# 좌상단 키포인트 찾기
BODY_NAMES = [
    "Nose", "Neck", "R-Shoulder", "R-Elbow", "R-Wrist",
    "L-Shoulder", "L-Elbow", "L-Wrist", "R-Hip", "R-Knee", "R-Ankle",
    "L-Hip", "L-Knee", "L-Ankle", "R-Eye", "L-Eye", "R-Ear", "L-Ear"
]
FEET_NAMES = ["R-BigToe", "R-SmallToe", "R-Heel", "L-BigToe", "L-SmallToe", "L-Heel"]

def get_kpt_name(i):
    if i < len(BODY_NAMES):
        return f"Body-{BODY_NAMES[i]}"
    elif i < len(BODY_NAMES) + len(FEET_NAMES):
        return f"Feet-{FEET_NAMES[i - len(BODY_NAMES)]}"
    else:
        return f"Unknown-{i}"

corner_kpts = []
for i in range(len(kpts)):
    x, y, score = kpts[i]
    if x < 300 and y < 250 and score > 0.1:
        name = get_kpt_name(i)
        corner_kpts.append((i, x, y, score, name))
        print(f"{i:<8} {name:<25} {x:>10.1f} {y:>10.1f} {score:>8.3f}")

print(f"\n총 {len(corner_kpts)}개 발견")

# 14_sk.jpg 이미지 읽기
img = cv2.imread('test_io/outputs/20260124_000920/14_sk.jpg')
print(f"\n이미지 크기: {img.shape[1]} x {img.shape[0]}")

# 좌상단 영역 확대해서 표시
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# 전체 이미지
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Full Image')
ax1.set_xlim(0, img.shape[1])
ax1.set_ylim(img.shape[0], 0)

# 좌상단 영역에 사각형 그리기
from matplotlib.patches import Rectangle
rect = Rectangle((0, 0), 300, 250, linewidth=3, edgecolor='yellow', facecolor='none')
ax1.add_patch(rect)

# 키포인트 표시
for i, x, y, score, name in corner_kpts:
    ax1.plot(x, y, 'ro', markersize=10)
    ax1.text(x, y, f"{i}", color='white', fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

# 좌상단 확대
crop = img[0:300, 0:400]  # y, x
ax2.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
ax2.set_title('Top-Left Corner (Zoomed)')
ax2.set_xlim(0, 400)
ax2.set_ylim(300, 0)

# 키포인트 표시
for i, x, y, score, name in corner_kpts:
    if x < 400 and y < 300:
        ax2.plot(x, y, 'ro', markersize=12)
        ax2.text(x, y+10, f"{i}:{name}", color='white', fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

plt.tight_layout()
plt.savefig('14_corner_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n이미지 저장됨: 14_corner_analysis.png")
plt.close()

# 모든 키포인트 분석
print(f"\n전체 키포인트 분포:")
print(f"X 범위: {min(k[0] for k in kpts):.1f} ~ {max(k[0] for k in kpts):.1f}")
print(f"Y 범위: {min(k[1] for k in kpts):.1f} ~ {max(k[1] for k in kpts):.1f}")
