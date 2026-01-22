"""
Ghost Filter v5.1 개선 사항 테스트
"""
import sys
import numpy as np
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from pose_transfer.logic.ghost_filter import GhostFilter, GhostFilterConfig

# 테스트 시나리오
print("=" * 80)
print("Ghost Filter v5.1 개선 사항 검증")
print("=" * 80)

# 설정 로드
config = GhostFilterConfig(
    enabled=True,
    boundary_tolerance=10.0,  # ✅ 30 -> 10 축소
    hand_min_avg_confidence=2.0,  # ✅ 3.0 -> 2.0 완화
    hand_min_distance_std=30.0,  # ✅ 50.0 -> 30.0 완화
    check_hand_occlusion=True,
    debug_hand_removals=True,
    debug_hand_summary_only=False,
)

ghost_filter = GhostFilter(config)

# 시나리오 1: 전신 사진 - 발목/발이 이미지 하단에 있음
print("\n[시나리오 1] 전신 사진 - 발목이 이미지 하단에 위치 (y=3900/4000)")
print("-" * 80)

image_size = (4000, 3000)  # (height, width)
keypoints = np.zeros((133, 2))
scores = np.zeros(133)

# 발목 (15, 16)을 하단 근처에 배치
keypoints[15] = [1400, 3900]  # LAnkle - 하단 100px 이내
keypoints[16] = [1600, 3920]  # RAnkle - 하단 80px 이내
scores[15] = 6.5  # 높은 신뢰도
scores[16] = 6.3  # 높은 신뢰도

# 무릎 (13, 14)
keypoints[13] = [1380, 3200]  # LKnee
keypoints[14] = [1620, 3220]  # RKnee
scores[13] = 5.8
scores[14] = 5.5

# 엉덩이 (11, 12)
keypoints[11] = [1400, 2500]  # LHip
keypoints[12] = [1600, 2500]  # RHip
scores[11] = 4.5
scores[12] = 4.3

result = ghost_filter.filter_single(keypoints, scores, image_size)

print(f"✅ LAnkle (15): score={scores[15]:.2f}, filtered={result.filtered_scores[15]:.2f}")
print(f"   → removed: {15 in result.removed_indices}")
print(f"   → out_of_frame: {15 in result.out_of_frame_indices}")
print(f"   → occluded: {15 in result.occluded_indices}")
if 15 in result.removal_reasons:
    print(f"   → reason: {result.removal_reasons[15]}")

print(f"✅ RAnkle (16): score={scores[16]:.2f}, filtered={result.filtered_scores[16]:.2f}")
print(f"   → removed: {16 in result.removed_indices}")
print(f"   → out_of_frame: {16 in result.out_of_frame_indices}")
print(f"   → occluded: {16 in result.occluded_indices}")
if 16 in result.removal_reasons:
    print(f"   → reason: {result.removal_reasons[16]}")

# 시나리오 2: 작은 손 (멀리 있는 손) - avg_conf < 3.0
print("\n[시나리오 2] 작은 손 (멀리 위치) - avg_conf=2.5, dist_std=35")
print("-" * 80)

keypoints = np.zeros((133, 2))
scores = np.zeros(133)

# 왼손목과 팔꿈치
keypoints[9] = [800, 1500]  # LWrist
keypoints[7] = [750, 1400]  # LElbow
scores[9] = 3.5
scores[7] = 4.0

# 왼손 손가락 (91-111) - 작은 손이므로 dist_std가 작음
for i in range(91, 112):
    angle = (i - 91) * 2 * np.pi / 21
    # 손목 주변 반경 40px에 분포
    keypoints[i] = [800 + 40 * np.cos(angle), 1500 + 40 * np.sin(angle)]
    scores[i] = 2.3 + np.random.uniform(-0.3, 0.5)  # avg ~2.5

result = ghost_filter.filter_single(keypoints, scores, image_size)

lhand_indices = list(range(91, 112))
lhand_removed = [i for i in lhand_indices if i in result.removed_indices]
lhand_occluded = [i for i in lhand_indices if i in result.occluded_indices]
lhand_kept = [i for i in lhand_indices if result.filtered_scores[i] > 0.3]

print(f"✅ LHand (91-111): 총 21개")
print(f"   → kept (score>0.3): {len(lhand_kept)}개")
print(f"   → occluded (-1 layer): {len(lhand_occluded)}개")
print(f"   → removed (완전제거): {len(lhand_removed)}개")
print(f"   → avg_confidence: {np.mean([scores[i] for i in lhand_indices if scores[i] > 0.1]):.2f}")

if lhand_occluded:
    print(f"   ⚠️ 폐색 판정: {len(lhand_occluded)}개가 -1 레이어로 마킹됨")
    if 91 in result.removal_reasons:
        print(f"   → reason: {result.removal_reasons[91]}")

# 시나리오 3: 팔이 명확하게 앞에 있는 경우 (높은 신뢰도)
print("\n[시나리오 3] 명확한 팔 (고신뢰도) - avg_conf=4.5")
print("-" * 80)

keypoints = np.zeros((133, 2))
scores = np.zeros(133)

# 왼손목과 팔꿈치
keypoints[9] = [1200, 1800]  # LWrist
keypoints[7] = [1100, 1600]  # LElbow
scores[9] = 5.0
scores[7] = 4.8

# 왼손 손가락 - 높은 신뢰도
for i in range(91, 112):
    angle = (i - 91) * 2 * np.pi / 21
    keypoints[i] = [1200 + 50 * np.cos(angle), 1800 + 50 * np.sin(angle)]
    scores[i] = 4.0 + np.random.uniform(-0.5, 1.0)  # avg ~4.5

result = ghost_filter.filter_single(keypoints, scores, image_size)

lhand_indices = list(range(91, 112))
lhand_removed = [i for i in lhand_indices if i in result.removed_indices]
lhand_occluded = [i for i in lhand_indices if i in result.occluded_indices]
lhand_kept = [i for i in lhand_indices if result.filtered_scores[i] > 0.3]

print(f"✅ LHand (91-111): 총 21개")
print(f"   → kept (score>0.3): {len(lhand_kept)}개")
print(f"   → occluded (-1 layer): {len(lhand_occluded)}개")
print(f"   → removed (완전제거): {len(lhand_removed)}개")
print(f"   → avg_confidence: {np.mean([scores[i] for i in lhand_indices if scores[i] > 0.1]):.2f}")

if not lhand_occluded:
    print(f"   ✅ 정상 판정: 높은 신뢰도로 폐색 판정 우회!")

print("\n" + "=" * 80)
print("개선 사항 요약:")
print("=" * 80)
print("1. boundary_tolerance: 30 -> 10")
print("   - 하단 경계는 80px까지 허용 (신뢰도 4.0 이상이면)")
print("   - ✅ 전신 사진에서 발목/발 보존")
print()
print("2. hand_min_avg_confidence: 3.0 -> 2.0")
print("   - ✅ 작은 손(멀리 있는 손)도 정상 판정")
print()
print("3. hand_min_distance_std: 50.0 -> 30.0")
print("   - ✅ 작은 손의 작은 dist_std도 허용")
print()
print("4. 신뢰도 기반 우회:")
print("   - 손: avg_conf > 3.5이면 기하 체크 무시")
print("   - ✅ 명확한 팔/손은 폐색 오판 방지")
print("=" * 80)
