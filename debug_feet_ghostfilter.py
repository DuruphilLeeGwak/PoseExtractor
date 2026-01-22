"""
Ghost Filter 발 키포인트 제거 원인 분석 스크립트
"""
import json
import numpy as np
from pathlib import Path
from pose_transfer.logic.ghost_filter import GhostFilter, GhostFilterConfig
from pose_transfer.config.config import load_config

# 테스트 데이터 로드
test_file = Path("test_io/outputs/ghost filter_use20260121/origin_full_0001_kp.json")
with open(test_file, 'r') as f:
    data = json.load(f)

# 키포인트 파싱
kp_flat = data['people'][0]['pose_keypoints_2d']
num_kpts = len(kp_flat) // 3
keypoints = np.array([[kp_flat[i*3], kp_flat[i*3+1]] for i in range(num_kpts)])
scores = np.array([kp_flat[i*3+2] for i in range(num_kpts)])

print("=" * 80)
print("Ghost Filter 발 키포인트 제거 원인 분석")
print("=" * 80)

# 원본 발 키포인트 출력
print("\n[1] 원본 발 키포인트 (Ghost Filter 적용 전)")
print("-" * 80)
feet_indices = {
    15: "LAnkle", 16: "RAnkle",
    17: "LBigToe", 18: "LSmallToe", 19: "LHeel",
    20: "RBigToe", 21: "RSmallToe", 22: "RHeel"
}
for idx, name in feet_indices.items():
    x, y = keypoints[idx]
    conf = scores[idx]
    print(f"  idx={idx:2d} {name:12s}: x={x:6.1f} y={y:6.1f} conf={conf:.2f}")

# 이미지 크기 (추정)
h, w = 916, 916

# Ghost Filter 설정 로드
config = load_config()
ghost_config = GhostFilterConfig(**config['ghost_filter'])
ghost_filter = GhostFilter(ghost_config)

print(f"\n[2] Ghost Filter 설정")
print("-" * 80)
print(f"  enabled: {ghost_config.enabled}")
print(f"  confidence_threshold: {ghost_config.confidence_threshold}")
print(f"  boundary_tolerance: {ghost_config.boundary_tolerance}")
print(f"  check_bounds: {ghost_config.check_bounds}")
print(f"  check_boundary_values: {ghost_config.check_boundary_values}")
print(f"  check_consistency: {ghost_config.check_consistency}")

# Ghost Filter 적용 (디버그 모드)
print(f"\n[3] Ghost Filter 실행 (디버그 모드)")
print("-" * 80)
result = ghost_filter.filter_keypoints(
    keypoints=keypoints,
    scores=scores,
    image_width=w,
    image_height=h,
    debug=True,
    print_to_console=True
)

filtered_keypoints = result.keypoints
filtered_scores = result.scores
removed_indices = result.removed_indices
occluded_indices = result.occluded_indices
out_of_frame_indices = result.out_of_frame_indices
removal_reasons = result.removal_reasons

# 발 키포인트 필터링 결과
print(f"\n[4] 필터링 결과 - 발 키포인트")
print("-" * 80)
for idx, name in feet_indices.items():
    original_conf = scores[idx]
    filtered_conf = filtered_scores[idx]
    status = ""
    
    if idx in removed_indices:
        status = f"🔴 REMOVED (reason: {removal_reasons.get(idx, 'unknown')})"
    elif idx in occluded_indices:
        status = "🟡 OCCLUDED"
    elif idx in out_of_frame_indices:
        status = "🟠 OUT_OF_FRAME"
    elif filtered_conf > 0.0:
        status = "✅ ALIVE"
    else:
        status = "❓ UNKNOWN"
    
    print(f"  idx={idx:2d} {name:12s}: original={original_conf:.2f} → filtered={filtered_conf:.2f} | {status}")

# Step 5 Chain Kill 로직 수동 시뮬레이션
print(f"\n[5] Chain Kill 로직 분석")
print("-" * 80)
hierarchy_rules = ghost_filter.hierarchy_rules
print(f"  hierarchy_rules for feet:")
for idx in [15, 16, 17, 18, 19, 20, 21, 22]:
    parent = hierarchy_rules.get(idx, None)
    if parent is not None:
        parent_score = filtered_scores[parent]
        parent_status = "ALIVE" if parent_score >= ghost_config.confidence_threshold else "DEAD"
        print(f"    idx={idx:2d} → parent={parent:2d} (score={parent_score:.2f}, status={parent_status})")
    else:
        print(f"    idx={idx:2d} → parent=None (root node)")

# 발목이 제거되었는지 확인
print(f"\n[6] 발목(Ankle) 상태 확인")
print("-" * 80)
for idx in [15, 16]:
    name = feet_indices[idx]
    if idx in removed_indices:
        print(f"  ⚠️  {name} (idx={idx}) 제거됨! 사유: {removal_reasons.get(idx, 'unknown')}")
        print(f"      → 이 경우 발가락(17-22)이 chain kill로 제거됩니다!")
    else:
        print(f"  ✅ {name} (idx={idx}) 살아있음 (score={filtered_scores[idx]:.2f})")

print("\n" + "=" * 80)
print("분석 완료")
print("=" * 80)
