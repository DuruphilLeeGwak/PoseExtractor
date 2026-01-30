"""
실제 Ghost Filter 실행 + 결과 확인
"""
import sys
import json
import numpy as np
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from pose_transfer.logic.ghost_filter import GhostFilter, GhostFilterConfig

# 테스트 데이터
test_file = "test_io/outputs/ghost filter_use20260121/origin_full_0001_kp.json"
with open(test_file, 'r') as f:
    data = json.load(f)

kp_flat = data['people'][0]['pose_keypoints_2d']
num_kpts = len(kp_flat) // 3
keypoints = np.array([[kp_flat[i*3], kp_flat[i*3+1]] for i in range(num_kpts)])
scores = np.array([kp_flat[i*3+2] for i in range(num_kpts)])

h, w = 916, 916

print("=" * 80)
print("실제 Ghost Filter 실행")
print("=" * 80)

# Ghost Filter 생성 (default 설정)
config = GhostFilterConfig()
config.enabled = True
config.boundary_tolerance = 10.0
config.confidence_threshold = 0.1

ghost_filter = GhostFilter(config)

print(f"\n[1] Ghost Filter 설정")
print("-" * 80)
print(f"  enabled: {config.enabled}")
print(f"  boundary_tolerance: {config.boundary_tolerance}")
print(f"  confidence_threshold: {config.confidence_threshold}")
print(f"  check_bounds: {config.check_bounds}")
print(f"  check_boundary_values: {config.check_boundary_values}")
print(f"  check_consistency: {config.check_consistency}")

# Ghost Filter 적용
print(f"\n[2] Ghost Filter 실행")
print("-" * 80)

result = ghost_filter.filter_keypoints(
    keypoints=keypoints,
    scores=scores,
    image_width=w,
    image_height=h,
    debug=True,
    print_to_console=False  # 너무 많은 출력 방지
)

print(f"\n[3] 필터링 결과")
print("-" * 80)
print(f"  removed_indices: {len(result.removed_indices)}개")
print(f"  occluded_indices: {len(result.occluded_indices)}개")
print(f"  out_of_frame_indices: {len(result.out_of_frame_indices)}개")

# 발 키포인트 상태 확인
feet_names = {
    15: "LAnkle", 16: "RAnkle",
    17: "LBigToe", 18: "LSmallToe", 19: "LHeel",
    20: "RBigToe", 21: "RSmallToe", 22: "RHeel"
}

print(f"\n[4] 발 키포인트 상태")
print("-" * 80)

for idx in [15, 16, 17, 18, 19, 20, 21, 22]:
    name = feet_names[idx]
    original_conf = scores[idx]
    filtered_conf = result.scores[idx]
    
    status_parts = []
    if idx in result.removed_indices:
        reason = result.removal_reasons.get(idx, 'unknown')
        status_parts.append(f"🔴 REMOVED ({reason})")
    if idx in result.occluded_indices:
        status_parts.append("🟡 OCCLUDED")
    if idx in result.out_of_frame_indices:
        status_parts.append("🟠 OUT_OF_FRAME")
    if not status_parts and filtered_conf > 0.0:
        status_parts.append("✅ ALIVE")
    
    status = " | ".join(status_parts) if status_parts else "❓ UNKNOWN"
    
    x, y = keypoints[idx]
    print(f"  {name:12s} (idx={idx:2d}): ")
    print(f"    위치: x={x:6.1f}, y={y:6.1f}")
    print(f"    신뢰도: {original_conf:.2f} → {filtered_conf:.2f}")
    print(f"    상태: {status}")

# 디버그 정보 출력
if result.debug_info and 'debug_lines' in result.debug_info:
    print(f"\n[5] Ghost Filter 디버그 출력 (발 관련)")
    print("-" * 80)
    debug_lines = result.debug_info['debug_lines']
    feet_debug = [line for line in debug_lines if any(str(idx) in line for idx in [15, 16, 17, 18, 19, 20, 21, 22])]
    if feet_debug:
        for line in feet_debug[:20]:  # 최대 20줄
            print(f"  {line}")
    else:
        print("  (발 관련 디버그 출력 없음)")

print("\n" + "=" * 80)
print("결론:")
print("=" * 80)

# 발 키포인트 제거/마킹 여부 판단
feet_removed = [idx for idx in [17, 18, 19, 20, 21, 22] if idx in result.removed_indices]
feet_out_of_frame = [idx for idx in [17, 18, 19, 20, 21, 22] if idx in result.out_of_frame_indices]
feet_occluded = [idx for idx in [17, 18, 19, 20, 21, 22] if idx in result.occluded_indices]

if feet_removed:
    print(f"발 키포인트 REMOVED: {feet_removed}")
    for idx in feet_removed:
        print(f"  - idx={idx} ({feet_names[idx]}): {result.removal_reasons.get(idx, 'unknown')}")

if feet_out_of_frame:
    print(f"발 키포인트 OUT_OF_FRAME: {feet_out_of_frame}")
    print(f"  → 렌더링 시 스킵됨 (skeleton_renderer.py line 186-188)")

if feet_occluded:
    print(f"발 키포인트 OCCLUDED: {feet_occluded}")
    print(f"  → 렌더링 시 50% 투명도로 표시")

if not feet_removed and not feet_out_of_frame and not feet_occluded:
    print(f"✅ 모든 발 키포인트가 정상 상태")
    print(f"   Ghost Filter는 발을 제거하지 않음!")
    print(f"   → 렌더링 문제일 가능성 높음")

print("=" * 80)
