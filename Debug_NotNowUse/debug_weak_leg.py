"""
Ghost Filter weak_leg 로직 테스트
"""
import json
import numpy as np

test_file = "test_io/outputs/ghost filter_use20260121/origin_full_0001_kp.json"
with open(test_file, 'r') as f:
    data = json.load(f)

kp_flat = data['people'][0]['pose_keypoints_2d']
num_kpts = len(kp_flat) // 3
keypoints = np.array([[kp_flat[i*3], kp_flat[i*3+1]] for i in range(num_kpts)])
scores = np.array([kp_flat[i*3+2] for i in range(num_kpts)])

h, w = 916, 916

print("=" * 80)
print("weak_leg 로직 테스트")
print("=" * 80)

# Ghost Filter의 weak_leg 로직 (line 323-334)
# if idx in [17, 18, 19, 20, 21, 22]:  # 발만 체크
#     boundary_near_threshold = h - 50  # 하단 50px 이내
#     if y >= boundary_near_threshold and scores[idx] < 4.5:
#         is_weak_leg = True

boundary_near_threshold = h - 50  # 916 - 50 = 866
weak_leg_conf_threshold = 4.5

print(f"\n[1] weak_leg 로직 파라미터")
print("-" * 80)
print(f"  이미지 높이: {h}px")
print(f"  하단 경계: {boundary_near_threshold}px (h - 50)")
print(f"  신뢰도 임계값: {weak_leg_conf_threshold}")
print(f"  조건: y >= {boundary_near_threshold} AND conf < {weak_leg_conf_threshold}")

print(f"\n[2] 발 키포인트 (idx 17-22) 분석")
print("-" * 80)

feet_names = {
    17: "LBigToe",
    18: "LSmallToe",
    19: "LHeel",
    20: "RBigToe",
    21: "RSmallToe",
    22: "RHeel"
}

for idx in [17, 18, 19, 20, 21, 22]:
    x, y = keypoints[idx]
    conf = scores[idx]
    
    y_condition = y >= boundary_near_threshold
    conf_condition = conf < weak_leg_conf_threshold
    is_weak = y_condition and conf_condition
    
    print(f"\n  {feet_names[idx]:12s} (idx={idx}):")
    print(f"    위치: x={x:6.1f}, y={y:6.1f}")
    print(f"    y >= {boundary_near_threshold}?: {y_condition} (실제 y={y:.1f})")
    print(f"    conf < {weak_leg_conf_threshold}?: {conf_condition} (실제 conf={conf:.2f})")
    
    if is_weak:
        print(f"    🔴 weak_leg 판정! → DUMMY로 처리됨")
    else:
        if not y_condition:
            print(f"    ✅ y 위치가 경계 밖 (y={y:.1f} < {boundary_near_threshold})")
        if not conf_condition:
            print(f"    ✅ 신뢰도 충분 (conf={conf:.2f} >= {weak_leg_conf_threshold})")

print("\n" + "=" * 80)
print("결론:")
print("=" * 80)
print("이 테스트 데이터에서:")
print("- 모든 발 키포인트의 y < 866 (하단 경계 밖)")
print("- 모든 발 키포인트의 conf > 6.0 (신뢰도 매우 높음)")
print("→ weak_leg 로직으로는 제거되지 않음!")
print()
print("그렇다면 다른 필터링 단계를 확인해야 함:")
print("1. Step 2의 다른 로직 (boundary values, out of frame, anatomical)")
print("2. Step 3의 clustering")
print("3. Step 3.5의 occlusion detection")
print("4. Step 5의 chain kill")
print("=" * 80)
