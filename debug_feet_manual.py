"""
Ghost Filter 발 키포인트 제거 원인 - 수동 추적
"""
import json
import numpy as np

# 테스트 데이터 로드
test_file = "test_io/outputs/ghost filter_use20260121/origin_full_0001_kp.json"
with open(test_file, 'r') as f:
    data = json.load(f)

kp_flat = data['people'][0]['pose_keypoints_2d']
num_kpts = len(kp_flat) // 3
keypoints = np.array([[kp_flat[i*3], kp_flat[i*3+1]] for i in range(num_kpts)])
scores = np.array([kp_flat[i*3+2] for i in range(num_kpts)])

print("=" * 80)
print("Ghost Filter 발 키포인트 제거 원인 추적")
print("=" * 80)

# 계층 구조
hierarchy = {
    # Feet -> Ankle
    17: 15, 18: 15, 19: 15,
    20: 16, 21: 16, 22: 16,
    # Ankle -> Knee
    15: 13, 16: 14,
    # Knee -> Hip
    13: 11, 14: 12
}

# 이미지 크기
h, w = 916, 916
boundary_tolerance = 10.0  # default.yaml에서
confidence_threshold = 0.1  # default.yaml에서

print("\n[1] 원본 키포인트 - 하체 전체")
print("-" * 80)
lower_body = {
    11: "LHip", 12: "RHip",
    13: "LKnee", 14: "RKnee",
    15: "LAnkle", 16: "RAnkle",
    17: "LBigToe", 18: "LSmallToe", 19: "LHeel",
    20: "RBigToe", 21: "RSmallToe", 22: "RHeel"
}

for idx, name in lower_body.items():
    x, y = keypoints[idx]
    conf = scores[idx]
    
    # boundary check
    near_boundary = (
        x <= boundary_tolerance or 
        x >= w - boundary_tolerance or
        y <= boundary_tolerance or 
        y >= h - boundary_tolerance
    )
    
    boundary_str = "⚠️ BOUNDARY!" if near_boundary else ""
    print(f"  idx={idx:2d} {name:12s}: x={x:6.1f} y={y:6.1f} conf={conf:.2f} {boundary_str}")

# 발목이 경계에 걸리는지 확인
print(f"\n[2] 발목(Ankle) Boundary Check")
print("-" * 80)
print(f"  boundary_tolerance = {boundary_tolerance}px")
print(f"  image size: {w} x {h}")
print(f"  boundary zones: x<={boundary_tolerance}, x>={w-boundary_tolerance}, y<={boundary_tolerance}, y>={h-boundary_tolerance}")

ankle_removed = {}
for idx in [15, 16]:
    name = lower_body[idx]
    x, y = keypoints[idx]
    conf = scores[idx]
    
    is_boundary = (
        x <= boundary_tolerance or 
        x >= w - boundary_tolerance or
        y <= boundary_tolerance or 
        y >= h - boundary_tolerance
    )
    
    # Ghost Filter Step 2 로직
    # boundary 값이면 is_dummy_coord = True
    # parent가 없으면(ankle은 knee가 parent) remove됨
    
    print(f"\n  {name} (idx={idx}):")
    print(f"    위치: x={x:.1f}, y={y:.1f}")
    print(f"    boundary?: {is_boundary}")
    print(f"    confidence: {conf:.2f}")
    
    if is_boundary:
        # parent (knee) 확인
        parent_idx = hierarchy[idx]
        parent_name = lower_body[parent_idx]
        parent_x, parent_y = keypoints[parent_idx]
        parent_conf = scores[parent_idx]
        parent_valid = parent_conf >= confidence_threshold
        
        print(f"    ⚠️  경계값 감지!")
        print(f"    부모: {parent_name} (idx={parent_idx})")
        print(f"    부모 위치: x={parent_x:.1f}, y={parent_y:.1f}")
        print(f"    부모 conf: {parent_conf:.2f} (threshold={confidence_threshold})")
        print(f"    부모 유효?: {parent_valid}")
        
        if not parent_valid:
            print(f"    🔴 결과: REMOVED (더미 + 부모 없음)")
            ankle_removed[idx] = True
        else:
            print(f"    🟡 결과: OUT_OF_FRAME (더미 + 부모 있음)")
            ankle_removed[idx] = False
    else:
        print(f"    ✅ 정상")
        ankle_removed[idx] = False

# Chain Kill 시뮬레이션
print(f"\n[3] Chain Kill 시뮬레이션")
print("-" * 80)

for foot_idx in [17, 18, 19, 20, 21, 22]:
    foot_name = lower_body[foot_idx]
    ankle_idx = hierarchy[foot_idx]
    ankle_name = lower_body[ankle_idx]
    
    print(f"\n  {foot_name} (idx={foot_idx}):")
    print(f"    부모: {ankle_name} (idx={ankle_idx})")
    
    if ankle_idx in ankle_removed and ankle_removed[ankle_idx]:
        print(f"    🔴 부모(발목)가 제거됨 → Chain Kill!")
        print(f"    사유: orphan_node(parent_{ankle_idx}_dead)")
    else:
        print(f"    ✅ 부모 살아있음 → 발가락도 유지")

print("\n" + "=" * 80)
print("결론:")
print("=" * 80)
print("발이 안 보이는 이유:")
print("1. 발목(Ankle)이 boundary 근처에 있음 (경계값 감지)")
print("2. Ghost Filter Step 2에서 발목이 'dummy' 또는 'out_of_frame'으로 마킹")
print("3. 만약 발목이 완전 제거되면(REMOVED), Chain Kill로 발가락도 함께 제거")
print("=" * 80)
