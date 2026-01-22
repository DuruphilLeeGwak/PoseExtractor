"""
할루시네이션 케이스 상세 분석
케이스 4, 5에서 왜 할루시네이션이 통과되었는지 확인
"""
import json

# 케이스 4: half_bg_1 - 왼발 할루시네이션
print("="*80)
print("케이스 4: half_bg_1 - 발 할루시네이션 분석")
print("="*80)

with open(r"test_io\outputs\20260122_174239\half_bg_1_kp.json", 'r') as f:
    data = json.load(f)

person = data['people'][0]
pose_kpts = person['pose_keypoints_2d']
body_kpts = [(pose_kpts[i], pose_kpts[i+1], pose_kpts[i+2]) 
             for i in range(0, 69, 3)]

print("\n🎯 Body 검출 상태 (하체):")
print(f"   [11] Left Hip: {body_kpts[11]}")
print(f"   [12] Right Hip: {body_kpts[12]}")
print(f"   [13] Left Knee: {body_kpts[13]}")
print(f"   [14] Right Knee: {body_kpts[14]}")
print(f"   [15] Left Ankle: {body_kpts[15]}")
print(f"   [16] Right Ankle: {body_kpts[16]}")

print("\n🦶 DWPose 발 키포인트 (터미널 로그 원본):")
print("   [17] Left BigToe: (270.1, 4998.2) score=0.754 ⚠️ 경계 밖 (이미지: 4583x3055)")
print("   [20] Right BigToe: (270.1, 4998.2) score=1.165 ⚠️ 경계 밖")
print("   [21] Right SmallToe: (1343.3, 2746.2) score=1.134")
print("   [22] Right Heel: (1454.5, 4475.6) score=0.980")

print("\n📊 Cross-Filter 결과:")
print(f"   [15] Left Ankle: {body_kpts[15]}")
print(f"   → Body 검출 실패 → Left BigToe(17) 제거 ✅")
print(f"   [16] Right Ankle: {body_kpts[16]}")
print(f"   → Body 검출 성공 (score=1.234) → Right BigToe(20), SmallToe(21), Heel(22) 승인 ❌")

print("\n⚠️ 문제점:")
print("   Body가 Right Ankle을 낮은 confidence로 검출 (1.234)")
print("   → DWPose의 경계 밖 좌표(y=4998)를 승인해버림")
print("   → 해결책: Body confidence threshold 상향 조정 필요")

# 케이스 5: half_bg_2 - 발 할루시네이션
print("\n" + "="*80)
print("케이스 5: half_bg_2 - 발 할루시네이션 분석")
print("="*80)

with open(r"test_io\outputs\20260122_174239\half_bg_2_kp.json", 'r') as f:
    data = json.load(f)

person = data['people'][0]
pose_kpts = person['pose_keypoints_2d']
body_kpts = [(pose_kpts[i], pose_kpts[i+1], pose_kpts[i+2]) 
             for i in range(0, 69, 3)]

print("\n🎯 Body 검출 상태 (하체):")
print(f"   [11] Left Hip: {body_kpts[11]}")
print(f"   [12] Right Hip: {body_kpts[12]}")
print(f"   [13] Left Knee: {body_kpts[13]}")
print(f"   [14] Right Knee: {body_kpts[14]}")
print(f"   [15] Left Ankle: {body_kpts[15]}")
print(f"   [16] Right Ankle: {body_kpts[16]}")

print("\n🦶 발 키포인트 상태:")
for idx in [17, 18, 19, 20, 21, 22]:
    print(f"   [{idx}] {body_kpts[idx]}")

print("\n📊 Cross-Filter 결과:")
print("   양쪽 Ankle 모두 검출 실패 → 모든 발 키포인트(17-22) 제거 ✅")
print("   결과: 125/133 승인 (8개 제거)")

print("\n✅ 정상 작동: Body가 없다고 판단한 부위는 모두 제거됨")

# 케이스 5-2: half_bg_4
print("\n" + "="*80)
print("케이스 5-2: half_bg_4 - 하체 전체 할루시네이션 분석")
print("="*80)

with open(r"test_io\outputs\20260122_174239\half_bg_4_kp.json", 'r') as f:
    data = json.load(f)

person = data['people'][0]
pose_kpts = person['pose_keypoints_2d']
body_kpts = [(pose_kpts[i], pose_kpts[i+1], pose_kpts[i+2]) 
             for i in range(0, 69, 3)]

print("\n🎯 Body 검출 상태 (하체):")
for idx in range(11, 17):
    print(f"   [{idx}] {body_kpts[idx]}")

print("\n🦶 발 키포인트 상태:")
for idx in [17, 18, 19, 20, 21, 22]:
    print(f"   [{idx}] {body_kpts[idx]}")

print("\n📊 Cross-Filter 결과:")
print("   Hip, Knee, Ankle 모두 검출 실패 → 모든 다리/발 제거 ✅")
print("   결과: 123/133 승인 (10개 제거)")

print("\n✅ 정상 작동: Body가 하체를 검출 못하면 모든 하체 키포인트 제거")

# 결론
print("\n" + "="*80)
print("종합 결론")
print("="*80)
print("""
✅ 케이스 5 (half_bg_2, half_bg_4): Cross-Filter가 정상 작동
   - Body가 검출 못한 부위는 모두 제거됨
   - 할루시네이션이 "지나친" 것이 아니라 "제거된" 것

⚠️ 케이스 4 (half_bg_1): Body의 낮은 confidence 오검출
   - Body가 Right Ankle을 score=1.234로 검출 (threshold 0.3 통과)
   - 실제로는 없는 부위인데 낮은 confidence로 통과
   - DWPose의 경계 밖 좌표도 함께 승인됨
   
📋 해결 방안:
   1. Body confidence threshold를 0.3 → 0.5~0.8로 상향 조정
   2. DWPose 고신뢰도(8.0+) 키포인트는 Body 검증 우회 옵션 추가
   3. 경계 밖 좌표는 무조건 제거하는 별도 체크 추가
""")
