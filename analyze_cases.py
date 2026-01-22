"""
특수 케이스 분석 스크립트
Cross-Filter 적용 전후 비교
"""
import json
import os

cases = {
    "1. 💜": "💜",
    "2. 6": "6",
    "3. 42 KB": "42 KB",
    "3. wallpapers for your phone": "wallpapers for your phone",
    "4. half_bg_1": "half_bg_1",
    "5. half_bg_2": "half_bg_2",
    "5. half_bg_4": "half_bg_4",
    "6. half_bg_3": "half_bg_3",
    "7. half_bg_5": "half_bg_5",
}

output_dir = r"test_io\outputs\20260122_174239"

# DWPose keypoint indices
dwpose_legs_feet = {
    "Left Hip": 11, "Right Hip": 12,
    "Left Knee": 13, "Right Knee": 14,
    "Left Ankle": 15, "Right Ankle": 16,
    "Left BigToe": 17, "Right BigToe": 20,
    "Left SmallToe": 18, "Right SmallToe": 21,
    "Left Heel": 19, "Right Heel": 22,
}

# COCO17 body keypoint indices (부모 관계)
coco17_legs = {
    "Left Hip": 11, "Right Hip": 12,
    "Left Knee": 13, "Right Knee": 14, 
    "Left Ankle": 15, "Right Ankle": 16,
}

print("="*80)
print("Cross-Filter 특수 케이스 분석")
print("="*80)

for case_name, filename in cases.items():
    print(f"\n{'='*80}")
    print(f"{case_name}: {filename}")
    print("="*80)
    
    json_path = os.path.join(output_dir, f"{filename}_kp.json")
    
    if not os.path.exists(json_path):
        print(f"❌ 파일 없음: {json_path}")
        continue
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data['people']:
        print("❌ 인물 검출 실패 (people 배열 비어있음)")
        continue
    
    person = data['people'][0]
    pose_kpts = person['pose_keypoints_2d']
    
    # DWPose 키포인트 분석 (총 133개 = 23 body + 68 hand + 42 face)
    # pose_keypoints_2d는 [x, y, score] * 23 = 69개 값
    body_kpts = [(pose_kpts[i], pose_kpts[i+1], pose_kpts[i+2]) 
                 for i in range(0, 69, 3)]
    
    print(f"\n📊 Body Keypoints (23개 중 유효한 것):")
    valid_count = sum(1 for x, y, s in body_kpts if s > 0)
    print(f"   유효 키포인트: {valid_count}/23")
    
    # 다리/발 키포인트 상태 확인
    print(f"\n🦵 다리/발 키포인트 상태:")
    for name, idx in dwpose_legs_feet.items():
        if idx < len(body_kpts):
            x, y, score = body_kpts[idx]
            status = "✅" if score > 0 else "❌"
            if score > 0:
                print(f"   {status} [{idx:2d}] {name:20s}: ({x:7.1f}, {y:7.1f}) score={score:.3f}")
            else:
                print(f"   {status} [{idx:2d}] {name:20s}: REMOVED by Cross-Filter")
    
    # COCO17 상위 관절 확인
    print(f"\n🎯 COCO17 다리 부모 관절 (Body 모델이 검출해야 하는 것):")
    for name, idx in coco17_legs.items():
        if idx < len(body_kpts):
            x, y, score = body_kpts[idx]
            status = "✅ 존재" if score > 0 else "❌ 부재"
            if score > 0:
                print(f"   {status} [{idx:2d}] {name:20s}: ({x:7.1f}, {y:7.1f}) score={score:.3f}")
            else:
                print(f"   {status} [{idx:2d}] {name:20s}: Body 모델이 검출하지 못함")

print("\n" + "="*80)
print("분석 완료")
print("="*80)
