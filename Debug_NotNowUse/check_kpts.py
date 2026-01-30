import json

# JSON 파일 읽기
with open('test_io/outputs/20260120_165846/half_bg_5_kp.json') as f:
    data = json.load(f)

kpts = data['people'][0]['pose_keypoints_2d']

# 이미지 크기는 로그에서 확인 필요, 일단 추정
# half_bg_5는 반신 이미지일 가능성 → 약 2000x3000 정도로 추정

print("===골반/다리/발 키포인트 (11-22)===\n")
for idx in range(11, 23):
    x, y, conf = kpts[idx*3], kpts[idx*3+1], kpts[idx*3+2]
    names = ["LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", 
             "LFoot1", "LFoot2", "LFoot3", "RFoot1", "RFoot2", "RFoot3"]
    print(f"idx{idx:2d}({names[idx-11]:7s}): x={x:7.1f}, y={y:7.1f}, conf={conf:.2f}")

print("\n===왼손 (91-111)===\n")
for idx in [91, 95, 99, 103, 107, 111]:  # 샘플
    x, y, conf = kpts[idx*3], kpts[idx*3+1], kpts[idx*3+2]
    print(f"idx{idx}: x={x:7.1f}, y={y:7.1f}, conf={conf:.2f}")
