import json
import numpy as np

# JSON 로드
with open('test_io/outputs/20260123_162439/14_kp.json') as f:
    data = json.load(f)
    
# 'people' 배열에서 첫 번째 사람 추출
person = data['people'][0]
kpts = np.array(person['keypoints']).reshape(-1, 2)
scores = np.array(person['scores'])

# Src, Ref는 없으므로 Trans 데이터만 분석
trans_kpts = kpts

def dist(a, b):
    return np.linalg.norm(a - b)

# Trans 발목-발가락 거리만 확인
trans_ankle_to_toe = dist(trans_kpts[16], trans_kpts[20])
print(f"Trans 발목-발가락 거리: {trans_ankle_to_toe:.1f}px")
print(f"Trans R-Ankle (16): {trans_kpts[16]}")
print(f"Trans R-BigToe (20): {trans_kpts[20]}")
print(f"\n이 파일은 이미 전이된 결과이므로 Src/Ref 비교가 불가능합니다.")
print("debug.txt에서 스케일 정보를 확인해야 합니다.")
