# DWPose vs MediaPipe Rendering Analysis

## 키포인트 형식 비교

### MediaPipe Pose (33 keypoints)
```
0: nose
1: left_eye_inner
2: left_eye
3: left_eye_outer
4: right_eye_inner
5: right_eye
6: right_eye_outer
7: left_ear
8: right_ear
9: mouth_left
10: mouth_right
11: left_shoulder
12: right_shoulder
13: left_elbow
14: right_elbow
15: left_wrist
16: right_wrist
17: left_pinky
18: right_pinky
19: left_index
20: right_index
21: left_thumb
22: right_thumb
23: left_hip
24: right_hip
25: left_knee
26: right_knee
27: left_ankle
28: right_ankle
29: left_heel
30: right_heel
31: left_foot_index
32: right_foot_index
```

### COCO-WholeBody (133 keypoints) - DWPose 사용
```
Body (0-16): 17개
  0: nose
  1-4: eyes, ears
  5-6: shoulders
  7-8: elbows
  9-10: wrists
  11-12: hips
  13-14: knees
  15-16: ankles

Feet (17-22): 6개
  17-19: left foot (big_toe, small_toe, heel)
  20-22: right foot

Face (23-90): 68개
  얼굴 랜드마크

Hands (91-132): 42개 (각 손 21개)
```

## 연결선 패턴 비교

### MediaPipe 연결 방식
```python
# MediaPipe POSE_CONNECTIONS
CONNECTIONS = [
    # 얼굴
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    
    # 상체 - 중심선 강조
    (11, 12),  # 어깨선
    (11, 23), (12, 24),  # 어깨-골반
    (23, 24),  # 골반선
    
    # 팔
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    
    # 다리
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]
```

### 현재 코드 (COCO-WholeBody 기반)
```python
BODY_BONES = [
    # 어깨선
    ('left_shoulder', 'right_shoulder'),
    
    # 몸통 (어깨-골반)
    ('left_shoulder', 'left_hip'), 
    ('right_shoulder', 'right_hip'),
    
    # 골반선
    ('left_hip', 'right_hip'),
    
    # 팔
    ('left_shoulder', 'left_elbow'), 
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), 
    ('right_elbow', 'right_wrist'),
    
    # 다리
    ('left_hip', 'left_knee'), 
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), 
    ('right_knee', 'right_ankle'),
    
    # 얼굴-목
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'left_ear'), ('right_shoulder', 'right_ear'),
]
```

## 평가: DWPose가 MediaPipe 방식으로 렌더링?

### ❌ **아니오, 다릅니다**

1. **키포인트 개수 차이**
   - MediaPipe: 33개
   - DWPose (COCO-WholeBody): 133개
   - 완전히 다른 체계!

2. **목(Neck) 처리**
   - MediaPipe: Neck 키포인트 없음 (shoulder-hip 직접 연결)
   - OpenPose: Neck 키포인트 있음 (1번)
   - COCO-WholeBody: Neck 없음 → **가상 neck 필요**

3. **손 표현**
   - MediaPipe: 손가락 끝 5개만 (pinky, index, thumb)
   - COCO-WholeBody: 손가락 전체 21개 관절

### ✅ **유사점**

1. **몸통 연결 패턴은 유사**
   ```
   shoulder-hip 직접 연결
   좌우 대칭 구조
   ```

2. **발 표현**
   - MediaPipe: heel, foot_index (2개)
   - COCO-WholeBody: heel, big_toe, small_toe (3개)
   - 유사한 개념

3. **중심축 부재**
   - MediaPipe: nose-neck 중심축 없음
   - COCO-WholeBody: neck 키포인트 없음
   - **둘 다 가상 neck 필요!**

## 결론

### 사용자가 본 것은 아마도:

1. **ControlNet DWPose Annotator**
   - ControlNet의 DWPose는 렌더링 시 MediaPipe **스타일**을 참고할 수 있음
   - 하지만 키포인트는 COCO-WholeBody 133개 사용

2. **렌더링 스타일 유사성**
   - 깔끔한 선
   - 색상 코딩 (body, hand, face)
   - 최소주의 디자인
   
3. **가상 Neck 사용**
   - MediaPipe: 원래 neck 없음
   - DWPose (COCO): neck 없음
   - **둘 다 어깨 중점을 neck으로 사용**하는 방식 채택 가능

### 현재 코드 상태:

✅ **개선 완료**: 가상 neck 추가로 MediaPipe/표준 OpenPose와 유사한 중심축 확보
✅ **COCO-WholeBody 기반**: DWPose의 표준 출력 형식
✅ **133 키포인트**: 더 상세한 손/얼굴 표현

### 차이점은 "데이터 형식"이 아니라 "렌더링 스타일":

| 항목 | MediaPipe | DWPose (현재 코드) |
|------|-----------|-------------------|
| 키포인트 개수 | 33 | 133 |
| 손 상세도 | 낮음 (5개) | 높음 (21개) |
| 얼굴 상세도 | 낮음 | 높음 (68개) |
| Neck | 없음 | 없음 (가상 생성) |
| 몸통 연결 | shoulder-hip | shoulder-hip |
| 렌더링 스타일 | 깔끔 | 유사 (개선 후) |

**평가: DWPose는 MediaPipe 형식이 아니라 COCO-WholeBody 형식이지만, 렌더링 스타일에서 유사성이 있을 수 있습니다.**
