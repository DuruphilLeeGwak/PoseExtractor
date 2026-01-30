# Skeleton Rendering 차이점 분석

## 현재 코드 vs 표준 DWPose/OpenPose

### 발견된 주요 차이점

#### 1. **목(Neck) 연결이 없음** ⚠️

**표준 DWPose/OpenPose:**
```python
# 중심축 (목-어깨 연결)
('nose', 'neck'),           # 코 -> 목
('neck', 'left_shoulder'),  # 목 -> 왼어깨
('neck', 'right_shoulder'), # 목 -> 오른어깨
```

**현재 코드:**
```python
# 목 연결 없음! 대신:
('left_shoulder', 'right_shoulder'),  # 어깨선만
('left_shoulder', 'left_ear'),        # 어깨 -> 귀
('right_shoulder', 'right_ear'),      # 어깨 -> 귀
```

→ **문제**: 코와 어깨 사이 중심축(목)이 없어서 상체 중심선이 없음!

---

#### 2. **코-어깨 직접 연결 없음**

**표준 DWPose:**
- `nose` → `neck` → `shoulder` 2단계 연결

**현재 코드:**
- `nose` → `eye` → `ear` → `shoulder` 3단계 우회
- 중심축 없음

---

#### 3. **COCO-WholeBody에는 Neck 키포인트가 없음**

COCO-WholeBody 133 키포인트:
- 0: nose
- 1-4: eyes, ears
- 5-6: shoulders
- **목(neck) 키포인트 없음!**

표준 OpenPose-18은 목(1번) 키포인트를 가지고 있지만, COCO-WholeBody는 없습니다.

---

## 해결 방법

### 옵션 1: 목 키포인트 가상 생성 (추천)

양쪽 어깨의 중점을 "가상 목"으로 사용:

```python
# 가상 neck 생성
if shoulder_left_valid and shoulder_right_valid:
    neck_pos = (keypoints[5] + keypoints[6]) / 2  # 어깨 중점
    neck_score = min(scores[5], scores[6])
    
    # 목 연결 그리기
    if nose_valid and neck_score > threshold:
        draw_line(nose, neck_pos)  # 코 -> 목
    if neck_score > threshold:
        draw_line(neck_pos, shoulder_left)   # 목 -> 왼어깨
        draw_line(neck_pos, shoulder_right)  # 목 -> 오른어깨
```

### 옵션 2: 코-어깨 직접 연결

```python
BODY_BONES = [
    # 중심축 추가
    ('nose', 'left_shoulder'),
    ('nose', 'right_shoulder'),
    
    # 기존 연결 유지
    ('left_shoulder', 'right_shoulder'),
    ...
]
```

→ **단점**: 코가 앞으로 나와있어서 어깨까지 직선으로 연결하면 부자연스러움

### 옵션 3: 어깨 중점 연결 (간단)

```python
# nose와 어깨선 중점을 연결
shoulder_midpoint = (left_shoulder + right_shoulder) / 2
draw_line(nose, shoulder_midpoint)
```

---

## 표준 OpenPose vs COCO-WholeBody 매핑

| OpenPose-18 | COCO-WholeBody | 설명 |
|-------------|----------------|------|
| 0: Nose | 0: Nose | 코 |
| 1: **Neck** | **없음** | **목 키포인트 없음!** |
| 2: RShoulder | 6: right_shoulder | 오른어깨 |
| 3: RElbow | 8: right_elbow | 오른팔꿈치 |
| ... | ... | ... |

---

## 권장 수정

### BODY_BONES에 중심축 추가:

```python
BODY_BONES = [
    # === 중심축 (가상 neck 사용) ===
    # renderer에서 동적으로 neck 위치 계산하여 그리기
    
    # 어깨선
    ('left_shoulder', 'right_shoulder'),
    
    # 몸통
    ('left_shoulder', 'left_hip'), 
    ('right_shoulder', 'right_hip'),
    
    # 골반선
    ('left_hip', 'right_hip'),
    
    # ... 나머지 동일
]
```

### Renderer 수정:

```python
def _draw_body_with_neck(self, canvas, keypoints, scores, ...):
    # 1. 기존 body bones 그리기
    self._draw_bones(...)
    
    # 2. 가상 neck 그리기
    l_shoulder_idx = 5
    r_shoulder_idx = 6
    nose_idx = 0
    
    if (scores[l_shoulder_idx] > threshold and 
        scores[r_shoulder_idx] > threshold):
        
        # 가상 목 위치 = 어깨 중점
        neck_pos = (keypoints[l_shoulder_idx] + keypoints[r_shoulder_idx]) / 2
        neck_score = min(scores[l_shoulder_idx], scores[r_shoulder_idx])
        
        # 코 -> 목
        if scores[nose_idx] > threshold and neck_score > threshold:
            cv2.line(canvas, 
                    tuple(keypoints[nose_idx].astype(int)),
                    tuple(neck_pos.astype(int)),
                    color, thickness)
        
        # 목 -> 어깨들 (이미 shoulder line으로 그려짐, 추가 불필요)
```

---

## 결론

**차이점 요약:**
1. ❌ **목(neck) 연결 없음** - 상체 중심축 누락
2. ✅ 팔/다리/발 연결은 정상
3. ✅ 얼굴/손 연결은 정상

**원인:**
- COCO-WholeBody에는 neck 키포인트가 없음
- 현재 코드는 neck 없이 어깨선만 그림

**해결:**
- 어깨 중점을 가상 neck으로 사용
- nose → virtual_neck → shoulders 연결 추가
