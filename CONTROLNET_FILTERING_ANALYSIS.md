# ControlNet DWPose Preprocessor 분석

## ControlNet의 필터링 메커니즘

### 1. **Confidence Threshold (매우 높음)**
ControlNet DWPose preprocessor:
```python
# ControlNet의 일반적인 설정
kpt_threshold = 0.5  # 또는 더 높음 (0.6~0.7)
```

**현재 코드:**
```python
kpt_threshold = 0.3  # 더 낮음 → 더 많은 키포인트 표시
```

→ **ControlNet은 신뢰도 0.5 미만 키포인트를 아예 그리지 않음!**

---

### 2. **프레임 외부 키포인트 완전 제거**

**ControlNet 방식:**
```python
# 렌더링 전에 프레임 밖 키포인트를 완전히 제거
if x < 0 or x >= width or y < 0 or y >= height:
    score = 0.0  # 신뢰도를 0으로 설정
    # 렌더링하지 않음
```

**현재 코드:**
```python
# Ghost Filter가 out_of_frame_indices에 마킹
# 하지만 score는 유지됨
if idx in out_of_frame_indices:
    continue  # 렌더링만 스킵
```

→ **ControlNet은 프레임 밖 키포인트의 score를 0으로 만듦!**

---

### 3. **더미 키포인트 감지 및 제거**

**ControlNet의 접근:**
```python
# DWPose가 감지 실패 시 생성하는 더미 패턴 제거:
# 1. 경계값 (0, 0), (w-1, h-1) 등
# 2. 비정상적으로 낮은 confidence
# 3. 해부학적으로 불가능한 위치

if is_dummy_coordinate(x, y, confidence):
    score = 0.0  # 완전 제거
```

**현재 코드:**
```python
# Ghost Filter가 더미 감지하지만
# - boundary_tolerance로 일부만 감지
# - 일부 더미는 여전히 통과 가능
```

---

### 4. **연결선(Bone) 렌더링 규칙**

**ControlNet:**
```python
# 양쪽 키포인트 모두 높은 신뢰도여야 선 그리기
if (scores[start] > 0.5 and scores[end] > 0.5 and
    is_within_frame(start) and is_within_frame(end)):
    draw_line(start, end)
```

**현재 코드:**
```python
# threshold 0.3 + out_of_frame 체크
if (scores[start] > 0.3 and scores[end] > 0.3 and
    start not in out_of_frame and end not in out_of_frame):
    draw_line(start, end)
```

---

## 주요 차이점 요약

| 항목 | ControlNet | 현재 코드 |
|------|-----------|----------|
| **Confidence Threshold** | 0.5~0.7 (높음) | 0.3 (낮음) |
| **프레임 밖 처리** | score=0 완전제거 | out_of_frame 마킹만 |
| **더미 감지** | 적극적 (경계/비정상) | 보수적 (일부만) |
| **렌더링 철학** | "확실한 것만 그림" | "있을 수 있는 것 그림" |

---

## 왜 ControlNet이 더 깔끔한가?

### 1. **높은 Threshold**
```python
0.3: "이 키포인트가 있을 수 있어" (30% 확신)
0.5: "이 키포인트가 아마 있어" (50% 확신)
0.7: "이 키포인트가 확실히 있어" (70% 확신)
```

→ ControlNet은 **"확실한 것만 그림"** 철학

### 2. **프레임 밖 = 존재하지 않음**
```python
# ControlNet
if out_of_frame:
    키포인트 자체를 삭제 (score=0)
    
# 현재 코드
if out_of_frame:
    렌더링만 안 함 (score 유지)
    → Ghost Filter에서 계속 처리됨
```

### 3. **더미 적극 제거**
```python
# ControlNet: 경계 ±5px 내 키포인트 모두 제거
boundary_tolerance = 5.0

# 현재 코드: 경계 ±10px (더 관대)
boundary_tolerance = 10.0
```

---

## 개선 방향

### Option 1: ControlNet 스타일 (추천)

**장점:**
- 깔끔한 출력
- 더미 최소화
- SD 입력으로 적합

**단점:**
- 실제 존재하는 일부 키포인트 손실 가능

```python
# 설정 변경
kpt_threshold = 0.5  # 0.3 → 0.5
boundary_tolerance = 5.0  # 10.0 → 5.0

# 프레임 밖 키포인트 완전 제거
if out_of_frame:
    filtered_scores[idx] = 0.0  # score를 0으로
```

### Option 2: 현재 방식 유지 (세밀)

**장점:**
- 더 많은 정보 보존
- 폐색/부분 가림도 표현

**단점:**
- 일부 더미 포함 가능
- 덜 깔끔

---

## 결론

### ✅ **사용자 관찰이 정확합니다**

ControlNet은 자체적인 강력한 필터링을 가지고 있습니다:

1. **높은 confidence threshold (0.5+)**
2. **프레임 밖 키포인트 완전 제거**
3. **더미 패턴 적극 감지**
4. **"확실한 것만 그림" 철학**

### 현재 코드의 접근:

- Ghost Filter로 필터링하지만 **더 보수적**
- threshold 0.3으로 **더 많은 키포인트 보존**
- 프레임 밖 키포인트를 **마킹만** 하고 제거하지 않음

### 추천:

**ControlNet과 같은 출력을 원한다면:**
```yaml
# default.yaml
ghost_filter:
  confidence_threshold: 0.5  # 0.1 → 0.5
  boundary_tolerance: 5.0     # 10.0 → 5.0
  
# skeleton_renderer
kpt_threshold: 0.5  # 0.3 → 0.5
```

**세밀한 정보 보존을 원한다면:**
- 현재 설정 유지
- Ghost Filter 신뢰
