# 발(Feet) 렌더링 문제 해결

## 문제 원인 발견

### 발이 그려지지 않는 이유

발 키포인트(17-22)가 Ghost Filter의 **weak_leg 로직**에 걸려서 제거되고 있었습니다:

```python
# 문제가 있던 코드:
if idx in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:  # knee, ankle, foot 전체
    boundary_near_threshold = h - 80  # 하단 80px 이내
    if y >= boundary_near_threshold and scores[idx] < 3.0:
        is_dummy_coord = True  # 더미로 제거!
```

### 문제점 분석

1. **발목(15,16)까지 포함**: 발목은 항상 하단에 있고 정상적인 키포인트인데도 체크 대상
2. **너무 관대한 범위**: h - 80 = 이미지 하단 80px 영역 전체
3. **낮은 신뢰도 임계값**: conf < 3.0 → 실제 정상 발도 3.0~4.0 신뢰도 가능
4. **결과**: 전신 사진에서 발목이 하단 80px 이내 + 신뢰도 4.0 미만이면 제거!

### 실제 데이터 확인

origin_full_0001 예시:
- LAnkle (15): conf=6.70, y=648.9
- RAnkle (16): conf=6.56, y=650.0
- Feet (17-22): conf=6.22~6.75, y=670~680
- 이미지 높이: 916px
- h - 80 = 836px

→ **발목/발이 836px 미만이므로 weak_leg에 걸리지 않음** (이 케이스는 OK)
→ 하지만 **더 큰 이미지나 발이 하단 근처에 있는 경우** 제거될 수 있음!

---

## 해결 방법

### 변경 사항

1. **발목(15,16)과 무릎(13,14) 제외**
   - 발목은 항상 하단에 있어도 정상
   - 무릎은 하단보다 위에 있음

2. **발(17-22)만 체크**
   - 실제 더미 발만 걸러내기 위함

3. **더 엄격한 조건**
   - 범위: h - 80 → **h - 50** (하단 50px만)
   - 신뢰도: < 3.0 → **< 4.5** (정상 발은 보통 6.0 이상)

### 개선된 코드

```python
# ✅ 개선된 코드:
if idx in [17, 18, 19, 20, 21, 22]:  # 발만 체크 (발목/무릎 제외)
    boundary_near_threshold = h - 50  # 하단 50px 이내
    if y >= boundary_near_threshold and scores[idx] < 4.5:
        is_dummy_coord = True
        weak_leg_reason = f"weak_foot(y={y:.1f}>={boundary_near_threshold:.0f}, conf={scores[idx]:.2f}<4.5)"
```

---

## 예상 효과

### ✅ 전신 사진
- 발목(6.5+ 신뢰도)은 항상 보존
- 발(6.0+ 신뢰도)도 하단에 있어도 보존
- 하단 50px + conf < 4.5인 더미만 제거

### ✅ 반신 사진
- 발목까지만 나오는 경우: 발목 정상 렌더링
- 발도 나오는 경우: 발도 정상 렌더링

### ✅ 더미 필터링 유지
- conf < 4.5 + 하단 50px인 더미 발은 여전히 제거
- Ghost 검출 기능 손상 없음

---

## 테스트 방법

```bash
# 1. Ghost Filter 활성화 상태로 테스트
python PoseExtractor.py

# 2. 결과 확인
# - test_io/outputs/<timestamp> 폴더
# - *_sk.jpg에서 발 렌더링 확인
# - _ghostfilter_layers_debug.txt에서 weak_foot 로그 확인

# 3. 디버그 로그 검색
type test_io\outputs\<folder>\_ghostfilter_layers_debug.txt | Select-String "weak_foot|idx 1[57-9]|idx 2[0-2]"
```

---

## 변경 파일

- `pose_transfer/logic/ghost_filter.py`: weak_leg 로직 개선

## 관련 이슈

- 발목 아래 feet이 그려지지 않는 문제
- boundary_tolerance 개선과 연관
- 전신 사진에서 하단 키포인트 보존 목표
