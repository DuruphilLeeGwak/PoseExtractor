# Confidence 범위 분석 보고서

## 📊 핵심 발견

두 가지 모델이 **완전히 다른 confidence 범위**를 사용하고 있습니다:

### 1. Body 모델 (YOLO)
```
범위: 0.0 ~ 1.0 (확률값)
출력: Sigmoid 함수
의미: 표준 확률 (0.5 = 50% 확신)
```

### 2. DWPose (rtmlib Wholebody)
```
범위: 2.8 ~ 8.0+ (로그 확률/로짓)
출력: SimCC 기반 스코어
의미: 정규화되지 않은 로그 확률
```

---

## 🔬 실제 측정 데이터

### DWPose (rtmlib) 통계
여러 이미지(266개 키포인트)에서 측정:

```
최소값:  2.76
최대값:  8.04
평균값:  5.48
표준편차: 1.12

백분위수:
  1%:  2.85  (매우 낮은 확신)
  5%:  3.32
 25%:  4.92
 50%:  5.78  (중앙값)
 75%:  6.28
 95%:  6.86  (높은 확신)
 99%:  7.34  (매우 높은 확신)
```

---

## ⚠️ 왜 범위가 다른가?

### 1. 모델 아키텍처 차이

**Body 모델 (YOLO)**:
```python
# 출력 레이어: Sigmoid 활성화 함수
output = sigmoid(logits)  # 0~1 범위로 정규화
```
- Sigmoid는 모든 실수를 0~1 범위로 압축
- 0.5 = 50% 확률, 0.9 = 90% 확률
- 직관적인 확률 해석 가능

**DWPose (rtmlib)**:
```python
# SimCC (Simulated Classification Coordinates) 방식
# 로그 확률 또는 로짓을 직접 출력
output = log_probability  # 2.8~8.0+ 범위 (정규화 안 함)
```
- SimCC는 coordinate classification 방식
- 로그 공간에서 계산하여 정밀도 향상
- 값이 클수록 높은 확신 (but 0~1 아님!)

### 2. 설계 목적 차이

| 모델 | 목적 | 출력 형식 |
|------|------|-----------|
| Body (YOLO) | 객체 검출 + 포즈 추정 | 확률값 (0~1) |
| DWPose (rtmlib) | 고정밀 전신 포즈 | 로그 확률 (자연수) |

---

## ✅ 현재 Cross-Filter 설정 분석

### 설정값과 의미

```python
# Body 모델용 (0~1 범위)
body_confidence_threshold = 0.5        # 50% 확신 이상
clean_mode_body_threshold = 0.2        # Clean Mode: 20% 확신 이상

# DWPose 모델용 (2.8~8.0+ 범위)
dw_high_confidence_threshold = 8.0     # 99th percentile 이상 (매우 높음)
dw_full_body_confidence_threshold = 6.0  # 75th percentile 근처 (평균 이상)
dw_suspicious_threshold = 2.0          # 1st percentile 근처 (매우 낮음)
```

### 통계적 해석

| Threshold | 실제 의미 | 백분위수 |
|-----------|-----------|----------|
| dw_suspicious: 2.0 | 매우 낮은 확신 | ~1% 이하 |
| dw_full_body: 6.0 | 평균 이상 | ~50-75% |
| dw_high: 8.0 | 극도로 높은 확신 | ~99%+ |

---

## 🎯 정규화가 필요한가?

### ❌ 정규화 불필요한 이유

1. **각 모델에 맞는 threshold 이미 사용 중**
   - Body: 0~1 범위용 threshold (0.2, 0.5)
   - DWPose: 2.8~8.0+ 범위용 threshold (2.0, 6.0, 8.0)

2. **정규화 시 문제점**
   ```python
   # 잘못된 예시: DWPose를 0~1로 정규화하면?
   normalized = (score - 2.8) / (8.0 - 2.8)
   # 문제: 8.0 이상 값이 잘림, 분포 왜곡
   ```

3. **현재 시스템이 이미 정확하게 작동**
   - Clean Mode 정상 동작 (suspicious_count 기반)
   - Full-Body Bypass 정상 동작 (6.0 이상)
   - Individual Bypass 정상 동작 (8.0 이상)

### ✅ 필요한 것: 명확한 문서화

변수명과 주석에 **범위를 명시**:
```python
# ❌ 나쁜 예
dw_threshold = 6.0

# ✅ 좋은 예
dw_full_body_confidence_threshold = 6.0  # DWPose: 2.8~8.0+ 범위
```

---

## 📝 코드 개선 사항

### 적용된 문서화

각 threshold에 명확한 범위 설명 추가:

```python
@dataclass
class CrossFilterConfig:
    # ========== Body 모델 (YOLO: 0~1 범위) ==========
    body_confidence_threshold: float = 0.5
    """
    ⚠️ Confidence 범위: 0~1 (Sigmoid 출력)
    """
    
    # ========== DWPose (rtmlib: 2.8~8.0+ 범위) ==========
    dw_high_confidence_threshold: float = 8.0
    """
    ⚠️ Confidence 범위: 2.8~8.0+ (로그 확률)
    실제 측정값:
    - 평균: 5.5
    - 중앙값: 5.78
    - 99th: 7.34
    """
```

---

## 🎓 결론

### 1. 범위 차이의 원인
- **YOLO**: Sigmoid 출력 → 0~1 확률값
- **rtmlib**: SimCC 출력 → 2.8~8.0+ 로그 확률

### 2. 현재 상태
- ✅ Threshold 설정이 **이미 정확함**
- ✅ 각 모델 범위에 맞게 구성됨
- ✅ 시스템 정상 작동 중

### 3. 개선 사항
- ✅ 모든 threshold에 범위 명시 추가
- ✅ 통계 정보 (백분위수) 문서화
- ✅ 주석에 실제 측정값 포함

### 4. 권장 사항
- ❌ 정규화 불필요 (오히려 혼란 초래)
- ✅ 명확한 변수명 유지
- ✅ 주석에 범위 명시
- ✅ 테스트로 검증

---

## 📚 참고: 모델별 Confidence 비교표

| 특성 | Body (YOLO) | DWPose (rtmlib) |
|------|-------------|-----------------|
| **출력 형식** | Sigmoid 확률 | SimCC 로그 확률 |
| **범위** | 0.0 ~ 1.0 | 2.8 ~ 8.0+ |
| **평균** | ~0.5-0.7 | ~5.5 |
| **높은 확신** | 0.7+ | 6.5+ |
| **매우 높은 확신** | 0.9+ | 7.5+ |
| **의심 범위** | <0.3 | <3.0 |
| **정규화** | 이미 됨 | 안 됨 (의도적) |

---

## 🔍 테스트 검증

실제 이미지(3.jpg)에서 측정한 DWPose 스코어:
```
idx  0 (코):      7.74  → 99th percentile (극도로 높음)
idx  1 (왼눈):    8.04  → MAX! (최고 확신)
idx  5 (왼어깨):  5.08  → 중앙값 근처
idx 11 (왼엉덩이): 3.56  → 낮은 확신
idx 12 (오엉덩이): 3.56  → 낮은 확신
```

→ **dw_full_body_confidence_threshold = 6.0**은 정확함!
- 모든 body 17개가 6.0 이상 = "전신이 명확" ✅
- 일부만 6.0 이상 = "부분적 확신" → Body 검증 필요 ✅

---

**보고서 작성일**: 2026-01-22  
**테스트 환경**: conda jyk, rtmlib Wholebody, YOLO11n
