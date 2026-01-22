# OpenPose + DWPose 교차 필터링 전략 구현 가능성 분석

## 📊 현재 시스템 구조

### 1. 현재 사용 중인 모델
- **DWPose (rtmlib Wholebody)**: COCO-WholeBody 133 keypoints
  - Body(17) + Face(68) + Hands(42) + Feet(6) = 133
  - Top-Down 방식 (YOLO detection → RTMPose estimation)
  - 파일: `pose_transfer/extractors/dwpose_extractor.py`

### 2. OpenPose 상태
- **현재 없음**: OpenPose 관련 코드가 프로젝트에 없음
- 검색 결과: `openpose*.py` 파일 없음
- rtmlib만 사용 중

---

## ✅ 구현 가능성 평가

### **결론: 구현 가능하나 추가 작업 필요**

---

## 🔧 구현 방법

### Option 1: OpenPose 직접 통합 (권장하지 않음)

**필요 작업:**
1. OpenPose Python API 설치 및 빌드
2. OpenPoseExtractor 클래스 구현
3. BODY_25 포맷 처리 로직

**문제점:**
- OpenPose 빌드 복잡 (CMake, CUDA, cuDNN)
- 무거운 모델 (VGG-19 백본)
- 추론 속도 느림 (100-200ms)
- 라이선스 이슈 (OpenPose는 학술용)

### Option 2: rtmlib의 Body 모델 활용 (강력 추천)

**핵심 아이디어:**
rtmlib에는 **Wholebody** 외에 **Body** 모델도 있습니다!
- `rtmlib.Body`: COCO 17 keypoints만 추출 (OpenPose와 유사)
- Bottom-Up 스타일로 작동 가능
- 경량화되어 있어 빠름

**구현 전략:**
```python
from rtmlib import Wholebody, Body

# 1. Body 모델로 "존재 유무" 판단
body_model = Body(...)
body_kpts, body_scores = body_model(image)

# 2. Wholebody 모델로 "정밀한 좌표" 추출
wholebody_model = Wholebody(...)
dw_kpts, dw_scores = wholebody_model(image)

# 3. 교차 필터링
filtered = cross_filter(body_kpts, body_scores, dw_kpts, dw_scores)
```

**장점:**
- 동일한 라이브러리 (rtmlib) 사용
- 추가 빌드 불필요
- 빠른 추론 속도
- COCO 17 keypoints를 COCO-WholeBody 133으로 쉽게 매핑 가능

### Option 3: MediaPipe Pose 활용 (차선책)

**MediaPipe**는 이미 설치되어 있을 가능성이 높습니다.

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# MediaPipe로 존재 유무 판단
results = pose.process(image)
if results.pose_landmarks:
    # 존재함
```

**장점:**
- 매우 빠름 (10-30ms)
- 설치 간단
- 절단 이미지에서 비교적 보수적

**단점:**
- 33 keypoints (매핑 복잡)
- 1인 전용

---

## 📝 구현 계획

### Phase 1: rtmlib Body 모델 추가 (1-2시간)

**1.1 새로운 Extractor 생성**
```
pose_transfer/extractors/body_extractor.py  # 신규
```

**1.2 매핑 테이블 구축**
```
pose_transfer/extractors/keypoint_mapping.py  # 신규
```

**1.3 Cross Filter 로직**
```
pose_transfer/logic/cross_filter.py  # 신규
```

### Phase 2: 파이프라인 통합 (2-3시간)

**2.1 PipelineConfig 수정**
```yaml
# default.yaml
cross_filter:
  enabled: true
  use_body_model: true  # rtmlib Body 사용
  body_confidence_threshold: 0.2
  spine_interpolation: true
```

**2.2 Pipeline 수정**
```python
# pipeline.py
if self.config.cross_filter_enabled:
    # Body 모델로 마스크 생성
    body_result = self.body_extractor.extract(src_img)
    
    # DWPose로 정밀 좌표 생성
    dw_result = self.dwpose_extractor.extract(src_img)
    
    # 교차 필터링
    filtered_result = cross_filter(body_result, dw_result)
```

### Phase 3: 척추 보간 (1시간)

**3.1 Spine Generator**
```python
# pose_transfer/logic/spine_generator.py
def generate_spine(keypoints, scores):
    neck = (kpts[5] + kpts[6]) / 2  # 어깨 중점
    mid_hip = (kpts[11] + kpts[12]) / 2  # 골반 중점
    spine = (neck + mid_hip) / 2  # 척추
    return neck, spine, mid_hip
```

**3.2 Renderer 수정**
- 척추 라인 그리기 추가

---

## 🎯 매핑 테이블 (rtmlib Body → Wholebody)

| Body Part | Body (17) | Wholebody (133) |
|-----------|-----------|-----------------|
| Nose | 0 | 0 |
| L-Eye | 1 | 1 |
| R-Eye | 2 | 2 |
| L-Ear | 3 | 3 |
| R-Ear | 4 | 4 |
| L-Shoulder | 5 | 5 |
| R-Shoulder | 6 | 6 |
| L-Elbow | 7 | 7 |
| R-Elbow | 8 | 8 |
| L-Wrist | 9 | 9 |
| R-Wrist | 10 | 10 |
| L-Hip | 11 | 11 |
| R-Hip | 12 | 12 |
| L-Knee | 13 | 13 |
| R-Knee | 14 | 14 |
| L-Ankle | 15 | 15 |
| R-Ankle | 16 | 16 |

**완벽한 1:1 매핑!**

---

## ⚡ 성능 영향 분석

### 현재 (DWPose만)
- 추론 시간: 약 50-80ms/frame

### 교차 필터링 적용 후
- Body 추론: 약 20-30ms
- Wholebody 추론: 약 50-80ms
- 필터링: 약 1-2ms
- **총합: 약 70-110ms** (약 30-40% 증가)

### 최적화 방안
1. **병렬 실행**: Body와 Wholebody를 동시에 추론 (멀티스레드)
   - 추론 시간: max(30ms, 80ms) = 80ms (거의 증가 없음!)

2. **조건부 실행**: Wholebody가 절단 의심 시에만 Body 실행
   - 전신 사진: Wholebody만 (80ms)
   - 반신 사진: Body + Wholebody (110ms)

---

## 🚀 즉시 실행 가능 여부

### Ghost Filter 비활성화
```yaml
# default.yaml
ghost_filter:
  enabled: false  # ✅ 즉시 적용 가능
```

### 교차 필터링 추가
**불가능 (현재 상태)**
- Body 모델 extractor 없음
- Cross filter 로직 없음
- 매핑 테이블 없음

**필요 작업:**
1. `body_extractor.py` 생성 (30분)
2. `cross_filter.py` 생성 (1시간)
3. `keypoint_mapping.py` 생성 (30분)
4. Pipeline 통합 (1시간)
5. 테스트 및 디버깅 (2시간)

**총 예상 시간: 5-6시간**

---

## 💡 권장 사항

### 단기 해결책 (즉시 적용)
1. Ghost Filter 비활성화
2. 발 전용 기준 적용 (이미 완료)
3. Confidence threshold 조정
   ```yaml
   kpt_threshold: 0.5  # 0.3 → 0.5 (ControlNet 스타일)
   ```

### 중기 해결책 (5-6시간 작업)
1. rtmlib Body 모델 통합
2. 교차 필터링 구현
3. 척추 보간 추가

### 장기 해결책 (미래)
1. 실제 OpenPose 통합 (성능이 중요하지 않은 경우)
2. 커스텀 모델 학습 (절단 이미지 특화)

---

## 📌 결론

**구현 가능성: ✅ 가능**

**추천 방법: rtmlib Body 모델 활용**

**작업 시간: 5-6시간**

**즉시 적용: ❌ 불가 (추가 개발 필요)**

Ghost Filter false 설정은 즉시 가능하나,
교차 필터링은 새로운 컴포넌트 구현이 필요합니다.

구현을 원하시면 단계별로 진행 가능합니다.
