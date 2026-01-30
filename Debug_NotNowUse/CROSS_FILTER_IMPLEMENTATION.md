# Cross-Filter 구현 완료 (v4.0)

## 📋 구현 개요

**핵심 전략**: "몸통이 보증하면, 디테일은 DWPose 것을 쓴다"

### 역할 분리
- **Body 모델** (감시자): "여기에 팔이 있는가?" (존재 여부만 판단)
- **DWPose 모델** (아티스트): "손가락 마디는 어디에?" (정밀 묘사 담당)

### 문제 상황
DWPose Wholebody는 Top-Down 방식으로 133개 키포인트를 **강제로** 예측합니다.
- 실제로 팔이 화면 밖에 있어도 → 화면 가장자리에 손가락 21개를 환각으로 찍음
- 실제로 발이 잘려도 → 이미지 하단에 발가락 3개를 환각으로 찍음

### 해결 방법
rtmlib Body 모델 (COCO 17 keypoints)로 "존재 여부"를 먼저 검증:
1. Body 모델이 "손목 신뢰도 > 0.3" → 손목 진짜 존재
2. Body가 승인한 손목에 대해서만 → DWPose 손가락 21개 사용
3. Body가 승인 안 하면 → DWPose 손가락 전부 제거 (환각 방지)

---

## 🏗️ 구현된 구조

### 1. BodyExtractor (pose_transfer/extractors/body_extractor.py)
```python
class BodyExtractor:
    """rtmlib Body 모델 래퍼 (COCO 17 keypoints)"""
    
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            keypoints: (N, 17, 2) - COCO 17 keypoints
            scores: (N, 17) - 신뢰도
        """
```

**특징**:
- COCO 17 keypoints (Nose, Neck, Shoulders, Elbows, Wrists, Hips, Knees, Ankles)
- Bottom-Up 스타일로 보수적 감지 (없는 것을 환각으로 찍지 않음)
- DWPose Wholebody의 인덱스 0-16과 **완벽한 1:1 매핑**

### 2. CrossFilter (pose_transfer/logic/cross_filter.py)
```python
class CrossFilter:
    """부모-자식 종속 필터링"""
    
    def filter(
        self,
        body_keypoints: np.ndarray,    # (17, 2)
        body_scores: np.ndarray,       # (17,)
        dw_keypoints: np.ndarray,      # (133, 2)
        dw_scores: np.ndarray          # (133,)
    ) -> Tuple[np.ndarray, Set[int]]:
        """
        Returns:
            filtered_keypoints: (133, 2) - 승인된 keypoint만 유지
            approved_indices: Set[int] - 승인된 인덱스 집합
        """
```

**4단계 필터링 알고리즘**:
1. **Stage 1**: Body 17 keypoints 검증 (신뢰도 > 0.3)
2. **Stage 2**: 손 종속 필터링
   - 손목(9/10) 승인 → 손가락(91-132) 전부 승인
   - 손목 없음 → 손가락 전부 제거
3. **Stage 3**: 발 종속 필터링
   - 발목(15/16) 승인 → 발가락(17-22) 전부 승인
   - 발목 없음 → 발가락 전부 제거
4. **Stage 4**: 얼굴 종속 필터링
   - 코(0) OR 목(1) 승인 → 얼굴(23-90) 전부 승인
   - 둘 다 없음 → 얼굴 전부 제거

---

## ⚙️ 설정 (default.yaml)

```yaml
cross_filter:
  enabled: false                   # true로 변경하면 Cross-Filter 활성화
  
  # Body 모델 신뢰도 임계값 (존재 판단)
  body_confidence_threshold: 0.3   # 0.2~0.4 권장
  
  # 부모-자식 종속 규칙
  enable_hand_dependency: true     # 손목 없으면 손가락도 없음
  enable_foot_dependency: true     # 발목 없으면 발가락도 없음
  enable_face_dependency: true     # 코/목 없으면 얼굴도 없음
  
  # DWPose 자체 신뢰도 안전 체크
  dw_min_confidence: 0.05          # 0.05~0.1 권장
```

**주의**: `ghost_filter.enabled`와 `cross_filter.enabled`는 **배타적**입니다.
- Ghost Filter: 통계 기반 환각 감지 (기존 방식)
- Cross Filter: 부모-자식 종속 필터링 (신규 방식)

---

## 🔄 파이프라인 통합

### Pipeline 초기화 (_init_modules)
```python
# Body extractor 초기화
if self.config.cross_filter_enabled:
    self.body_extractor = BodyExtractor(
        mode='balanced',
        backend='onnxruntime',
        device='cpu'
    )
    
# CrossFilter 초기화
if self.config.cross_filter_enabled:
    self.cross_filter = CrossFilter(
        config=CrossFilterConfig(
            body_confidence_threshold=0.3,
            enable_hand_dependency=True,
            enable_foot_dependency=True,
            enable_face_dependency=True,
            dw_min_confidence=0.05
        )
    )
```

### 포즈 추출 (extract_pose)
```python
# DWPose Wholebody 추출
all_kpts, all_scores = self.extractor.extract(img)

# Cross-Filter 적용
if self.config.cross_filter_enabled:
    # Body 17 keypoints 추출
    body_kpts, body_scores = self.body_extractor.extract(img)
    
    # CrossFilter로 필터링
    filtered_kpts, approved_indices = self.cross_filter.filter(
        body_keypoints=body_kpts[0],
        body_scores=body_scores[0],
        dw_keypoints=kpts,
        dw_scores=scores
    )
    
    # 승인된 keypoint만 유지
    kpts = filtered_kpts
    scores[~approved_indices] = 0  # 승인 안 된 건 신뢰도 0
```

---

## 📊 Index Mapping (Body ↔ Wholebody)

| Body Index | Wholebody Index | Keypoint Name |
|------------|-----------------|---------------|
| 0 | 0 | Nose |
| 1 | 1 | Neck (virtual) |
| 2 | 2 | RShoulder |
| 3 | 3 | RElbow |
| 4 | 4 | RWrist |
| 5 | 5 | LShoulder |
| 6 | 6 | LElbow |
| 7 | 7 | LWrist |
| 8 | 8 | MidHip |
| 9 | 9 | RHip |
| 10 | 10 | RKnee |
| 11 | 11 | RAnkle |
| 12 | 12 | LHip |
| 13 | 13 | LKnee |
| 14 | 14 | LAnkle |
| 15 | 15 | REye |
| 16 | 16 | LEye |

**주의**: Wrist 인덱스가 다름!
- Body: LWrist=7, RWrist=4
- Wholebody (종속 필터링에서 사용): LWrist=9, RWrist=10

---

## 🧪 테스트 방법

### 1. 설정 파일 수정
`pose_transfer/config/default.yaml`:
```yaml
cross_filter:
  enabled: true  # false → true로 변경
```

### 2. 테스트 스크립트 실행
```bash
python test_cross_filter.py
```

### 3. 예상 출력
```
✅ Cross-Filter 적용: 78/133 keypoints 승인

그룹별 승인 개수:
  Body (0-16):       17/17
  LFoot (17-19):     3/3
  RFoot (20-22):     3/3
  Face (23-90):      68/68
  LHand (91-111):    0/21  ← 왼손목 없음 → 손가락 제거
  RHand (112-132):   21/21 ← 오른손목 있음 → 손가락 승인

부모-자식 종속성 검증:
  왼손목 (9):        conf=0.00 → 왼손가락 0/21
  오른손목 (10):     conf=5.20 → 오른손가락 21/21
```

---

## 📝 주요 파일 변경 내역

### 신규 생성
1. `pose_transfer/extractors/body_extractor.py` (108 lines)
   - BodyExtractor 클래스
   
2. `pose_transfer/logic/cross_filter.py` (160+ lines)
   - CrossFilter 클래스
   - CrossFilterConfig 데이터클래스
   - 4단계 필터링 알고리즘

3. `test_cross_filter.py` (테스트 스크립트)

### 수정
1. `pose_transfer/config/default.yaml`
   - cross_filter 섹션 추가 (line ~138)
   
2. `pose_transfer/pipeline.py`
   - PipelineConfig: cross_filter 파라미터 6개 추가
   - from_yaml(): cross_filter 설정 파싱
   - _init_modules(): Body extractor, CrossFilter 초기화
   - extract_pose(): CrossFilter 적용 로직
   
3. `pose_transfer/extractors/__init__.py`
   - BodyExtractor import/export 추가
   
4. `pose_transfer/logic/__init__.py`
   - CrossFilter, CrossFilterConfig import/export 추가

---

## 🎯 기대 효과

### Before (Ghost Filter 사용)
- 발목 아래 발가락 3개가 안 그려짐
- dist_std=16.9 < 50.0 → OCCLUDED 오판
- 통계 기반 필터링의 한계

### After (Cross-Filter 사용)
- Body 모델이 "발목 있음" 승인 → 발가락 3개 전부 승인
- DWPose 환각 제거 (손목 없으면 손가락도 제거)
- 부모-자식 종속 필터링으로 논리적 일관성 확보

---

## 🔍 디버깅 가이드

### Cross-Filter가 작동하지 않는 경우
1. `default.yaml`에서 `cross_filter.enabled: true` 확인
2. 파이프라인 초기화 로그 확인:
   ```
   ✅ Body Extractor 초기화 완료 (Cross-Filter 모드)
   ✅ Cross Filter 초기화 완료
   ```
3. 포즈 추출 로그 확인:
   ```
   ✅ Cross-Filter 적용: 78/133 keypoints 승인
   ```

### 손가락/발가락이 여전히 안 나오는 경우
1. Body 모델이 손목/발목을 감지했는지 확인:
   - `body_confidence_threshold: 0.3` → 0.2로 낮춰보기
2. DWPose 자체 신뢰도 문제:
   - `dw_min_confidence: 0.05` → 0.01로 낮춰보기

### 너무 많은 keypoint가 제거되는 경우
- `body_confidence_threshold: 0.3` → 0.4로 높이기 (Body 모델 더 보수적으로)

---

## 📚 참고 문서

- [FEET_NOT_RENDERING_ROOT_CAUSE.md](FEET_NOT_RENDERING_ROOT_CAUSE.md): 원래 문제 진단
- [CROSS_FILTER_FEASIBILITY.md](CROSS_FILTER_FEASIBILITY.md): 구현 가능성 분석
- rtmlib Body 모델 문서: https://github.com/Tau-J/rtmlib

---

**구현 완료일**: 2025-01-XX
**구현자**: GitHub Copilot (Claude Sonnet 4.5)
**테스트 상태**: ⏳ 사용자 테스트 대기중
