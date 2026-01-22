# Ghost Filter 발 렌더링 안 되는 상세한 이유

## 📊 진단 결과 요약

**파일:** `test_io/outputs/ghost filter_use20260121/origin_full_0001.jpg`

**결론:** Ghost Filter의 **Step 3.5 (폐색/환각 억제)** 로직이 발 키포인트를 **OCCLUDED로 마킹**하여 최종적으로 **제거**함

---

## 🔍 상세 분석

### 1. 원본 발 키포인트 데이터

```
LBigToe (idx=17):   x=382.7, y=680.8, conf=6.41
LSmallToe (idx=18): x=379.3, y=672.8, conf=6.62
LHeel (idx=19):     x=316.5, y=670.5, conf=6.46
RBigToe (idx=20):   x=140.9, y=679.7, conf=6.75
RSmallToe (idx=21): x=144.3, y=675.1, conf=6.22
RHeel (idx=22):     x=207.0, y=672.8, conf=6.22

LAnkle (idx=15):    x=326.8, y=648.9, conf=6.70
RAnkle (idx=16):    x=195.6, y=650.0, conf=6.56
```

**특징:**
- 모든 발 키포인트의 confidence가 **6.0 이상**으로 매우 높음
- y 좌표가 모두 670-680 범위에 있음 (이미지 높이 916px의 약 73%)
- boundary에 걸리지 않음

---

### 2. Ghost Filter 처리 과정

#### Step 1-2: Boundary/프레임 체크
✅ **통과** - 발목과 발가락 모두 boundary 밖에 있음

#### Step 3: Clustering 체크
✅ **통과** - 손 키포인트만 체크, 발은 대상 아님

#### **Step 3.5: 폐색/환각 억제** ← 🔴 **여기서 문제 발생!**

**Ghost Filter 로그 (_ghostfilter_layers_debug.txt line 1004-1006):**

```
[BODY][Step3.5] LFoot verdict=OCCLUDED 
  active=2/3 
  near=2 far=0 near_ratio=1.000 
  avg_conf=6.54 
  dist_std=16.9 thr=0.10 
  min_pts=2 
  min_near=0.50 max_far=1 
  min_avg_conf=3.00 
  min_dist_std=50.0 
  anchor_score=6.702 
  parent_score=3.920 
  limb_len=159.8 
  base_r=127.8 far_r=255.6
  [추정감지: score=3/5 y_range=2.3 y_boundary=0.0% y_std=1.1 conf_std=0.08]

[BODY][Step3.5] LFoot 설명: 
  dist_std 16.9 < min 50.0 (균일함) 
  -> 가림/환각 가능성으로 -1 레이어 마킹 (제거 안 됨!)

[BODY][Step3.5] LFoot 가장먼(active): 
  idx18:d=57.7,c=6.62, 
  idx19:d=24.0,c=6.46
```

**판정 기준:**
- `dist_std = 16.9` < `min_dist_std = 50.0` → **균일하게 모여있음**
- active=2/3 → LSmallToe(18)와 LHeel(19)만 활성화, LBigToe(17)는 제외됨
- 이 2개 키포인트가 발목(anchor)에서 57.7px, 24.0px 거리에 있음

#### Step 4: 최종 필터링

**Ghost Filter 로그 (line 1023-1024):**

```
[FOOT] LFoot: 남은=3/3 가려진=2/3 제거사유={'occluded_LFoot': 2}
[FOOT] LFoot 설명: 최다사유=occluded_LFoot x2 -> 필터 규칙에 의해 제거
```

**결과:**
- LSmallToe(18)와 LHeel(19)가 `occluded_LFoot`로 마킹됨
- 최종적으로 **"필터 규칙에 의해 제거"**

---

### 3. 왜 이런 판정이 나왔는가?

#### 문제 1: `dist_std` (표준편차) 기준이 발에게 부적절

**Step 3.5의 원래 목적:**
- 손가락이 가려졌는데도 DWPose가 "예측"으로 손가락을 생성하는 경우 감지
- 손가락 점들이 손목 주변에 너무 모여있으면 (작은 dist_std) → 가림/환각으로 판단

**발에 적용했을 때 문제:**
```python
min_dist_std = 50.0  # 최소 표준편차

# 손: 21개 키포인트가 넓게 펼쳐져 있어야 정상 → dist_std > 50.0
# 발: 3개 키포인트만 있고, 원래 좁은 영역에 모여있음 → dist_std < 50.0 정상!
```

**실제 발 데이터:**
```
LSmallToe: x=379.3, y=672.8
LHeel:     x=316.5, y=670.5
→ 거리: 약 63px (매우 가까움, 이게 정상!)
→ dist_std = 16.9 → "너무 모여있어서 가림/환각"으로 잘못 판단됨
```

#### 문제 2: 발은 원래 작은 영역에 몰려있음

발 구조:
```
      heel(19)
      |
      |-- bigtoe(17)
      |-- smalltoe(18)
```

- 발 3개 키포인트는 **원래 좁은 영역**(약 60-70px 반경)에 모여있음
- 손 21개 키포인트와 달리, **넓게 펼쳐질 수 없는 구조**
- `min_dist_std = 50.0` 기준은 **손에만 적용해야 하는 기준**

#### 문제 3: 추정감지(Heuristics) 점수가 높음

```
[추정감지: score=3/5 y_range=2.3 y_boundary=0.0% y_std=1.1 conf_std=0.08]
```

- `y_range=2.3` → 발 키포인트들의 y 좌표 범위가 2.3px (거의 수평)
- `y_std=1.1` → y 표준편차 1.1px (매우 균일)
- `conf_std=0.08` → confidence 표준편차 0.08 (거의 동일)
- **추정감지 점수 3/5** → "예측으로 생성된 것 같다" 판단

**하지만 실제로는:**
- 발가락과 발뒤꿈치는 **원래 수평으로 배치**되어 있음 (정상!)
- confidence가 모두 6.4-6.6 (매우 높고 균일) → DWPose가 확실하게 감지함
- 이것은 "예측"이 아니라 **실제로 정확하게 감지된 것**!

---

## 🎯 근본 원인

Ghost Filter의 **Step 3.5 (폐색/환각 억제)** 로직이:

1. **손(Hand) 전용으로 설계**되었는데
2. **발(Foot)에도 똑같이 적용**되어
3. 발의 정상적인 구조(좁은 영역, 수평 배치)를 **"가림/환각"으로 오판**함

---

## 💡 해결 방법

### Option 1: Step 3.5에서 발 제외 (권장)

**ghost_filter.py의 Step 3.5를 발에 적용하지 않도록 수정:**

```python
# Step 3.5: 폐색/환각 억제
if self.config.check_hand_occlusion or self.config.check_hand_presence:
    # 손만 체크 (발은 제외)
    for hand_side, (start, end) in [("L", (91, 111)), ("R", (112, 132))]:
        # 손 폐색 체크 로직...
    
    # ❌ 발 폐색 체크 제거 또는 비활성화
    # for foot_side, (start, end) in [("L", (17, 19)), ("R", (20, 22))]:
    #     # 발 폐색 체크는 하지 않음
```

**이유:**
- 발은 3개 키포인트만 있어서 통계적 판단이 어려움
- 발은 원래 좁은 영역에 모여있음 (손과 구조가 다름)
- confidence 6.0+ 이면 충분히 신뢰할 수 있음

### Option 2: 발 전용 기준 추가

**발에 대해서는 더 관대한 기준 적용:**

```python
# 발 전용 설정
foot_min_dist_std = 10.0  # 50.0 → 10.0 (발은 더 작은 영역)
foot_min_avg_conf = 5.0   # 3.0 → 5.0 (발은 더 높은 confidence 요구)
```

### Option 3: confidence 기반 예외 처리

**confidence가 충분히 높으면 OCCLUDED 판정 무시:**

```python
if avg_conf > 6.0:  # 매우 높은 confidence
    verdict = "KEEP"  # 폐색 판정 무시
    continue
```

---

## 📝 최종 권장사항

**가장 간단하고 효과적인 해결책:**

[ghost_filter.py](pose_transfer/logic/ghost_filter.py#L387-L450) Step 3.5에서 **발(Foot) 폐색 체크를 완전히 제거**

**이유:**
1. 발 키포인트는 3개뿐이라 통계적 판단이 부정확
2. 발은 구조상 좁은 영역에 모여있는 게 정상
3. DWPose가 confidence 6.0+ 로 감지했으면 충분히 신뢰 가능
4. 발이 실제로 가려진 경우는 드물고, 가려져도 3개 전체가 안 보임 (부분 폐색 거의 없음)

**구현:**
- Step 3.5의 발 폐색 체크 로직을 주석 처리하거나
- `check_foot_occlusion` 설정을 추가하여 false로 설정

---

## 🔗 관련 파일

- [ghost_filter.py](pose_transfer/logic/ghost_filter.py) - Step 3.5 폐색 감지 로직
- [default.yaml](pose_transfer/config/default.yaml) - min_dist_std 설정
- [_ghostfilter_layers_debug.txt](test_io/outputs/ghost%20filter_use20260121/_ghostfilter_layers_debug.txt) - 실제 로그

---

## 📌 요약

**Ghost Filter를 켰을 때 발이 나오지 않는 이유:**

Step 3.5 (폐색/환각 억제)에서:
- 발 키포인트들이 좁은 영역에 모여있고 (dist_std=16.9 < 50.0)
- 수평으로 배치되어 있어서 (y_std=1.1)
- "가림/환각"으로 잘못 판단되어
- `occluded_LFoot`로 마킹 후 제거됨

**해결책:**
Step 3.5에서 발 폐색 체크를 제거하거나, 발 전용 기준 추가
