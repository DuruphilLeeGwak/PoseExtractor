# Ghost Filter 개선 사항 테스트 가이드

## 개선 전후 비교

### 현재 문제점 (개선 전)
1. **전신 사진에서 발목/발이 사라짐**
   - boundary_tolerance=30px로 인해 하단 경계 근처 정상 발도 제거
   
2. **명확한 팔/손도 반투명 처리**
   - hand_min_avg_confidence=3.0이 너무 높아 정상 손도 "추정"으로 판단
   - 작은 손(멀리 있는)의 낮은 dist_std도 "비정상"으로 판단

3. **폐색 오판**
   - 모든 부위에 동일 기준 적용으로 인한 과도한 필터링

### 개선 사항
1. **boundary_tolerance 동적 적용**
   - 기본: 10px (상/좌/우)
   - 하단: 80px + 신뢰도 4.0 이상만 통과
   
2. **부위별 맞춤 기준**
   - 손: min_avg_conf 2.0, min_dist_std 30.0
   - 얼굴: min_avg_conf 4.0, min_dist_std 15.0
   - 팔/다리: 기하 체크 무시, min_avg_conf 1.5
   - 발: 기하 체크 무시, min_avg_conf 1.5

3. **신뢰도 기반 우회**
   - 손: avg_conf > 3.5이면 기하 체크 무시
   - 얼굴: avg_conf > 5.0이면 모든 체크 무시

## 테스트 방법

### 1. 설정 확인
\`\`\`bash
# default.yaml 확인
type pose_transfer\\config\\default.yaml | Select-String -Pattern "ghost_filter|boundary_tolerance|hand_min"
\`\`\`

예상 출력:
\`\`\`
ghost_filter:
  enabled: true
  boundary_tolerance: 10.0
  hand_min_avg_confidence: 2.0
  hand_min_distance_std: 30.0
\`\`\`

### 2. 테스트 실행
\`\`\`bash
# Ghost Filter 활성화 상태로 실행
python PoseExtractor.py
# 또는
python -m pose_transfer.api
\`\`\`

### 3. 결과 확인
출력 폴더에서 다음 파일들을 확인:
- `*_sk.jpg`: 스켈레톤 렌더링 이미지
- `*_rend.jpg`: 최종 렌더링 이미지
- `_ghostfilter_layers_debug.txt`: 디버그 로그

### 4. 디버그 로그 분석
\`\`\`bash
# 손 판정 결과 확인
type test_io\\outputs\\<폴더명>\\_ghostfilter_layers_debug.txt | Select-String -Pattern "LHand verdict"

# 발목/발 판정 결과 확인
type test_io\\outputs\\<폴더명>\\_ghostfilter_layers_debug.txt | Select-String -Pattern "LAnkle|RAnkle"
\`\`\`

## 예상 결과

### ✅ 전신 사진
- **발목/발**: 이미지 하단에 있어도 정상 렌더링
- **디버그 로그**: `LAnkle/RAnkle`가 제거되지 않음

### ✅ 명확한 팔/손
- **팔/손**: 불투명하게 렌더링 (반투명 아님)
- **디버그 로그**: `[BODY][Step3.5] LHand verdict=KEEP`

### ✅ 더미/환각
- **경계 더미**: 여전히 제거됨
- **디버그 로그**: `dummy_no_parent` 또는 `clustered_hand`

## 변경 파일
- `pose_transfer/config/default.yaml`: 파라미터 조정
- `pose_transfer/logic/ghost_filter.py`: 로직 개선

## 롤백 방법
원래 설정으로 되돌리려면:
\`\`\`yaml
# default.yaml
ghost_filter:
  boundary_tolerance: 30.0
  hand_min_avg_confidence: 3.0
  hand_min_distance_std: 50.0
\`\`\`
