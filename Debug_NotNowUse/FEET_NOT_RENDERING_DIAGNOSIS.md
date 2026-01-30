"""
최종 진단: Ghost Filter가 발을 제거하는 상세한 원인 분석

이 파일은 실제 코드와 테스트 데이터를 기반으로 
발이 렌더링되지 않는 정확한 이유를 설명합니다.
"""

print("=" * 80)
print("Ghost Filter 발 제거 원인 - 최종 진단 리포트")
print("=" * 80)

print("\n[1] 테스트 데이터 분석 결과")
print("-" * 80)
print("""
테스트 파일: test_io/outputs/ghost filter_use20260121/origin_full_0001_kp.json

발 키포인트 원본 데이터:
  - LBigToe (idx=17):   x=382.7, y=680.8, conf=6.41
  - LSmallToe (idx=18): x=379.3, y=672.8, conf=6.62
  - LHeel (idx=19):     x=316.5, y=670.5, conf=6.46
  - RBigToe (idx=20):   x=140.9, y=679.7, conf=6.75
  - RSmallToe (idx=21): x=144.3, y=675.1, conf=6.22
  - RHeel (idx=22):     x=207.0, y=672.8, conf=6.22

발목 키포인트:
  - LAnkle (idx=15):    x=326.8, y=648.9, conf=6.70
  - RAnkle (idx=16):    x=195.6, y=650.0, conf=6.56

이미지 크기: 916 x 916px
""")

print("\n[2] Ghost Filter 검증 결과")
print("-" * 80)
print("""
✅ weak_leg 로직: 발을 제거하지 않음
   - 조건: y >= (h-50=866) AND conf < 4.5
   - 실제: 모든 발 y < 866, 모든 발 conf > 6.0
   - 결론: weak_leg 조건에 걸리지 않음

✅ boundary check: 발목을 제거하지 않음
   - boundary_tolerance = 10.0px
   - LAnkle: x=326.8 (10.0 < x < 906.0) ✓
   - RAnkle: x=195.6 (10.0 < x < 906.0) ✓
   - 결론: 발목이 boundary에 걸리지 않음

✅ chain kill: 발목이 살아있어서 발도 유지됨
   - hierarchy: 발(17-22) → 발목(15-16)
   - 발목이 제거되지 않았으므로 chain kill 발생 안 함
""")

print("\n[3] 발이 렌더링되지 않는 가능한 원인")
print("-" * 80)
print("""
Ghost Filter 자체는 발을 제거하지 않습니다!
그렇다면 렌더링 단계에서 문제가 발생한 것입니다.

가능한 원인:

📌 원인 1: FEET_BONES 연결이 없음
   skeleton_renderer.py에서 feet_bones를 그리지 않을 수 있습니다.
   확인 필요: FEET_BONES 정의와 렌더링 로직

📌 원인 2: out_of_frame_indices에 발이 마킹됨
   Ghost Filter가 발을 REMOVED하지 않고 OUT_OF_FRAME으로 마킹했을 가능성
   → skeleton_renderer.py line 186-188:
      if start_idx in out_of_frame_indices or end_idx in out_of_frame_indices:
          continue  # 렌더링 스킵!

📌 원인 3: 렌더링 threshold 문제
   kpt_threshold=0.3이지만, 발가락 중 일부가 threshold 미만일 가능성
   → 실제로는 모두 6.0 이상이므로 이 원인은 아님

📌 원인 4: FEET_BONES 정의 누락
   keypoint_constants.py에서 FEET_BONES가 제대로 정의되지 않았거나
   skeleton_renderer.py에서 feet_bones를 그리는 부분이 누락됨
""")

print("\n[4] 실제 코드 분석")
print("-" * 80)
print("""
ghost_filter.py (line 323-334): weak_leg 로직
  - 발(17-22)만 체크
  - y >= h-50 AND conf < 4.5 조건
  - 이 데이터에는 해당 없음 ✓

ghost_filter.py (line 686-701): chain kill 로직
  - hierarchy_rules: 발(17-22) → 발목(15-16)
  - 발목이 죽으면 발도 제거
  - 이 데이터에서 발목 살아있음 ✓

skeleton_renderer.py (line 55): feet_bones 초기화
  - self.feet_bones = get_feet_bone_indices()

skeleton_renderer.py (line 118): feet_bones 렌더링
  - self.feet_bones를 렌더링하는 코드 존재

skeleton_renderer.py (line 186-188): out_of_frame 체크
  - if start_idx in out_of_frame_indices or end_idx in out_of_frame_indices:
        continue
  - **여기서 발 bone이 스킵될 가능성 높음!**
""")

print("\n[5] 진단 결론")
print("=" * 80)
print("""
🎯 가장 가능성 높은 원인:

Ghost Filter가 발 키포인트(17-22)를 out_of_frame_indices에 마킹하고 있습니다.

이유:
1. Ghost Filter Step 2에서 발이 더미로 판단되거나
2. 프레임 밖으로 판단되거나
3. 해부학적 비정상으로 판단됨

결과:
- filtered_scores는 그대로 유지 (REMOVED 아님)
- out_of_frame_indices에 추가됨
- skeleton_renderer가 out_of_frame_indices를 받아서
  해당 키포인트가 포함된 bone을 렌더링하지 않음

확인 방법:
실제 Ghost Filter를 실행하여 out_of_frame_indices를 확인해야 합니다.
하지만 import 오류로 직접 실행이 불가능하므로,
코드 상에서 논리적으로 추론한 결과입니다.

해결 방법:
1. Ghost Filter의 out_of_frame 판정 로직 완화
2. boundary_tolerance를 더 줄임 (10.0 → 5.0)
3. 발 특화 로직 추가 (발은 더 관대하게 처리)
""")

print("=" * 80)
print("\n다음 단계: ")
print("실제 렌더링 결과 이미지를 확인하거나")
print("Ghost Filter의 out_of_frame_indices 로그를 출력하여 검증 필요")
print("=" * 80)
