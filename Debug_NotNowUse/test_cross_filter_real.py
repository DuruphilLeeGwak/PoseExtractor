"""Cross-Filter 실제 동작 테스트"""
import cv2
import numpy as np
from pathlib import Path
from pose_transfer.extractors import BodyExtractor, DWPoseExtractorFactory
from pose_transfer.logic import CrossFilter, CrossFilterConfig

print("=" * 80)
print("Cross-Filter 실제 동작 테스트")
print("=" * 80)

# 테스트 이미지 찾기
test_dirs = [
    Path("io/inputs/src"),
    Path("test_io/inputs"),
    Path("io/inputs/ref")
]

test_image = None
for test_dir in test_dirs:
    if test_dir.exists():
        images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if images:
            test_image = images[0]
            break

if not test_image:
    print("❌ 테스트 이미지를 찾을 수 없습니다.")
    print("   io/inputs/src/, io/inputs/ref/, test_io/inputs/ 폴더에 이미지를 넣어주세요")
    exit(1)

print(f"✅ 테스트 이미지: {test_image}")
print()

# 이미지 읽기
img = cv2.imread(str(test_image))
if img is None:
    print(f"❌ 이미지를 읽을 수 없습니다: {test_image}")
    exit(1)

print(f"🖼️  이미지 크기: {img.shape[1]}x{img.shape[0]}")
print()

# 1. DWPose Extractor 초기화
print("📦 DWPose Extractor 초기화 중...")
dwpose = DWPoseExtractorFactory.get_instance(
    backend='onnxruntime',
    device='cpu',
    mode='lightweight',
    to_openpose=False
)
print("✅ DWPose 초기화 완료")
print()

# 2. Body Extractor 초기화
print("📦 Body Extractor 초기화 중...")
body_extractor = BodyExtractor(mode='balanced', backend='onnxruntime', device='cpu')
print("✅ Body 초기화 완료")
print()

# 3. CrossFilter 초기화
print("📦 CrossFilter 초기화 중...")
cross_filter = CrossFilter(
    config=CrossFilterConfig(
        body_confidence_threshold=0.3,
        enable_hand_dependency=True,
        enable_foot_dependency=True,
        enable_face_dependency=True,
        dw_min_confidence=0.05
    )
)
print("✅ CrossFilter 초기화 완료")
print()

# 4. DWPose 추출
print("🔍 DWPose (Wholebody 133 keypoints) 추출 중...")
dw_kpts_all, dw_scores_all = dwpose.extract(img)
if len(dw_kpts_all) == 0:
    print("❌ DWPose가 사람을 감지하지 못했습니다.")
    exit(1)
dw_kpts = dw_kpts_all[0]
dw_scores = dw_scores_all[0]
print(f"✅ DWPose 추출 완료: {dw_kpts.shape}, 평균 신뢰도={dw_scores.mean():.2f}")
print()

# 5. Body 추출
print("🔍 Body (COCO 17 keypoints) 추출 중...")
body_kpts_all, body_scores_all = body_extractor.extract(img)
if len(body_kpts_all) == 0:
    print("❌ Body가 사람을 감지하지 못했습니다.")
    exit(1)
body_kpts = body_kpts_all[0]
body_scores = body_scores_all[0]
print(f"✅ Body 추출 완료: {body_kpts.shape}, 평균 신뢰도={body_scores.mean():.2f}")
print()

# 6. CrossFilter 적용
print("🔄 CrossFilter 적용 중...")
filtered_kpts, filtered_scores, approved_indices = cross_filter.filter(
    body_keypoints=body_kpts,
    body_scores=body_scores,
    dw_keypoints=dw_kpts,
    dw_scores=dw_scores
)
print(f"✅ CrossFilter 적용 완료")
print()

# 7. 결과 분석
print("=" * 80)
print("결과 분석")
print("=" * 80)
print(f"✅ 승인된 Keypoints: {len(approved_indices)}/133")
print()

# Body 17 검증
body_17_approved = [i for i in range(17) if i in approved_indices]
print(f"Body (0-16):       {len(body_17_approved)}/17 승인")
for i in body_17_approved:
    print(f"  - [{i}] conf={body_scores[i]:.2f}")
print()

# 발 검증
left_foot_approved = [i for i in range(17, 20) if i in approved_indices]
right_foot_approved = [i for i in range(20, 23) if i in approved_indices]
print(f"LFoot (17-19):     {len(left_foot_approved)}/3 승인")
print(f"RFoot (20-22):     {len(right_foot_approved)}/3 승인")
print()

# 얼굴 검증
face_approved = [i for i in range(23, 91) if i in approved_indices]
print(f"Face (23-90):      {len(face_approved)}/68 승인")
print()

# 손 검증
left_hand_approved = [i for i in range(91, 112) if i in approved_indices]
right_hand_approved = [i for i in range(112, 133) if i in approved_indices]
print(f"LHand (91-111):    {len(left_hand_approved)}/21 승인")
print(f"RHand (112-132):   {len(right_hand_approved)}/21 승인")
print()

# 부모-자식 검증
print("=" * 80)
print("부모-자식 종속성 검증")
print("=" * 80)
left_wrist_body = body_scores[7]  # Body의 LWrist는 인덱스 7
right_wrist_body = body_scores[4]  # Body의 RWrist는 인덱스 4
left_ankle = body_scores[14]  # Body의 LAnkle는 인덱스 14
right_ankle = body_scores[11]  # Body의 RAnkle는 인덱스 11
nose = body_scores[0]

print(f"왼손목 (Body 7):   conf={left_wrist_body:.2f} → 왼손가락 {len(left_hand_approved)}/21")
print(f"오른손목 (Body 4): conf={right_wrist_body:.2f} → 오른손가락 {len(right_hand_approved)}/21")
print(f"왼발목 (Body 14):  conf={left_ankle:.2f} → 왼발 {len(left_foot_approved)}/3")
print(f"오른발목 (Body 11): conf={right_ankle:.2f} → 오른발 {len(right_foot_approved)}/3")
print(f"코 (Body 0):       conf={nose:.2f} → 얼굴 {len(face_approved)}/68")
print()

print("=" * 80)
print("✅ 테스트 완료!")
print("=" * 80)
print()
print("💡 Cross-Filter 동작 원리:")
print("   1. Body 모델이 손목/발목/코의 '존재 여부' 판단 (신뢰도 > 0.3)")
print("   2. Body가 승인한 부위에 대해서만 DWPose 디테일 사용")
print("   3. Body가 승인 안 하면 → DWPose 환각 제거")
print("=" * 80)
