"""Cross-Filter 환각 제거 테스트 (잘린 이미지)"""
import cv2
import numpy as np
from pathlib import Path
from pose_transfer.extractors import BodyExtractor, DWPoseExtractorFactory
from pose_transfer.logic import CrossFilter, CrossFilterConfig

print("=" * 80)
print("Cross-Filter 환각 제거 테스트 (잘린 이미지)")
print("=" * 80)

# 이미지 경로
test_image = Path("io/inputs/ref/martial art reference.jpg")

if not test_image.exists():
    print(f"❌ 이미지가 없습니다: {test_image}")
    exit(1)

print(f"✅ 테스트 이미지: {test_image.name}")
img = cv2.imread(str(test_image))
print(f"🖼️  이미지 크기: {img.shape[1]}x{img.shape[0]}")
print()

# 초기화
print("📦 모델 초기화 중...")
dwpose = DWPoseExtractorFactory.get_instance(backend='onnxruntime', device='cpu', mode='lightweight')
body_extractor = BodyExtractor(mode='balanced', backend='onnxruntime', device='cpu')
cross_filter = CrossFilter(
    config=CrossFilterConfig(
        body_confidence_threshold=0.3,
        enable_hand_dependency=True,
        enable_foot_dependency=True,
        enable_face_dependency=True,
        dw_min_confidence=0.05
    )
)
print("✅ 모델 초기화 완료\n")

# 추출
print("🔍 포즈 추출 중...")
dw_kpts_all, dw_scores_all = dwpose.extract(img)
body_kpts_all, body_scores_all = body_extractor.extract(img)

if len(dw_kpts_all) == 0 or len(body_kpts_all) == 0:
    print("❌ 사람을 감지하지 못했습니다.")
    exit(1)

dw_kpts = dw_kpts_all[0]
dw_scores = dw_scores_all[0]
body_kpts = body_kpts_all[0]
body_scores = body_scores_all[0]
print(f"✅ 추출 완료\n")

# CrossFilter 적용
print("🔄 CrossFilter 적용 중...")
filtered_kpts, filtered_scores, approved_indices = cross_filter.filter(
    body_keypoints=body_kpts,
    body_scores=body_scores,
    dw_keypoints=dw_kpts,
    dw_scores=dw_scores
)
print(f"✅ CrossFilter 적용 완료\n")

# 결과 비교
print("=" * 80)
print("DWPose vs Cross-Filter 비교")
print("=" * 80)

def count_keypoints(scores, threshold=0.05):
    return np.sum(scores > threshold)

# DWPose 원본
dw_body_count = count_keypoints(dw_scores[:17])
dw_lfoot_count = count_keypoints(dw_scores[17:20])
dw_rfoot_count = count_keypoints(dw_scores[20:23])
dw_face_count = count_keypoints(dw_scores[23:91])
dw_lhand_count = count_keypoints(dw_scores[91:112])
dw_rhand_count = count_keypoints(dw_scores[112:133])

# CrossFilter 적용 후
cf_body_count = sum(1 for i in range(17) if i in approved_indices)
cf_lfoot_count = sum(1 for i in range(17, 20) if i in approved_indices)
cf_rfoot_count = sum(1 for i in range(20, 23) if i in approved_indices)
cf_face_count = sum(1 for i in range(23, 91) if i in approved_indices)
cf_lhand_count = sum(1 for i in range(91, 112) if i in approved_indices)
cf_rhand_count = sum(1 for i in range(112, 133) if i in approved_indices)

print(f"{'부위':<15} {'DWPose 원본':<15} {'CrossFilter 후':<15} {'제거된 환각'}")
print("-" * 80)
print(f"{'Body (0-16)':<15} {dw_body_count:>6}/17 {cf_body_count:>15}/17 {dw_body_count - cf_body_count:>12}")
print(f"{'LFoot (17-19)':<15} {dw_lfoot_count:>6}/3 {cf_lfoot_count:>16}/3 {dw_lfoot_count - cf_lfoot_count:>12}")
print(f"{'RFoot (20-22)':<15} {dw_rfoot_count:>6}/3 {cf_rfoot_count:>16}/3 {dw_rfoot_count - cf_rfoot_count:>12}")
print(f"{'Face (23-90)':<15} {dw_face_count:>6}/68 {cf_face_count:>14}/68 {dw_face_count - cf_face_count:>12}")
print(f"{'LHand (91-111)':<15} {dw_lhand_count:>6}/21 {cf_lhand_count:>14}/21 {dw_lhand_count - cf_lhand_count:>12}")
print(f"{'RHand (112-132)':<15} {dw_rhand_count:>6}/21 {cf_rhand_count:>14}/21 {dw_rhand_count - cf_rhand_count:>12}")
print("-" * 80)
dw_total = count_keypoints(dw_scores)
cf_total = len(approved_indices)
print(f"{'전체 (0-132)':<15} {dw_total:>6}/133 {cf_total:>13}/133 {dw_total - cf_total:>12}")
print()

# Body 부위별 신뢰도
print("=" * 80)
print("Body 모델 신뢰도 (존재 판단 기준)")
print("=" * 80)
keypoint_names = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", 
                 "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye"]
for i, name in enumerate(keypoint_names):
    conf = body_scores[i]
    status = "✅ 존재" if conf > 0.3 else "❌ 없음"
    print(f"{i:2d}. {name:<12} conf={conf:.2f}  {status}")
print()

# 환각 제거 효과
removed_count = dw_total - cf_total
if removed_count > 0:
    print("=" * 80)
    print(f"🎯 환각 제거 효과: {removed_count}개 키포인트 제거")
    print("=" * 80)
    print("Body 모델이 '없음'으로 판단한 부위의 DWPose 환각을 제거했습니다!")
    print()
else:
    print("=" * 80)
    print("✅ 전신 이미지: 모든 부위가 실제로 존재하므로 제거 없음")
    print("=" * 80)
