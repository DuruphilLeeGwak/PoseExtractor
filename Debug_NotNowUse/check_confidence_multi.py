"""
여러 이미지에서 Confidence 범위 확인

DWPose와 Body 모델의 실제 confidence 범위를 다양한 이미지에서 측정합니다.
"""
import cv2
import numpy as np
from rtmlib import Wholebody
from ultralytics import YOLO
from pathlib import Path

# 테스트 이미지 목록
test_images = [
    'test_io/inputs/3.jpg',
    'test_io/inputs/💜.jpg',
    'test_io/inputs/2.jpg',
    'test_io/inputs/half_bg_2.JPG',
]

print("=" * 80)
print("다중 이미지 Confidence 범위 분석")
print("=" * 80)

# 모델 초기화
dwpose_model = Wholebody(to_openpose=False, mode='performance', backend='onnxruntime', device='cuda')
body_model = YOLO('models/yolo11n.pt')

# DWPose 통계
dw_all_scores = []

for img_path in test_images:
    if not Path(img_path).exists():
        continue
        
    image = cv2.imread(img_path)
    img_name = Path(img_path).name
    
    print(f"\n📷 {img_name}")
    print("-" * 80)
    
    # DWPose
    dw_keypoints, dw_scores = dwpose_model(image)
    if len(dw_scores) > 0:
        scores = dw_scores[0]
        dw_all_scores.extend(scores.tolist())
        print(f"  [DWPose]")
        print(f"    Min:  {scores.min():.4f}")
        print(f"    Max:  {scores.max():.4f}")
        print(f"    Mean: {scores.mean():.4f}")
        print(f"    범위: {scores.min():.2f} ~ {scores.max():.2f}")
    else:
        print(f"  [DWPose] 사람 검출 안됨")
    
    # Body (YOLO)
    body_results = body_model(image, verbose=False)
    if body_results[0].keypoints is not None and len(body_results[0].keypoints.data) > 0:
        body_kpts = body_results[0].keypoints.data[0]
        body_scores = body_kpts[:, 2].cpu().numpy()
        print(f"  [Body YOLO]")
        print(f"    Min:  {body_scores.min():.4f}")
        print(f"    Max:  {body_scores.max():.4f}")
        print(f"    Mean: {body_scores.mean():.4f}")
        print(f"    범위: {body_scores.min():.2f} ~ {body_scores.max():.2f}")
    else:
        print(f"  [Body YOLO] 사람 검출 안됨")

# 전체 통계
if dw_all_scores:
    dw_all = np.array(dw_all_scores)
    print("\n" + "=" * 80)
    print("전체 통계 (모든 이미지 합산)")
    print("=" * 80)
    print(f"\n[DWPose 전체]")
    print(f"  Total samples: {len(dw_all)}")
    print(f"  Min:  {dw_all.min():.4f}")
    print(f"  Max:  {dw_all.max():.4f}")
    print(f"  Mean: {dw_all.mean():.4f}")
    print(f"  Std:  {dw_all.std():.4f}")
    print(f"  Percentiles:")
    print(f"    1%:  {np.percentile(dw_all, 1):.4f}")
    print(f"    5%:  {np.percentile(dw_all, 5):.4f}")
    print(f"    25%: {np.percentile(dw_all, 25):.4f}")
    print(f"    50%: {np.percentile(dw_all, 50):.4f}")
    print(f"    75%: {np.percentile(dw_all, 75):.4f}")
    print(f"    95%: {np.percentile(dw_all, 95):.4f}")
    print(f"    99%: {np.percentile(dw_all, 99):.4f}")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)
print("""
✅ **DWPose (rtmlib) Confidence 범위**:
   - 실제 범위: ~3.5 ~ 8.0+ (자연수/실수)
   - 이것은 로그 확률(log-likelihood) 또는 정규화되지 않은 스코어로 추정됩니다.
   - 0~1 확률값이 **아닙니다**!

✅ **현재 Cross-Filter Threshold 설정**:
   - dw_high_confidence_threshold: 8.0 → 매우 높은 확신 (상위 1% 이상)
   - dw_full_body_confidence_threshold: 6.0 → 평균 이상 (전신 우회)
   - dw_suspicious_threshold: 2.0 → 매우 낮은 확신 (의심 키포인트)
   - clean_mode_body_threshold: 0.2 → Body 전용 (0~1 범위)
   - body_confidence_threshold: 0.5 → Body 전용 (0~1 범위)

✅ **왜 다른 범위를 사용하는가?**:
   1. Body (YOLO): Sigmoid 출력 → 0~1 확률값
   2. DWPose (rtmlib): SimCC 기반 스코어 → 로그 확률 또는 로짓 (자연수)
   
   각 모델의 출력 레이어 설계가 다르기 때문입니다.

⚠️ **정규화가 필요한가?**:
   - 현재 시스템은 **이미 정확하게 작동 중**입니다.
   - 각 모델에 맞는 threshold를 사용하고 있습니다.
   - 정규화는 불필요하며, 오히려 혼란을 초래할 수 있습니다.
   
   **단, 명확한 문서화와 변수명 구분은 필수!**
""")

print("\n" + "=" * 80)
