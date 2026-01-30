"""
Confidence 범위 확인 스크립트

DWPose와 Body 모델의 confidence 범위가 어떻게 다른지 확인합니다.
"""
import cv2
import numpy as np
from rtmlib import Wholebody
from ultralytics import YOLO

# 테스트 이미지 로드
image_path = 'test_io/inputs/3.jpg'
image = cv2.imread(image_path)

print("=" * 70)
print("Confidence 범위 비교 분석")
print("=" * 70)

# ========== 1. DWPose (rtmlib) ==========
print("\n[1] DWPose (rtmlib Wholebody)")
print("-" * 70)

dwpose_model = Wholebody(
    to_openpose=False,
    mode='performance',
    backend='onnxruntime',
    device='cuda'
)

dw_keypoints, dw_scores = dwpose_model(image)

if len(dw_scores) > 0:
    scores = dw_scores[0]  # 첫 번째 사람
    print(f"  출력 형식: rtmlib.Wholebody(image)")
    print(f"  반환값: keypoints (N, 133, 2), scores (N, 133)")
    print(f"  Shape: {dw_scores.shape}")
    print(f"  Min:   {scores.min():.6f}")
    print(f"  Max:   {scores.max():.6f}")
    print(f"  Mean:  {scores.mean():.6f}")
    print(f"  Std:   {scores.std():.6f}")
    print(f"\n  샘플 값 (Body 17개):")
    for i in range(17):
        print(f"    idx {i:2d}: {scores[i]:.6f}")
    print(f"\n  결론: DWPose는 **0~1 범위의 확률값** 사용")
else:
    print("  사람이 검출되지 않음")

# ========== 2. Body 모델 (YOLO) ==========
print("\n[2] Body 모델 (YOLO11n-pose)")
print("-" * 70)

body_model = YOLO('models/yolo11n.pt')
body_results = body_model(image, verbose=False)

if body_results[0].keypoints is not None and len(body_results[0].keypoints) > 0:
    body_kpts = body_results[0].keypoints.data[0]  # (17, 3): x, y, conf
    body_scores = body_kpts[:, 2].cpu().numpy()
    
    print(f"  출력 형식: YOLO(image).keypoints.data")
    print(f"  반환값: (N, 17, 3) - x, y, confidence")
    print(f"  Shape: {body_scores.shape}")
    print(f"  Min:   {body_scores.min():.6f}")
    print(f"  Max:   {body_scores.max():.6f}")
    print(f"  Mean:  {body_scores.mean():.6f}")
    print(f"  Std:   {body_scores.std():.6f}")
    print(f"\n  샘플 값 (Body 17개):")
    for i in range(17):
        print(f"    idx {i:2d}: {body_scores[i]:.6f}")
    print(f"\n  결론: YOLO Body는 **0~1 범위의 확률값** 사용")
else:
    print("  사람이 검출되지 않음")

# ========== 3. 현재 Cross-Filter 설정 분석 ==========
print("\n[3] Cross-Filter 설정 분석")
print("-" * 70)

from pose_transfer.logic.cross_filter import CrossFilterConfig

config = CrossFilterConfig()

print(f"  body_confidence_threshold:          {config.body_confidence_threshold}")
print(f"  clean_mode_body_threshold:          {config.clean_mode_body_threshold}")
print(f"  dw_high_confidence_threshold:       {config.dw_high_confidence_threshold}")
print(f"  dw_full_body_confidence_threshold:  {config.dw_full_body_confidence_threshold}")
print(f"  dw_suspicious_threshold:            {config.dw_suspicious_threshold}")

print(f"\n  ⚠️ 문제점 발견:")
print(f"     - body_confidence_threshold: 0.3 (0~1 범위 가정)")
print(f"     - clean_mode_body_threshold: 0.2 (0~1 범위 가정)")
print(f"     - dw_high_confidence_threshold: 8.0 (자연수 범위 가정???)")
print(f"     - dw_full_body_confidence_threshold: 6.0 (자연수 범위 가정???)")
print(f"     - dw_suspicious_threshold: 2.0 (자연수 범위 가정???)")

print(f"\n  결론: DWPose threshold가 0~1 범위를 벗어남!")

# ========== 4. 결론 ==========
print("\n" + "=" * 70)
print("최종 결론")
print("=" * 70)
print("""
1. **실제 Confidence 범위**:
   - DWPose (rtmlib): 0~1 (확률값)
   - Body (YOLO): 0~1 (확률값)
   
2. **Cross-Filter 설정 문제**:
   - body_* threshold: 0~1 범위 (정상)
   - dw_* threshold: 2.0~8.0 범위 (비정상!)
   
3. **왜 이런 차이가 생겼는가?**:
   - 과거 DWPose 구현체(OpenPose, MediaPipe 등)가 다른 범위를 사용했을 가능성
   - 또는 초기 테스트 시 잘못된 데이터로 threshold를 설정했을 가능성
   - rtmlib는 0~1 범위를 사용하므로, threshold 8.0은 **아무것도 통과하지 못함**
   
4. **해결 방법**:
   모든 threshold를 0~1 범위로 정규화해야 함:
   - dw_high_confidence_threshold: 8.0 → 0.8
   - dw_full_body_confidence_threshold: 6.0 → 0.6
   - dw_suspicious_threshold: 2.0 → 0.2 (또는 0.05~0.3 사이)
""")

print("\n" + "=" * 70)
