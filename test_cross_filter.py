"""
Cross-Filter (부모-자식 종속 필터링) 테스트 스크립트

Body 모델의 "존재 유무" + DWPose의 "정밀 좌표" 결합
- 손목 없으면 → 손가락도 없음
- 발목 없으면 → 발가락도 없음  
- 코/목 없으면 → 얼굴도 없음
"""
import cv2
import numpy as np
from pathlib import Path
from pose_transfer import PoseTransferPipeline

def test_cross_filter():
    """Cross-Filter 테스트"""
    
    # YAML 설정 파일에서 cross_filter.enabled = true로 설정 필요
    config_path = Path(__file__).parent / "pose_transfer" / "config" / "default.yaml"
    
    # 테스트 이미지 경로
    test_image = Path(__file__).parent / "test_io" / "inputs" / "test_image.jpg"
    
    if not test_image.exists():
        print(f"❌ 테스트 이미지가 없습니다: {test_image}")
        print("   io/inputs/src/ 폴더의 이미지 중 하나를 복사해주세요")
        return
    
    print("="*80)
    print("Cross-Filter (부모-자식 종속 필터링) 테스트")
    print("="*80)
    print(f"✅ 테스트 이미지: {test_image.name}")
    print()
    
    # 파이프라인 초기화
    print("📦 파이프라인 초기화 중...")
    pipeline = PoseTransferPipeline.from_yaml(str(config_path))
    print()
    
    # 이미지 읽기
    img = cv2.imread(str(test_image))
    if img is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {test_image}")
        return
    
    print(f"🖼️  이미지 크기: {img.shape[1]}x{img.shape[0]}")
    print()
    
    # 포즈 추출 (Cross-Filter 적용됨)
    print("🔍 포즈 추출 중 (Cross-Filter 적용)...")
    kpts, scores, person_idx, image_size = pipeline.extract_pose(img)
    
    print()
    print("="*80)
    print("결과 분석")
    print("="*80)
    
    # 승인된 keypoints 개수 세기
    approved_count = np.sum(scores > 0)
    print(f"✅ 승인된 Keypoints: {approved_count}/133")
    print()
    
    # 그룹별 승인 개수
    body_17_approved = np.sum(scores[:17] > 0)
    face_approved = np.sum(scores[23:91] > 0)
    left_hand_approved = np.sum(scores[91:112] > 0)
    right_hand_approved = np.sum(scores[112:133] > 0)
    left_foot_approved = np.sum(scores[17:20] > 0)
    right_foot_approved = np.sum(scores[20:23] > 0)
    
    print("그룹별 승인 개수:")
    print(f"  Body (0-16):       {body_17_approved}/17")
    print(f"  LFoot (17-19):     {left_foot_approved}/3")
    print(f"  RFoot (20-22):     {right_foot_approved}/3")
    print(f"  Face (23-90):      {face_approved}/68")
    print(f"  LHand (91-111):    {left_hand_approved}/21")
    print(f"  RHand (112-132):   {right_hand_approved}/21")
    print()
    
    # 부모-자식 검증
    print("부모-자식 종속성 검증:")
    left_wrist = scores[9]
    right_wrist = scores[10]
    left_ankle = scores[15]
    right_ankle = scores[16]
    nose = scores[0]
    
    print(f"  왼손목 (9):        conf={left_wrist:.2f} → 왼손가락 {left_hand_approved}/21")
    print(f"  오른손목 (10):     conf={right_wrist:.2f} → 오른손가락 {right_hand_approved}/21")
    print(f"  왼발목 (15):       conf={left_ankle:.2f} → 왼발 {left_foot_approved}/3")
    print(f"  오른발목 (16):     conf={right_ankle:.2f} → 오른발 {right_foot_approved}/3")
    print(f"  코 (0):            conf={nose:.2f} → 얼굴 {face_approved}/68")
    print()
    
    # 예상 동작 설명
    print("="*80)
    print("Cross-Filter 동작 원리")
    print("="*80)
    print("1. Body 모델 (17 keypoints)이 '존재 유무' 판단")
    print("   → 신뢰도 > 0.3인 Body 키포인트만 '진짜 존재'로 인정")
    print()
    print("2. DWPose 모델 (133 keypoints)이 '정밀 좌표' 제공")
    print("   → Body가 승인한 부위의 DWPose 좌표만 사용")
    print()
    print("3. 부모-자식 종속 필터링:")
    print("   ✅ 손목(9/10) 있음 → DWPose 손가락(91-132) 전부 승인")
    print("   ✅ 발목(15/16) 있음 → DWPose 발가락(17-22) 전부 승인")
    print("   ✅ 코(0) 또는 목(1) 있음 → DWPose 얼굴(23-90) 전부 승인")
    print()
    print("   ❌ 손목 없음 → 손가락 전부 제거 (DWPose 환각 방지)")
    print("   ❌ 발목 없음 → 발가락 전부 제거")
    print("="*80)

if __name__ == "__main__":
    test_cross_filter()
