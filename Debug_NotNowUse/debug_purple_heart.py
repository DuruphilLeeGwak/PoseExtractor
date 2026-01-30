"""💜.jpg 디버깅 정보 생성 스크립트"""
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# debug_cross_filter를 임포트하여 사용
sys.path.insert(0, os.path.dirname(__file__))

# inputs 폴더에서 💜.jpg 찾기
inputs_dir = Path("test_io/inputs")
output_dir = Path("test_io/debug_outputs")

# 💜.jpg 파일 찾기
target_file = None
for file in inputs_dir.glob("*.jpg"):
    if "💜" in file.name:
        target_file = file
        break

if target_file is None:
    print("❌ 💜.jpg 파일을 찾을 수 없습니다")
    sys.exit(1)

print(f"✅ 찾은 파일: {target_file}")

# 출력 디렉토리 생성
output_dir.mkdir(exist_ok=True, parents=True)

# OpenCV가 이모지 파일명을 처리하지 못하므로 바이트로 읽기
try:
    # 바이트로 읽고 디코드
    with open(target_file, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        print("❌ 이미지 디코드 실패")
        sys.exit(1)
    
    print(f"📐 이미지 크기: {image.shape[1]}x{image.shape[0]}")
    
    # debug_cross_filter의 analyze_image 함수를 직접 사용하되, 이미지를 직접 전달
    from pose_transfer.extractors.dwpose_extractor import DWPoseExtractor
    from pose_transfer.extractors.keypoint_constants import BODY_KEYPOINTS
    
    # COCO17 body keypoint 이름
    BODY_17_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    BODY_17_TO_133 = {name: i for i, name in enumerate(BODY_17_NAMES)}
    
    # DWPose 초기화
    print("\n" + "=" * 80)
    print(f"분석 중: {target_file.name}")
    print("=" * 80)
    
    dwpose = DWPoseExtractor(
        backend='onnxruntime',
        device='cuda',
        mode='performance'
    )
    
    # DWPose 추출
    keypoints, scores = dwpose.extract_single(image, person_idx=0)
    
    if keypoints is None or scores is None:
        print("❌ DWPose 추출 실패 (사람 감지 안됨)")
        sys.exit(1)
    
    h, w = image.shape[:2]
    print(f"📐 이미지 크기: {w}x{h}")
    
    # 17개 body keypoints 추출
    body_17_indices = [BODY_17_TO_133[name] for name in BODY_17_NAMES]
    
    # 전신 확신 모드 체크
    dw_full_body_threshold = 6.0
    body_17_scores = [scores[idx] for idx in body_17_indices]
    is_full_body_confident = all(score > dw_full_body_threshold for score in body_17_scores)
    
    # 개별 고신뢰 키포인트 체크
    high_confidence_indices = [i for i in range(133) if scores[i] > 8.0]
    
    # 낮은 신뢰도 키포인트
    suspicious_indices = [i for i in range(133) if 0.05 < scores[i] <= 2.0]
    
    # 출력 파일명 생성
    output_txt = output_dir / f"{target_file.stem}_debug.txt"
    
    # 디버깅 정보 작성
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Cross-Filter 디버깅 정보\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"📁 파일명: {target_file.name}\n")
        f.write(f"📐 이미지 크기: {w}x{h}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"[1] 전신 확신 모드 체크\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"전신 확신 임계값: {dw_full_body_threshold}\n")
        f.write(f"전신 확신 모드: {'✅ 활성' if is_full_body_confident else '❌ 비활성'}\n")
        
        if is_full_body_confident:
            f.write(f"→ 모든 17개 body keypoints가 {dw_full_body_threshold} 이상\n")
            f.write(f"→ Body 모델 검증 없이 전체 승인\n")
        else:
            f.write(f"→ 일부 body keypoints가 {dw_full_body_threshold} 미만\n")
            f.write(f"→ Cross-Filter 정상 작동 (Body 모델 검증)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[2] DWPose Body 17 Keypoints 신뢰도\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"{'No':<4} {'Name':<18} {'Index':<6} {'X':<8} {'Y':<8} {'Score':<8} {'Status'}\n")
        f.write(f"{'-'*80}\n")
        
        for i, name in enumerate(BODY_17_NAMES):
            idx = BODY_17_TO_133[name]
            kpt = keypoints[idx]
            score = scores[idx]
            
            if score > dw_full_body_threshold:
                status = "✅ 고신뢰"
            elif score > 2.0:
                status = "⚠️  중간"
            elif score > 0.05:
                status = "⚠️  의심"
            else:
                status = "❌ 낮음"
            
            f.write(f"{i:<4} {name:<18} {idx:<6} {kpt[0]:<8.1f} {kpt[1]:<8.1f} {score:<8.3f} {status}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[3] 전체 133 Keypoints 통계\n")
        f.write(f"{'='*80}\n\n")
        
        total_kpts = len(scores)
        high_conf = sum(1 for s in scores if s > 8.0)
        full_body_conf = sum(1 for s in scores if s > 6.0)
        medium_conf = sum(1 for s in scores if 2.0 < s <= 6.0)
        suspicious_conf = sum(1 for s in scores if 0.05 < s <= 2.0)
        low_conf = sum(1 for s in scores if s <= 0.05)
        
        f.write(f"전체 키포인트: {total_kpts}\n")
        f.write(f"  > 8.0 (개별 고신뢰): {high_conf} 개\n")
        f.write(f"  > 6.0 (전신 기준): {full_body_conf} 개\n")
        f.write(f"  2.0 ~ 6.0 (중간): {medium_conf} 개\n")
        f.write(f"  0.05 ~ 2.0 (의심): {suspicious_conf} 개  ⚠️ 할루시네이션 가능\n")
        f.write(f"  ≤ 0.05 (매우낮음): {low_conf} 개\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[4] 의심스러운 키포인트 (0.05 ~ 2.0)\n")
        f.write(f"{'='*80}\n\n")
        
        if suspicious_indices:
            f.write(f"총 {len(suspicious_indices)}개의 의심 키포인트 발견\n")
            f.write(f"→ 이들은 할루시네이션일 가능성이 높습니다\n\n")
            
            f.write(f"{'Index':<8} {'X':<10} {'Y':<10} {'Score':<10} {'영역'}\n")
            f.write(f"{'-'*80}\n")
            
            for idx in suspicious_indices[:30]:
                kpt = keypoints[idx]
                score = scores[idx]
                
                if idx < 17:
                    region = "Body"
                elif 17 <= idx < 23:
                    region = "Feet"
                elif 23 <= idx < 91:
                    region = "Face"
                elif 91 <= idx < 112:
                    region = "Left Hand"
                else:
                    region = "Right Hand"
                
                f.write(f"{idx:<8} {kpt[0]:<10.1f} {kpt[1]:<10.1f} {score:<10.3f} {region}\n")
            
            if len(suspicious_indices) > 30:
                f.write(f"\n... 외 {len(suspicious_indices) - 30}개\n")
        else:
            f.write("의심 키포인트 없음\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[5] 개별 고신뢰 키포인트 (> 8.0)\n")
        f.write(f"{'='*80}\n\n")
        
        if high_confidence_indices:
            f.write(f"총 {len(high_confidence_indices)}개의 고신뢰 키포인트\n")
            f.write(f"→ 이들은 Body 검증 없이 승인됩니다\n\n")
            
            f.write(f"{'Index':<8} {'X':<10} {'Y':<10} {'Score':<10} {'영역'}\n")
            f.write(f"{'-'*80}\n")
            
            for idx in high_confidence_indices[:20]:
                kpt = keypoints[idx]
                score = scores[idx]
                
                if idx < 17:
                    region = "Body"
                elif 17 <= idx < 23:
                    region = "Feet"
                elif 23 <= idx < 91:
                    region = "Face"
                elif 91 <= idx < 112:
                    region = "Left Hand"
                else:
                    region = "Right Hand"
                
                f.write(f"{idx:<8} {kpt[0]:<10.1f} {kpt[1]:<10.1f} {score:<10.3f} {region}\n")
            
            if len(high_confidence_indices) > 20:
                f.write(f"\n... 외 {len(high_confidence_indices) - 20}개\n")
        else:
            f.write("개별 고신뢰 키포인트 없음\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[6] 목(Neck) 생성 관련\n")
        f.write(f"{'='*80}\n\n")
        
        nose_idx = 0
        l_shoulder_idx = 5
        r_shoulder_idx = 6
        
        nose_score = scores[nose_idx]
        l_shoulder_score = scores[l_shoulder_idx]
        r_shoulder_score = scores[r_shoulder_idx]
        
        f.write(f"⚠️  주의: COCO-WholeBody에는 목(neck) 키포인트가 없습니다\n")
        f.write(f"현재 구현: 양쪽 어깨의 중점을 목으로 간주\n")
        f.write(f"문제점: 정면이 아닌 각도에서는 부정확함\n\n")
        
        f.write(f"목 생성에 사용되는 키포인트:\n")
        f.write(f"  - Nose (idx=0): score={nose_score:.3f}\n")
        f.write(f"  - Left Shoulder (idx=5): score={l_shoulder_score:.3f}\n")
        f.write(f"  - Right Shoulder (idx=6): score={r_shoulder_score:.3f}\n\n")
        
        neck_threshold = 0.3
        can_draw_neck = (nose_score >= neck_threshold and 
                        l_shoulder_score >= neck_threshold and 
                        r_shoulder_score >= neck_threshold)
        
        if can_draw_neck:
            f.write(f"✅ 목 라인 생성 가능 (모두 {neck_threshold} 이상)\n")
            f.write(f"   (하지만 draw_neck=False이므로 실제로는 그리지 않음)\n")
            
            nose_kpt = keypoints[nose_idx]
            l_shoulder_kpt = keypoints[l_shoulder_idx]
            r_shoulder_kpt = keypoints[r_shoulder_idx]
            neck_kpt = (l_shoulder_kpt + r_shoulder_kpt) / 2
            
            f.write(f"\n가상 목 위치: ({neck_kpt[0]:.1f}, {neck_kpt[1]:.1f})\n")
            f.write(f"  = (Left Shoulder + Right Shoulder) / 2\n")
            
            shoulder_dist = np.linalg.norm(l_shoulder_kpt - r_shoulder_kpt)
            body_height = abs(nose_kpt[1] - neck_kpt[1])
            
            f.write(f"\n어깨 간격: {shoulder_dist:.1f}px\n")
            f.write(f"코-목 거리: {body_height:.1f}px\n")
            
            if shoulder_dist < w * 0.1:
                f.write(f"⚠️  어깨 간격이 좁음 → 측면 각도 가능성\n")
                f.write(f"⚠️  목 위치가 부정확할 수 있음\n")
        else:
            f.write(f"❌ 목 라인 생성 불가 (신뢰도 부족)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[7] 결론 및 권장사항\n")
        f.write(f"{'='*80}\n\n")
        
        if is_full_body_confident:
            f.write(f"✅ 전신 확신 모드 활성\n")
            f.write(f"   → 133개 키포인트 모두 승인\n")
            f.write(f"   → Body 모델 검증 생략\n")
        else:
            f.write(f"⚠️  Cross-Filter 작동 중\n")
            f.write(f"   → Body 모델로 검증 필요\n")
            
            if suspicious_conf > 20:
                f.write(f"   → ⚠️ {suspicious_conf}개의 의심 키포인트\n")
                f.write(f"   → 할루시네이션 가능성 높음\n")
        
        f.write(f"\n")
    
    print(f"✅ 디버깅 정보 저장: {output_txt}")
    print(f"\n✨ 완료! 디버깅 정보는 {output_dir}에 저장되었습니다")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
