"""
Cross-Filter 디버깅: 각 이미지별 DWPose 분석 및 필터링 상태
Body 모델과 DWPose 비교 분석 포함
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path

from pose_transfer.extractors.dwpose_extractor import DWPoseExtractor
from pose_transfer.extractors import BodyExtractor  # Body 모델 추가
from pose_transfer.logic.cross_filter import CrossFilter, CrossFilterConfig
from pose_transfer.extractors.keypoint_constants import BODY_KEYPOINTS


# COCO17 body keypoint 이름 (순서)
BODY_17_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# COCO-WholeBody 133 키포인트에서 17 body 키포인트 매핑
BODY_17_TO_133 = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def get_keypoint_name(idx: int) -> str:
    """키포인트 인덱스에서 정확한 이름 반환"""
    if idx < 17:
        return f"Body - {BODY_17_NAMES[idx]}"
    elif 17 <= idx < 23:
        foot_names = ['left_foot_1', 'left_foot_2', 'left_foot_3', 
                      'right_foot_1', 'right_foot_2', 'right_foot_3']
        return f"Feet - {foot_names[idx - 17]}"
    elif 23 <= idx < 91:
        return f"Face - landmark_{idx - 23 + 1}"
    elif 91 <= idx < 112:
        return f"Left Hand - finger_{idx - 91 + 1}"
    else:
        return f"Right Hand - finger_{idx - 112 + 1}"


def analyze_image(image_path: str, output_dir: str):
    """이미지 분석 및 디버깅 정보 생성 (Body + DWPose 비교)"""
    
    print(f"\n{'='*80}")
    print(f"분석 중: {os.path.basename(image_path)}")
    print('='*80)
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 읽을 수 없음: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"📐 이미지 크기: {w}x{h}")
    
    # DWPose 초기화
    dwpose = DWPoseExtractor(
        backend='onnxruntime',
        device='cuda',
        mode='performance'
    )
    
    # Body 모델 초기화
    body_extractor = BodyExtractor(
        backend='onnxruntime',
        device='cpu',  # Body는 CPU로 실행 (default.yaml 설정)
        mode='balanced'
    )
    
    # DWPose 추출 (extract_single 사용)
    try:
        dw_keypoints, dw_scores = dwpose.extract_single(image, person_idx=0)
    except Exception as e:
        print(f"❌ DWPose 추출 실패: {e}")
        return
    
    if dw_keypoints is None or dw_scores is None:
        print("❌ DWPose 추출 실패 (사람 감지 안됨)")
        return
    
    # Body 모델 추출
    try:
        body_keypoints, body_scores = body_extractor.extract_single(image, person_idx=0)
        has_body = True
        print(f"✅ Body 모델 추출 성공 (17 keypoints)")
    except Exception as e:
        print(f"⚠️ Body 모델 추출 실패: {e}")
        has_body = False
        body_keypoints = None
        body_scores = None
    
    # 17개 body keypoints 추출
    body_17_indices = [BODY_17_TO_133[name] for name in BODY_17_NAMES]
    
    # 전신 확신 모드 체크
    dw_full_body_threshold = 6.0  # default.yaml 설정
    body_17_scores_dw = [dw_scores[idx] for idx in body_17_indices]
    is_full_body_confident = all(score > dw_full_body_threshold for score in body_17_scores_dw)
    
    # 개별 고신뢰 키포인트 체크 (8.0 이상)
    high_confidence_indices = [i for i in range(133) if dw_scores[i] > 8.0]
    
    # 낮은 신뢰도 키포인트 (0.05 ~ 2.0 사이 - 의심스러운 범위)
    suspicious_indices = [i for i in range(133) if 0.05 < dw_scores[i] <= 2.0]
    
    # 출력 파일명 생성
    base_name = Path(image_path).stem
    output_txt = os.path.join(output_dir, f"{base_name}_debug.txt")
    
    # 디버깅 정보 작성
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Cross-Filter 디버깅 정보\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"📁 파일명: {os.path.basename(image_path)}\n")
        f.write(f"📐 이미지 크기: {w}x{h}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"[1] Body vs DWPose Body 17 Keypoints 비교\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"이 섹션은 Body 모델과 DWPose 모델의 Body 17개 키포인트를 비교합니다.\n")
        f.write(f"Body 모델이 wrist를 정확히 감지해도 DWPose 손가락이 할루시네이션일 수 있습니다.\n\n")
        
        if has_body:
            f.write(f"{'No':<4} {'Name':<18} {'Body Conf':<12} {'DWPose Conf':<12} {'차이':<10} {'상태'}\n")
            f.write(f"{'-'*80}\n")
            
            for i, name in enumerate(BODY_17_NAMES):
                idx = BODY_17_TO_133[name]
                body_conf = body_scores[i]
                dw_conf = dw_scores[idx]
                diff = dw_conf - body_conf
                
                # 상태 판정
                if body_conf > 0.5 and dw_conf > 6.0:
                    status = "✅ 양쪽 높음"
                elif body_conf > 0.5 and dw_conf <= 6.0:
                    status = "⚠️ Body만 높음"
                elif body_conf <= 0.5 and dw_conf > 6.0:
                    status = "⚠️ DW만 높음"
                else:
                    status = "❌ 양쪽 낮음"
                
                f.write(f"{i:<4} {name:<18} {body_conf:<12.3f} {dw_conf:<12.3f} {diff:+<10.3f} {status}\n")
            
            f.write(f"\n💡 해석:\n")
            f.write(f"  - Body Conf: Body 모델(YOLO) confidence (0~1 범위, Sigmoid 출력)\n")
            f.write(f"  - DWPose Conf: DWPose confidence (2.8~8.0+ 범위, SimCC 로그 확률)\n")
            f.write(f"  - 차이: DWPose - Body (단위가 다르므로 직접 비교 불가)\n")
            f.write(f"  - '⚠️ DW만 높음' = Body는 낮은데 DWPose 높음 → 할루시네이션 의심\n")
        else:
            f.write(f"❌ Body 모델 결과 없음\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[2] 전신 확신 모드 체크\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"전신 확신 임계값 (dw_full_body_confidence_threshold): {dw_full_body_threshold}\n")
        f.write(f"전신 확신 모드: {'✅ 활성' if is_full_body_confident else '❌ 비활성'}\n")
        
        if is_full_body_confident:
            f.write(f"→ 모든 17개 body keypoints가 {dw_full_body_threshold} 이상\n")
            f.write(f"→ Body 모델 검증 없이 전체 승인 (Full-Body Bypass)\n")
        else:
            f.write(f"→ 일부 body keypoints가 {dw_full_body_threshold} 미만\n")
            f.write(f"→ Cross-Filter 정상 작동 (Body 모델 검증 필요)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[3] DWPose Body 17 Keypoints 상세\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"{'No':<4} {'Name':<18} {'Index':<6} {'X':<8} {'Y':<8} {'Score':<8} {'Status'}\n")
        f.write(f"{'-'*80}\n")
        
        for i, name in enumerate(BODY_17_NAMES):
            idx = BODY_17_TO_133[name]
            kpt = dw_keypoints[idx]
            score = dw_scores[idx]
            
            # 상태 판정
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
        f.write(f"[4] 전체 133 Keypoints 통계\n")
        f.write(f"{'='*80}\n\n")
        
        total_kpts = len(dw_scores)
        high_conf = sum(1 for s in dw_scores if s > 8.0)
        full_body_conf = sum(1 for s in dw_scores if s > 6.0)
        medium_conf = sum(1 for s in dw_scores if 2.0 < s <= 6.0)
        suspicious_conf = sum(1 for s in dw_scores if 0.05 < s <= 2.0)
        low_conf = sum(1 for s in dw_scores if s <= 0.05)
        
        f.write(f"전체 키포인트: {total_kpts}\n")
        f.write(f"  > 8.0 (dw_high_confidence_threshold - 개별 바이패스): {high_conf} 개\n")
        f.write(f"  > 6.0 (dw_full_body_confidence_threshold - 전신 바이패스): {full_body_conf} 개\n")
        f.write(f"  2.0 ~ 6.0 (중간 신뢰도): {medium_conf} 개\n")
        f.write(f"  0.05 ~ 2.0 (dw_suspicious_threshold - 의심 영역): {suspicious_conf} 개  ⚠️ 할루시네이션 가능\n")
        f.write(f"  ≤ 0.05 (dw_min_confidence - 매우 낮음): {low_conf} 개\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[5] 의심스러운 키포인트 (0.05 ~ 2.0) - Clean Mode 판정\n")
        f.write(f"{'='*80}\n\n")
        
        if suspicious_indices:
            f.write(f"총 {len(suspicious_indices)}개의 의심 키포인트 발견\n")
            f.write(f"→ suspicious_count > 0 → Normal Mode 작동\n")
            f.write(f"→ 이들은 할루시네이션일 가능성이 높습니다\n\n")
            
            f.write(f"{'Index':<8} {'X':<10} {'Y':<10} {'Score':<10} {'정확한 이름'}\n")
            f.write(f"{'-'*80}\n")
            
            for idx in suspicious_indices:
                kpt = dw_keypoints[idx]
                score = dw_scores[idx]
                kpt_name = get_keypoint_name(idx)
                
                f.write(f"{idx:<8} {kpt[0]:<10.1f} {kpt[1]:<10.1f} {score:<10.3f} {kpt_name}\n")
        else:
            f.write(f"의심 키포인트 없음\n")
            f.write(f"→ suspicious_count == 0 → Clean Mode 활성\n")
            f.write(f"→ body_confidence_threshold: 0.3 → clean_mode_body_threshold: 0.2로 완화\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[6] 개별 고신뢰 키포인트 (> 8.0) - Individual Bypass\n")
        f.write(f"{'='*80}\n\n")
        
        if high_confidence_indices:
            f.write(f"총 {len(high_confidence_indices)}개의 고신뢰 키포인트\n")
            f.write(f"→ 이들은 Body 검증 없이 무조건 승인됩니다 (Individual Bypass)\n\n")
            
            f.write(f"{'Index':<8} {'X':<10} {'Y':<10} {'Score':<10} {'정확한 이름'}\n")
            f.write(f"{'-'*80}\n")
            
            for idx in high_confidence_indices:
                kpt = dw_keypoints[idx]
                score = dw_scores[idx]
                kpt_name = get_keypoint_name(idx)
                
                f.write(f"{idx:<8} {kpt[0]:<10.1f} {kpt[1]:<10.1f} {score:<10.3f} {kpt_name}\n")
        else:
            f.write(f"개별 고신뢰 키포인트 없음\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"[7] 결론 및 권장사항\n")
        f.write(f"{'='*80}\n\n")
        
        if is_full_body_confident:
            f.write(f"✅ 전신 확신 모드 활성 (Full-Body Bypass)\n")
            f.write(f"   → 133개 키포인트 모두 승인\n")
            f.write(f"   → Body 모델 검증 생략\n")
        else:
            f.write(f"⚠️  Cross-Filter 작동 중\n")
            f.write(f"   → Body 모델로 검증 필요\n")
            
            if suspicious_conf > 20:
                f.write(f"   → ⚠️ {suspicious_conf}개의 의심 키포인트 (Normal Mode)\n")
                f.write(f"   → 할루시네이션 가능성 높음\n")
            elif suspicious_conf == 0:
                f.write(f"   → ✅ 의심 키포인트 없음 (Clean Mode)\n")
                f.write(f"   → Body threshold 0.3 → 0.2로 완화\n")
            else:
                f.write(f"   → {suspicious_conf}개의 의심 키포인트 (Normal Mode)\n")
        
        if has_body:
            f.write(f"\n💡 Body vs DWPose 비교 분석:\n")
            # wrist 확인
            left_wrist_idx = 9
            right_wrist_idx = 10
            left_wrist_body = body_scores[BODY_17_NAMES.index('left_wrist')]
            right_wrist_body = body_scores[BODY_17_NAMES.index('right_wrist')]
            left_wrist_dw = dw_scores[left_wrist_idx]
            right_wrist_dw = dw_scores[right_wrist_idx]
            
            # 손가락 확인 (91-132)
            left_hand_indices = range(91, 112)
            right_hand_indices = range(112, 133)
            left_hand_suspicious = [idx for idx in left_hand_indices if 0.05 < dw_scores[idx] <= 2.0]
            right_hand_suspicious = [idx for idx in right_hand_indices if 0.05 < dw_scores[idx] <= 2.0]
            
            if left_wrist_body > 0.5 and len(left_hand_suspicious) > 0:
                f.write(f"   ⚠️ 왼손: Body wrist 신뢰={left_wrist_body:.3f} (높음)\n")
                f.write(f"      하지만 DWPose 손가락 중 {len(left_hand_suspicious)}개가 의심 범위\n")
                f.write(f"      → 손가락 할루시네이션 가능성!\n")
            
            if right_wrist_body > 0.5 and len(right_hand_suspicious) > 0:
                f.write(f"   ⚠️ 오른손: Body wrist 신뢰={right_wrist_body:.3f} (높음)\n")
                f.write(f"      하지만 DWPose 손가락 중 {len(right_hand_suspicious)}개가 의심 범위\n")
                f.write(f"      → 손가락 할루시네이션 가능성!\n")
        
        f.write(f"\n")
    
    print(f"✅ 디버깅 정보 저장: {output_txt}")


def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python debug_cross_filter.py <input_path> [output_dir]")
        print("  <input_path>: 이미지 파일 또는 폴더")
        print("예시: python debug_cross_filter.py test_io/inputs test_io/debug_outputs")
        print("예시: python debug_cross_filter.py test_io/inputs/half_bg_2.jpg test_io/debug_outputs")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_io/debug_outputs"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력이 파일인지 폴더인지 확인
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    if os.path.isfile(input_path):
        # 단일 파일
        if Path(input_path).suffix in image_extensions:
            image_files.append(input_path)
        else:
            print(f"❌ 지원하지 않는 파일 형식: {input_path}")
            sys.exit(1)
    elif os.path.isdir(input_path):
        # 폴더
        for file in os.listdir(input_path):
            if Path(file).suffix in image_extensions:
                image_files.append(os.path.join(input_path, file))
    else:
        print(f"❌ 존재하지 않는 경로: {input_path}")
        sys.exit(1)
    
    if not image_files:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    print(f"📁 입력: {input_path}")
    print(f"📁 출력 폴더: {output_dir}")
    print(f"🖼️  발견된 이미지: {len(image_files)}개")
    
    # 각 이미지 분석
    for image_path in sorted(image_files):
        try:
            analyze_image(image_path, output_dir)
        except Exception as e:
            print(f"❌ 오류 발생 ({os.path.basename(image_path)}): {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✨ 완료! 디버깅 정보는 {output_dir}에 저장되었습니다")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
