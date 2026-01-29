"""
Cross-Filter 디버깅 정보 생성 모듈

v2 리팩토링:
- 파일 저장 로직 제거
- 문자열만 반환하여 api.py에서 저장 제어
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from ..extractors.dwpose_extractor import DWPoseExtractor
from ..extractors import BodyExtractor
from ..extractors.keypoint_constants import BODY_KEYPOINTS


# COCO17 body keypoint 이름
BODY_17_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO-WholeBody 133 키포인트에서 17 body 키포인트 매핑
BODY_17_TO_133 = {name: idx for idx, name in enumerate(BODY_17_NAMES)}


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


def generate_debug_text(
    image_path: Path,
    dw_extractor: Optional[DWPoseExtractor] = None,
    body_extractor: Optional[BodyExtractor] = None,
    config: Optional[dict] = None
) -> Optional[str]:
    """
    Cross-Filter 디버깅 정보 문자열 생성
    
    Args:
        image_path: 분석할 이미지 경로
        dw_extractor: DWPose 추출기 (None이면 새로 생성)
        body_extractor: Body 추출기 (None이면 새로 생성)
        config: 설정 딕셔너리 (yaml_config)
    
    Returns:
        디버그 정보 문자열 (실패 시 None)
    """
    
    # 이미지 읽기
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ [Debug] 이미지를 읽을 수 없음: {image_path}")
        return None
    
    h, w = image.shape[:2]
    
    # Extractor 초기화 (필요 시)
    if dw_extractor is None:
        dw_extractor = DWPoseExtractor(backend='onnxruntime', device='cuda', mode='performance')
    
    if body_extractor is None:
        body_extractor = BodyExtractor(backend='onnxruntime', device='cpu', mode='balanced')
    
    # DWPose 추출
    try:
        dw_keypoints, dw_scores = dw_extractor.extract_single(image, person_idx=0)
    except Exception as e:
        print(f"⚠️ [Debug] DWPose 추출 실패: {e}")
        return None
    
    if dw_keypoints is None or dw_scores is None:
        print(f"⚠️ [Debug] DWPose 사람 감지 실패: {image_path.name}")
        return None
    
    # Body 모델 추출
    has_body = False
    body_keypoints = None
    body_scores = None
    try:
        body_keypoints, body_scores = body_extractor.extract_single(image, person_idx=0)
        has_body = True
    except Exception as e:
        pass  # Body 실패는 치명적이지 않음
    
    # 설정값 로드
    if config is None:
        config = {}
    
    cross_filter_cfg = config.get('cross_filter', {})
    dw_full_body_threshold = cross_filter_cfg.get('dw_full_body_confidence_threshold', 6.0)
    dw_high_threshold = cross_filter_cfg.get('dw_high_confidence_threshold', 8.0)
    dw_suspicious_threshold = cross_filter_cfg.get('dw_suspicious_threshold', 2.0)
    body_threshold = cross_filter_cfg.get('body_confidence_threshold', 0.5)
    
    # Body 17개 분석
    body_17_indices = [BODY_17_TO_133[name] for name in BODY_17_NAMES]
    body_17_scores_dw = [dw_scores[idx] for idx in body_17_indices]
    is_full_body_confident = all(score > dw_full_body_threshold for score in body_17_scores_dw)
    
    # 통계 계산
    high_confidence_indices = [i for i in range(133) if dw_scores[i] > dw_high_threshold]
    above_full_body = [i for i in range(133) if dw_scores[i] > dw_full_body_threshold]
    mid_range = [i for i in range(133) if dw_suspicious_threshold < dw_scores[i] <= dw_full_body_threshold]
    suspicious_indices = [i for i in range(133) if 0.05 < dw_scores[i] <= dw_suspicious_threshold]
    very_low = [i for i in range(133) if dw_scores[i] <= 0.05]
    
    # 문자열 빌더
    lines = []
    
    # 헤더
    lines.append(f"{'='*80}")
    lines.append(f"Cross-Filter 디버깅 정보")
    lines.append(f"{'='*80}")
    lines.append(f"")
    lines.append(f"📁 파일명: {image_path.name}")
    lines.append(f"📐 이미지 크기: {w}x{h}")
    lines.append(f"")
    
    # [1] Body vs DWPose 비교표
    lines.append(f"{'='*80}")
    lines.append(f"[1] Body vs DWPose Body 17 Keypoints 비교")
    lines.append(f"{'='*80}")
    lines.append(f"")
    lines.append("이 섹션은 Body 모델과 DWPose 모델의 Body 17개 키포인트를 비교합니다.")
    lines.append("Body 모델이 wrist를 정확히 감지해도 DWPose 손가락이 할루시네이션일 수 있습니다.")
    lines.append(f"")
    
    if has_body:
        lines.append(f"{'No':<5}{'Name':<19}{'Body Conf':<13}{'DWPose Conf':<13}{'차이':<13}{'상태'}")
        lines.append(f"{'-'*80}")
        
        for i, name in enumerate(BODY_17_NAMES):
            dw_idx = body_17_indices[i]
            body_conf = body_scores[i]
            dw_conf = dw_scores[dw_idx]
            diff = dw_conf - body_conf
            
            # 상태 판정
            body_high = body_conf > body_threshold
            dw_high = dw_conf > dw_full_body_threshold
            
            if body_high and dw_high:
                status = "✅ 양쪽 높음"
            elif body_high and not dw_high:
                status = "⚠️ Body만 높음"
            elif not body_high and dw_high:
                status = "⚠️ DW만 높음"
            else:
                status = "❌ 양쪽 낮음"
            
            diff_str = f"{diff:.3f}" + "+"*min(5, int(abs(diff)))
            
            lines.append(f"{i:<5}{name:<19}{body_conf:<13.3f}{dw_conf:<13.3f}{diff_str:<13}{status}")
        
        lines.append(f"")
        lines.append(f"💡 해석:")
        lines.append(f"  - Body Conf: Body 모델(YOLO) confidence (0~1 범위, Sigmoid 출력)")
        lines.append(f"  - DWPose Conf: DWPose confidence (2.8~8.0+ 범위, SimCC 로그 확률)")
        lines.append(f"  - 차이: DWPose - Body (단위가 다르므로 직접 비교 불가)")
        lines.append(f"  - '⚠️ DW만 높음' = Body는 낮은데 DWPose 높음 → 할루시네이션 의심")
        lines.append(f"")
    else:
        lines.append("⚠️ Body 모델 추출 실패")
        lines.append(f"")
    
    # [2] 전신 확신 모드 체크
    lines.append(f"{'='*80}")
    lines.append(f"[2] 전신 확신 모드 체크")
    lines.append(f"{'='*80}")
    lines.append(f"")
    lines.append(f"전신 확신 임계값 (dw_full_body_confidence_threshold): {dw_full_body_threshold}")
    
    if is_full_body_confident:
        lines.append(f"전신 확신 모드: ✅ 활성")
        lines.append(f"→ 모든 body keypoints가 {dw_full_body_threshold} 이상")
        lines.append(f"→ Cross-Filter 바이패스 (Body 검증 생략)")
        lines.append(f"")
    else:
        lines.append(f"전신 확신 모드: ❌ 비활성")
        lines.append(f"→ 일부 body keypoints가 {dw_full_body_threshold} 미만")
        lines.append(f"→ Cross-Filter 정상 작동 (Body 모델 검증 필요)")
        lines.append(f"")
    
    # [3] Body 17 상세
    lines.append(f"{'='*80}")
    lines.append(f"[3] DWPose Body 17 Keypoints 상세")
    lines.append(f"{'='*80}")
    lines.append(f"")
    lines.append(f"{'No':<5}{'Name':<19}{'Index':<7}{'X':<9}{'Y':<9}{'Score':<9}{'Status'}")
    lines.append(f"{'-'*80}")
    
    for i, name in enumerate(BODY_17_NAMES):
        idx = body_17_indices[i]
        x, y = dw_keypoints[idx]
        score = dw_scores[idx]
        
        if score > dw_high_threshold:
            status = "✅ High"
        elif score > dw_full_body_threshold:
            status = "✅ Full"
        elif score > dw_suspicious_threshold:
            status = "⚠️ Mid"
        else:
            status = "❌ Low"
        
        lines.append(f"{i:<5}{name:<19}{idx:<7}{x:<9.1f}{y:<9.1f}{score:<9.3f}{status}")
    
    lines.append(f"")
    
    # [4] 전체 133 Keypoints 통계
    lines.append(f"{'='*80}")
    lines.append(f"[4] 전체 133 Keypoints 통계")
    lines.append(f"{'='*80}")
    lines.append(f"")
    lines.append(f"전체 키포인트: 133")
    lines.append(f"  > {dw_high_threshold} (dw_high_confidence_threshold - 개별 바이패스): {len(high_confidence_indices)} 개")
    lines.append(f"  > {dw_full_body_threshold} (dw_full_body_confidence_threshold - 전신 바이패스): {len(above_full_body)} 개")
    lines.append(f"  {dw_suspicious_threshold} ~ {dw_full_body_threshold} (중간 신뢰도): {len(mid_range)} 개")
    lines.append(f"  0.05 ~ {dw_suspicious_threshold} (dw_suspicious_threshold - 의심 영역): {len(suspicious_indices)} 개  ⚠️ 할루시네이션 가능")
    lines.append(f"  ≤ 0.05 (dw_min_confidence - 매우 낮음): {len(very_low)} 개")
    lines.append(f"")
    
    # [4-1] 발가락 DWPose Confidence 리스트
    lines.append(f"{'='*80}")
    lines.append(f"[4-1] 발가락 DWPose Confidence 상세 (인덱스 17-22)")
    lines.append(f"{'='*80}")
    lines.append(f"")
    
    foot_indices = list(range(17, 23))
    foot_names = [
        'left_big_toe', 'left_small_toe', 'left_heel',
        'right_big_toe', 'right_small_toe', 'right_heel'
    ]
    
    lines.append(f"{'Index':<8}{'Name':<20}{'X':<10}{'Y':<10}{'Confidence':<12}{'상태'}")
    lines.append(f"{'-'*80}")
    
    foot_confidences = []
    for i, idx in enumerate(foot_indices):
        x, y = dw_keypoints[idx]
        score = dw_scores[idx]
        foot_confidences.append(score)
        
        # foot_dw_min_confidence 기준으로 상태 표시
        foot_threshold = cross_filter_cfg.get('foot_dw_min_confidence', 4.0)
        if score > foot_threshold:
            status = "✅ 통과"
        elif score > dw_suspicious_threshold:
            status = "⚠️  경계 (suspicious~foot_min 사이)"
        else:
            status = "❌ 의심 (suspicious 이하)"
        
        lines.append(f"{idx:<8}{foot_names[i]:<20}{x:<10.1f}{y:<10.1f}{score:<12.3f}{status}")
    
    lines.append(f"")
    lines.append(f"📊 발가락 통계:")
    lines.append(f"   평균: {np.mean(foot_confidences):.3f}")
    lines.append(f"   최소: {np.min(foot_confidences):.3f}")
    lines.append(f"   최대: {np.max(foot_confidences):.3f}")
    lines.append(f"   중앙값: {np.median(foot_confidences):.3f}")
    lines.append(f"   foot_dw_min_confidence 기준: {cross_filter_cfg.get('foot_dw_min_confidence', 4.0)}")
    lines.append(f"   → {sum(1 for s in foot_confidences if s > cross_filter_cfg.get('foot_dw_min_confidence', 4.0))}/6 개 통과")
    lines.append(f"")
    
    # [4-2] 손가락 DWPose Confidence 리스트
    lines.append(f"{'='*80}")
    lines.append(f"[4-2] 손가락 DWPose Confidence 상세 (인덱스 91-133)")
    lines.append(f"{'='*80}")
    lines.append(f"")
    
    lines.append(f"◆ 왼손 (91-112, 21개):")
    lines.append(f"")
    lines.append(f"{'Index':<8}{'X':<10}{'Y':<10}{'Confidence':<12}{'상태'}")
    lines.append(f"{'-'*80}")
    
    left_hand_indices = list(range(91, 112))
    left_hand_confidences = []
    for idx in left_hand_indices:
        x, y = dw_keypoints[idx]
        score = dw_scores[idx]
        left_hand_confidences.append(score)
        
        if score > dw_full_body_threshold:
            status = "✅ 고신뢰"
        elif score > dw_suspicious_threshold:
            status = "⚠️  중간"
        else:
            status = "❌ 의심"
        
        lines.append(f"{idx:<8}{x:<10.1f}{y:<10.1f}{score:<12.3f}{status}")
    
    lines.append(f"")
    lines.append(f"📊 왼손 통계:")
    lines.append(f"   평균: {np.mean(left_hand_confidences):.3f}")
    lines.append(f"   최소: {np.min(left_hand_confidences):.3f}")
    lines.append(f"   최대: {np.max(left_hand_confidences):.3f}")
    lines.append(f"   중앙값: {np.median(left_hand_confidences):.3f}")
    lines.append(f"   suspicious 기준 ({dw_suspicious_threshold}) 이하: {sum(1 for s in left_hand_confidences if s <= dw_suspicious_threshold)}/21 개")
    lines.append(f"")
    
    lines.append(f"◆ 오른손 (112-133, 21개):")
    lines.append(f"")
    lines.append(f"{'Index':<8}{'X':<10}{'Y':<10}{'Confidence':<12}{'상태'}")
    lines.append(f"{'-'*80}")
    
    right_hand_indices = list(range(112, 133))
    right_hand_confidences = []
    for idx in right_hand_indices:
        x, y = dw_keypoints[idx]
        score = dw_scores[idx]
        right_hand_confidences.append(score)
        
        if score > dw_full_body_threshold:
            status = "✅ 고신뢰"
        elif score > dw_suspicious_threshold:
            status = "⚠️  중간"
        else:
            status = "❌ 의심"
        
        lines.append(f"{idx:<8}{x:<10.1f}{y:<10.1f}{score:<12.3f}{status}")
    
    lines.append(f"")
    lines.append(f"📊 오른손 통계:")
    lines.append(f"   평균: {np.mean(right_hand_confidences):.3f}")
    lines.append(f"   최소: {np.min(right_hand_confidences):.3f}")
    lines.append(f"   최대: {np.max(right_hand_confidences):.3f}")
    lines.append(f"   중앙값: {np.median(right_hand_confidences):.3f}")
    lines.append(f"   suspicious 기준 ({dw_suspicious_threshold}) 이하: {sum(1 for s in right_hand_confidences if s <= dw_suspicious_threshold)}/21 개")
    lines.append(f"")
    
    # [5] 의심 키포인트
    lines.append(f"{'='*80}")
    lines.append(f"[5] 의심스러운 키포인트 (0.05 ~ {dw_suspicious_threshold}) - Clean Mode 판정")
    lines.append(f"{'='*80}")
    lines.append(f"")
    
    if len(suspicious_indices) > 0:
        lines.append(f"총 {len(suspicious_indices)}개의 의심 키포인트 발견")
        lines.append(f"→ suspicious_count > 0 → Normal Mode 작동")
        lines.append(f"→ 이들은 할루시네이션일 가능성이 높습니다")
        lines.append(f"")
        lines.append(f"{'Index':<9}{'X':<11}{'Y':<11}{'Score':<11}{'정확한 이름'}")
        lines.append(f"{'-'*80}")
        for idx in suspicious_indices:
            x, y = dw_keypoints[idx]
            score = dw_scores[idx]
            name = get_keypoint_name(idx)
            lines.append(f"{idx:<9}{x:<11.1f}{y:<11.1f}{score:<11.3f}{name}")
        lines.append(f"")
    else:
        lines.append(f"의심 키포인트 없음")
        lines.append(f"→ suspicious_count == 0 → Clean Mode 활성")
        lines.append(f"→ body_confidence_threshold가 0.3에서 0.2로 완화됨")
        lines.append(f"")
    
    # [6] 고신뢰 키포인트
    lines.append(f"{'='*80}")
    lines.append(f"[6] 개별 고신뢰 키포인트 (> {dw_high_threshold}) - Individual Bypass")
    lines.append(f"{'='*80}")
    lines.append(f"")
    
    if len(high_confidence_indices) > 0:
        lines.append(f"총 {len(high_confidence_indices)}개의 고신뢰 키포인트")
        lines.append(f"→ 이들은 Body 검증 없이 승인됩니다")
        lines.append(f"")
        lines.append(f"{'Index':<9}{'X':<11}{'Y':<11}{'Score':<11}{'정확한 이름'}")
        lines.append(f"{'-'*80}")
        for idx in high_confidence_indices:
            x, y = dw_keypoints[idx]
            score = dw_scores[idx]
            name = get_keypoint_name(idx)
            lines.append(f"{idx:<9}{x:<11.1f}{y:<11.1f}{score:<11.3f}{name}")
        lines.append(f"")
    else:
        lines.append(f"개별 고신뢰 키포인트 없음")
        lines.append(f"")
    
    # [7] 결론 및 손가락 할루시네이션 체크
    lines.append(f"{'='*80}")
    lines.append(f"[7] 결론 및 권장사항")
    lines.append(f"{'='*80}")
    lines.append(f"")
    
    if is_full_body_confident:
        lines.append(f"✅ 전신 확신 모드 활성")
        lines.append(f"   → Cross-Filter 바이패스")
        lines.append(f"   → 모든 키포인트 신뢰도 높음")
        lines.append(f"")
    else:
        lines.append(f"⚠️  Cross-Filter 작동 중")
        lines.append(f"   → Body 모델로 검증 필요")
        lines.append(f"   → ⚠️ {len(suspicious_indices)}개의 의심 키포인트 ({'Clean' if len(suspicious_indices) == 0 else 'Normal'} Mode)")
        if len(suspicious_indices) > 0:
            lines.append(f"   → 할루시네이션 가능성 높음")
        lines.append(f"")
    
    # 손가락 할루시네이션 체크
    if has_body:
        lines.append(f"💡 Body vs DWPose 비교 분석:")
        
        # 왼손 체크
        left_wrist_body_conf = body_scores[9]  # left_wrist
        left_hand_indices = list(range(91, 112))  # 왼손 21개
        left_suspicious = [idx for idx in left_hand_indices if idx in suspicious_indices]
        
        if left_wrist_body_conf > body_threshold and len(left_suspicious) > 0:
            lines.append(f"   ⚠️ 왼손: Body wrist 신뢰={left_wrist_body_conf:.3f} (높음)")
            lines.append(f"      하지만 DWPose 손가락 중 {len(left_suspicious)}개가 의심 범위")
            lines.append(f"      → 손가락 할루시네이션 가능성!")
        
        # 오른손 체크
        right_wrist_body_conf = body_scores[10]  # right_wrist
        right_hand_indices = list(range(112, 133))  # 오른손 21개
        right_suspicious = [idx for idx in right_hand_indices if idx in suspicious_indices]
        
        if right_wrist_body_conf > body_threshold and len(right_suspicious) > 0:
            lines.append(f"   ⚠️ 오른손: Body wrist 신뢰={right_wrist_body_conf:.3f} (높음)")
            lines.append(f"      하지만 DWPose 손가락 중 {len(right_suspicious)}개가 의심 범위")
            lines.append(f"      → 손가락 할루시네이션 가능성!")
        
        if len(left_suspicious) == 0 and len(right_suspicious) == 0:
            lines.append(f"   ✅ Body와 DWPose 일치성 양호")
        
        lines.append(f"")
    
    return "\n".join(lines)


# ============================================================
# Legacy 호환: 기존 generate_debug_info 인터페이스 유지
# ============================================================
def generate_debug_info(
    image_path: Path,
    output_dir: Path,
    dw_extractor: Optional[DWPoseExtractor] = None,
    body_extractor: Optional[BodyExtractor] = None,
    config: Optional[dict] = None
) -> Optional[str]:
    """
    [DEPRECATED] Legacy 호환용 - 직접 파일 저장
    
    새로운 코드에서는 generate_debug_text()를 사용하고
    api.py에서 파일 저장을 제어하세요.
    
    Returns:
        생성된 debug 파일 경로 (실패 시 None)
    """
    debug_text = generate_debug_text(
        image_path=image_path,
        dw_extractor=dw_extractor,
        body_extractor=body_extractor,
        config=config
    )
    
    if debug_text is None:
        return None
    
    # 출력 파일명
    output_path = output_dir / f"{image_path.stem}_debug.txt"
    
    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(debug_text)
    
    return str(output_path)
