"""
포즈 전이 테스트 스크립트 (디버그 포함)
"""
import sys
import numpy as np
from pathlib import Path

from pose_transfer.pipeline import PipelineConfig, PoseTransferPipeline
from pose_transfer.utils.io import load_image, save_image, save_json


def analyze_keypoints(name: str, keypoints: np.ndarray, scores: np.ndarray, threshold: float = 0.3):
    """키포인트 유효성 분석"""
    print(f"\n[{name}] 키포인트 분석:")
    
    # 부위별 인덱스
    regions = {
        'Body (0-16)': (0, 17),
        'Feet (17-22)': (17, 23),
        'Face (23-90)': (23, 91),
        'Left Hand (91-111)': (91, 112),
        'Right Hand (112-132)': (112, 133),
    }
    
    total_valid = 0
    for region_name, (start, end) in regions.items():
        region_scores = scores[start:end]
        valid_count = np.sum(region_scores > threshold)
        total_count = end - start
        pct = valid_count / total_count * 100
        status = "✅" if valid_count > total_count * 0.5 else "⚠️" if valid_count > 0 else "❌"
        print(f"  {status} {region_name}: {valid_count}/{total_count} ({pct:.0f}%)")
        total_valid += valid_count
    
    print(f"  총 유효 키포인트: {total_valid}/133")
    return total_valid


def check_compatibility(source_scores: np.ndarray, ref_scores: np.ndarray, threshold: float = 0.3):
    """소스-레퍼런스 호환성 체크"""
    print("\n" + "="*50)
    print("[호환성 분석]")
    print("="*50)
    
    # 주요 본 정의
    key_bones = [
        ('left_shoulder_left_elbow', 5, 7),
        ('left_elbow_left_wrist', 7, 9),
        ('right_shoulder_right_elbow', 6, 8),
        ('right_elbow_right_wrist', 8, 10),
        ('left_hip_left_knee', 11, 13),
        ('left_knee_left_ankle', 13, 15),
        ('right_hip_right_knee', 12, 14),
        ('right_knee_right_ankle', 14, 16),
    ]
    
    warnings = []
    for bone_name, start_idx, end_idx in key_bones:
        src_valid = source_scores[start_idx] > threshold and source_scores[end_idx] > threshold
        ref_valid = ref_scores[start_idx] > threshold and ref_scores[end_idx] > threshold
        
        if src_valid and ref_valid:
            status = "✅ 정상"
        elif src_valid and not ref_valid:
            status = "⚠️ Reference 없음 → 방향 폴백"
            warnings.append(f"{bone_name}: Reference에 없음")
        elif not src_valid and ref_valid:
            status = "⚠️ Source 없음 → 길이 폴백"
            warnings.append(f"{bone_name}: Source에 없음")
        else:
            status = "❌ 둘 다 없음"
            warnings.append(f"{bone_name}: 둘 다 없음")
        
        print(f"  {bone_name}: {status}")
    
    if warnings:
        print(f"\n⚠️ 경고 {len(warnings)}개:")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("\n✅ 모든 주요 본 호환!")
    
    return warnings


def main():
    import argparse
    parser = argparse.ArgumentParser(description='포즈 전이 테스트')
    parser.add_argument('--source', type=str, default='inputs/source.jpg', help='원본 이미지 (비율)')
    parser.add_argument('--reference', type=str, default='inputs/reference.jpg', help='레퍼런스 이미지 (포즈)')
    parser.add_argument('--output', type=str, default='outputs', help='출력 폴더')
    parser.add_argument('--config', type=str, default=None, help='설정 파일')
    args = parser.parse_args()
    
    # 경로 확인
    source_path = Path(args.source)
    ref_path = Path(args.reference)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not source_path.exists():
        print(f"❌ Source 이미지 없음: {source_path}")
        sys.exit(1)
    if not ref_path.exists():
        print(f"❌ Reference 이미지 없음: {ref_path}")
        sys.exit(1)
    
    print("="*50)
    print("포즈 전이 테스트")
    print("="*50)
    print(f"Source: {source_path}")
    print(f"Reference: {ref_path}")
    print(f"Output: {output_dir}")
    
    # 설정 로드
    config_path = args.config or Path(__file__).parent / "pose_transfer/config/default.yaml"
    if Path(config_path).exists():
        print(f"Config: {config_path}")
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        print("Config: 기본값 사용")
        config = PipelineConfig()
    
    # 파이프라인 초기화
    pipeline = PoseTransferPipeline(config)
    
    # 1. 각각 키포인트 추출
    print("\n" + "="*50)
    print("[Step 1] 키포인트 추출")
    print("="*50)
    
    source_kpts, source_scores, _, source_size = pipeline.extract_pose(source_path)
    ref_kpts, ref_scores, _, ref_size = pipeline.extract_pose(ref_path)
    
    print(f"\nSource 이미지 크기: {source_size[1]}x{source_size[0]}")
    print(f"Reference 이미지 크기: {ref_size[1]}x{ref_size[0]}")
    
    # 2. 키포인트 분석
    analyze_keypoints("Source", source_kpts, source_scores, config.kpt_threshold)
    analyze_keypoints("Reference", ref_kpts, ref_scores, config.kpt_threshold)
    
    # 3. 호환성 체크
    warnings = check_compatibility(source_scores, ref_scores, config.kpt_threshold)
    
    # 4. 포즈 전이 실행
    print("\n" + "="*50)
    print("[Step 2] 포즈 전이 실행")
    print("="*50)
    
    result = pipeline.transfer(source_path, ref_path)
    
    # 전이 로그 출력
    print("\n[전이 로그]")
    methods = {}
    for kpt_name, method in result.processing_info.get('transfer_log', {}).items():
        methods[method] = methods.get(method, 0) + 1
    for method, count in methods.items():
        print(f"  {method}: {count}개")
    
    # 5. 결과 저장
    print("\n" + "="*50)
    print("[Step 3] 결과 저장")
    print("="*50)
    
    # JSON
    json_path = output_dir / "transferred_keypoints.json"
    save_json(result.to_json(), str(json_path))
    print(f"✅ {json_path}")
    
    # 스켈레톤 이미지
    skeleton_path = output_dir / "transferred_skeleton.png"
    save_image(result.skeleton_image, str(skeleton_path))
    print(f"✅ {skeleton_path}")
    
    # 원본 위에 전이된 포즈 오버레이
    source_img = load_image(source_path)
    overlay = pipeline.renderer.render(source_img, result.transferred_keypoints, result.transferred_scores)
    overlay_path = output_dir / "transferred_overlay.png"
    save_image(overlay, str(overlay_path))
    print(f"✅ {overlay_path}")
    
    print("\n" + "="*50)
    print("✅ 포즈 전이 완료!")
    print("="*50)


if __name__ == "__main__":
    main()

def clip_invalid_keypoints(kpts, scores, img_size, margin=0.1):
    """이미지 범위 밖 키포인트 무효화"""
    h, w = img_size
    max_x = w * (1 + margin)
    max_y = h * (1 + margin)
    
    for i in range(len(kpts)):
        x, y = kpts[i]
        if x < -w * margin or x > max_x or y < -h * margin or y > max_y:
            scores[i] = 0
            print(f"  [CLIP] 키포인트 {i}: ({x:.0f}, {y:.0f}) → 무효화")
    
    return kpts, scores
