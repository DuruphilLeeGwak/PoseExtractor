"""
Pose Transfer Batch Test Script (Auto Clean)
- 목적: test_Inputs 폴더 내의 모든 이미지를 일괄 테스트
- 기능:
  1. 시작 시 기존 output 폴더 삭제 후 재생성 (Clean Start)
  2. 폴더 내 모든 이미지에 대해 키포인트 분석 (Reference 없을 때)
  3. 폴더 내 모든 이미지에 특정 Reference 포즈 전이 (Reference 있을 때)
"""
import sys
import yaml
import shutil  # [NEW] 폴더 삭제용
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional

# 패키지 임포트
from pose_transfer.pipeline import PipelineConfig, PoseTransferPipeline
from pose_transfer.utils.io import save_json, save_image, load_image, convert_to_openpose_format

# 이미지 확장자 목록
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_image_files(directory: Path) -> List[Path]:
    """폴더 내 이미지 파일 목록 반환"""
    return [
        p for p in directory.iterdir() 
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]

def analyze_keypoints(name: str, scores: np.ndarray, threshold: float = 0.3):
    """키포인트 유효성 요약 출력"""
    total = len(scores)
    valid = np.sum(scores > threshold)
    pct = (valid / total) * 100
    print(f"   📊 [{name}] Valid Keypoints: {valid}/{total} ({pct:.1f}%)")

def process_image(
    pipeline: PoseTransferPipeline,
    src_path: Path,
    out_dir: Path,
    ref_data: Optional[dict] = None, # (kpts, scores, size)
    config_threshold: float = 0.3
):
    """단일 이미지 처리 함수"""
    file_stem = src_path.stem  # 확장자 뺀 파일명 (라벨링용)
    print(f"\nProcessing: {src_path.name} ...")

    try:
        # [Step 1] Source 추출
        src_img = load_image(src_path)
        src_kpts, src_scores, _, src_size = pipeline.extract_pose(src_img)
        
        analyze_keypoints("Source", src_scores, config_threshold)

        # Source 결과 저장 (공통)
        # 1. JSON
        src_json = convert_to_openpose_format(src_kpts[None], src_scores[None], src_size)
        save_json(src_json, str(out_dir / f"{file_stem}_keypoints.json"))
        
        # 2. Skeleton
        src_skel = pipeline.renderer.render_skeleton_only((src_size[0], src_size[1], 3), src_kpts, src_scores)
        save_image(src_skel, str(out_dir / f"{file_stem}_skeleton.png"))
        
        # 3. Overlay
        src_overlay = pipeline.renderer.render(src_img, src_kpts, src_scores)
        save_image(src_overlay, str(out_dir / f"{file_stem}_overlay.png"))

        # [Step 2] 전이 (Reference가 있을 경우에만)
        if ref_data:
            ref_kpts, ref_scores, ref_size = ref_data
            
            # 전이 실행 (이미지 사이즈 전달하여 하반신 검증 활성화)
            result = pipeline.transfer_engine.transfer(
                src_kpts, src_scores,
                ref_kpts, ref_scores,
                source_image_size=src_size,
                reference_image_size=ref_size
            )
            
            # 렌더링
            res_skel = pipeline.renderer.render_skeleton_only((src_size[0], src_size[1], 3), result.keypoints, result.scores)
            res_overlay = pipeline.renderer.render(src_img, result.keypoints, result.scores)
            
            # 전이 결과 저장 (라벨링: 원본명_transferred)
            save_image(res_skel, str(out_dir / f"{file_stem}_transferred_skeleton.png"))
            save_image(res_overlay, str(out_dir / f"{file_stem}_transferred_overlay.png"))
            save_json(result.to_json(), str(out_dir / f"{file_stem}_transferred_keypoints.json"))
            
            print(f"   ✅ Transfer Complete -> {file_stem}_transferred_*.png")
        else:
            print(f"   ✅ Extraction Complete -> {file_stem}_*.png")

    except Exception as e:
        print(f"   ❌ Error processing {src_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Pose Transfer Batch Test')
    # 기본값을 test_Inputs 폴더로 설정
    parser.add_argument('--source', type=str, default='test_Inputs', help='Input Directory or File')
    parser.add_argument('--reference', type=str, default=None, help='Reference Image Path (Optional)')
    parser.add_argument('--output', type=str, default='outputs_test', help='Output Directory')
    parser.add_argument('--config', type=str, default='pose_transfer/config/default.yaml', help='Config Path')
    
    args = parser.parse_args()
    
    # 1. 경로 설정
    source_input = Path(args.source)
    out_dir = Path(args.output)

    # [NEW] 기존 출력 폴더 정리 (Reset)
    if out_dir.exists():
        print(f"🧹 Cleaning up existing output directory: {out_dir}")
        shutil.rmtree(out_dir)  # 폴더 통째로 삭제
    
    out_dir.mkdir(parents=True, exist_ok=True) # 다시 생성

    # 소스 파일 목록 확보
    if source_input.is_dir():
        src_files = get_image_files(source_input)
        if not src_files:
            print(f"❌ '{source_input}' 폴더에 이미지 파일이 없습니다.")
            return
        print(f"📂 Batch Mode: '{source_input}' 폴더 내 {len(src_files)}개 이미지 처리")
    elif source_input.exists():
        src_files = [source_input]
        print(f"📄 Single Mode: {source_input} 처리")
    else:
        print(f"❌ Source 경로를 찾을 수 없습니다: {source_input}")
        return

    # 2. 파이프라인 초기화
    config_path = Path(args.config)
    yaml_config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        config = PipelineConfig()

    pipeline = PoseTransferPipeline(config, yaml_config=yaml_config)

    # 3. Reference 로드 (옵션)
    ref_data = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            print(f"💃 Reference Loading: {ref_path}")
            ref_kpts, ref_scores, _, ref_size = pipeline.extract_pose(ref_path)
            ref_data = (ref_kpts, ref_scores, ref_size)
            
            # Reference 분석 결과도 한 번 저장
            r_skel = pipeline.renderer.render_skeleton_only((ref_size[0], ref_size[1], 3), ref_kpts, ref_scores)
            save_image(r_skel, str(out_dir / "reference_skeleton.png"))
        else:
            print(f"❌ Reference 파일을 찾을 수 없어 '추출 모드'로 진행합니다: {ref_path}")

    print("="*60)
    
    # 4. 일괄 처리 루프
    for src_path in src_files:
        process_image(
            pipeline, 
            src_path, 
            out_dir, 
            ref_data, 
            config.kpt_threshold
        )

    print("="*60)
    print(f"✨ 모든 작업 완료! 결과물 위치: {out_dir}")

if __name__ == "__main__":
    main()