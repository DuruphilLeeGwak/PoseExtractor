"""
Pose Transfer Batch Test Script (Refactored v2.0)
- 목적: test_Inputs 폴더 내의 모든 이미지를 일괄 테스트
- 기능: API를 사용하여 간결하고 안정적인 배치 테스트 수행
"""
import sys
import shutil
import argparse
from pathlib import Path

# 프로젝트 루트 경로 확보
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_transfer.api import PoseTransferAPI
from pose_transfer.utils.io import get_image_files

# 테스트용 기본 경로
TEST_INPUT_DIR = PROJECT_ROOT / "test_Inputs"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test_Outputs"

def main():
    parser = argparse.ArgumentParser(description='Batch Test Transfer')
    parser.add_argument('--input', type=str, default=str(TEST_INPUT_DIR), help='Input folder')
    parser.add_argument('--output', type=str, default=str(TEST_OUTPUT_DIR), help='Output folder')
    parser.add_argument('--ref', type=str, help='Specific reference image path (Optional)')
    args = parser.parse_args()
    
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    
    # 1. 초기화 (Clean Start)
    if out_dir.exists():
        print(f"🧹 Cleaning output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 이미지 스캔
    if not in_dir.exists():
        print(f"❌ Input directory not found: {in_dir}")
        return

    images = get_image_files(in_dir)
    if not images:
        print(f"⚠️ No images found in {in_dir}")
        return
        
    print(f"🚀 Found {len(images)} images. Initializing API...")
    
    # 3. API 초기화
    api = PoseTransferAPI(base_dir=str(PROJECT_ROOT))
    
    # Reference 이미지 결정
    # 인자로 주어지면 그걸 쓰고, 아니면 폴더 내 첫 번째 이미지를 Ref로 가정 (혹은 테스트 정책에 따름)
    ref_path = Path(args.ref) if args.ref else None
    
    if not ref_path or not ref_path.exists():
        # 테스트용: 첫 번째 이미지를 Reference로 사용
        ref_path = images[0]
        print(f"ℹ️ No reference provided. Using first image as ref: {ref_path.name}")
    
    # 4. 배치 실행
    print(f"\n⚡ Starting Batch Transfer (Ref: {ref_path.name})")
    print("=" * 60)
    
    for i, src_path in enumerate(images):
        # Ref 이미지는 Skip (자기 자신에게 전이하는 테스트가 아니라면)
        if src_path == ref_path:
            continue
            
        print(f"[{i+1}/{len(images)}] Processing: {src_path.name}...")
        
        case_out_dir = out_dir / f"case_{src_path.stem}"
        
        try:
            api.execute(
                source_path=src_path,
                reference_path=ref_path,
                output_dir=case_out_dir,
                prefix=src_path.stem
            )
        except Exception as e:
            print(f"❌ Failed on {src_path.name}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*60)
    print("✅ Batch Test Finished.")
    print(f"📁 Results: {out_dir}")

if __name__ == "__main__":
    main()