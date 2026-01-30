"""
Pose Transfer CLI Entry Point (Refactored v2.1 - Folder Structure Restore)

변경사항:
- [복구] 출력 폴더명을 '날짜_시간_src_to_ref' 형식으로 자동 생성하도록 변경
- api.execute 호출 시 해당 경로 전달
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# OpenMP 중복 로딩 에러 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent))

from pose_transfer.api import PoseTransferAPI
from pose_transfer.utils.io import get_image_files

# 기본 경로 설정
BASE_DIR = Path(__file__).parent
DEFAULT_IO_DIR = BASE_DIR / "io"
DEFAULT_SRC_DIR = DEFAULT_IO_DIR / "inputs" / "source"
DEFAULT_REF_DIR = DEFAULT_IO_DIR / "inputs" / "reference"
DEFAULT_OUT_DIR = DEFAULT_IO_DIR / "outputs"


def find_input_images(src_arg: Optional[str], ref_arg: Optional[str]) -> Tuple[Path, Path]:
    """입력 이미지 자동 탐색"""
    if src_arg and ref_arg:
        src_path = Path(src_arg)
        ref_path = Path(ref_arg)
        if not src_path.exists(): raise FileNotFoundError(f"Source not found: {src_path}")
        if not ref_path.exists(): raise FileNotFoundError(f"Reference not found: {ref_path}")
        return src_path, ref_path

    print("\n🔍 Arguments not provided. Scanning default directories...")
    src_dir = DEFAULT_SRC_DIR if DEFAULT_SRC_DIR.exists() else DEFAULT_IO_DIR / "inputs" / "src"
    ref_dir = DEFAULT_REF_DIR if DEFAULT_REF_DIR.exists() else DEFAULT_IO_DIR / "inputs" / "ref"
    
    if not src_dir.exists() or not ref_dir.exists():
        raise FileNotFoundError(f"Default input folders not found: {src_dir}, {ref_dir}")

    src_files = get_image_files(src_dir)
    ref_files = get_image_files(ref_dir)
    
    if not src_files: raise FileNotFoundError(f"No images found in {src_dir}")
    if not ref_files: raise FileNotFoundError(f"No images found in {ref_dir}")
    
    src_path = src_files[0]
    ref_path = ref_files[0]
    
    print(f"   👉 Auto-selected Source: {src_path.name}")
    print(f"   👉 Auto-selected Ref   : {ref_path.name}")
    
    return src_path, ref_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '--src', type=str)
    parser.add_argument('--reference', '--ref', type=str)
    parser.add_argument('--output', '--out', type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    
    try:
        # 1. API 초기화
        api = PoseTransferAPI(base_dir=str(BASE_DIR))
        
        # 2. 실행 대상 결정
        src_path, ref_path = find_input_images(args.source, args.reference)
        
        # 3. [복구] 폴더명 생성 로직 (YYYY-MM-DD_HH-MM-SS_src_to_ref)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{src_path.stem}_to_{ref_path.stem}"
        
        # 최종 출력 경로: io/outputs/날짜_시간_src_to_ref/
        final_output_dir = Path(args.output) / folder_name
        
        # 4. 실행
        results = api.execute(
            source_path=src_path,
            reference_path=ref_path,
            output_dir=final_output_dir,
            prefix="trans" # 파일명 접두사는 깔끔하게 trans로 통일
        )
        
        print(f"\n{'='*50}")
        print("✅ Execution Finished Successfully")
        print(f"{'='*50}")
        print(f"📁 Output Folder: {final_output_dir}")
        print(f"   (Check inside for src/, ref/, trans/ folders)")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()