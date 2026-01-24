"""
Pose Transfer CLI Entry Point

실행 모드:
1. 독립 실행: python PoseExtractor.py
   → io/inputs/src, io/inputs/ref 에서 이미지 자동 탐색
   → io/outputs/ 에 결과 저장

2. 경로 지정 실행: python PoseExtractor.py --source img1.jpg --reference img2.jpg
   → 지정된 경로 사용
   → --output 으로 출력 경로 지정 가능

3. Pozibility 통합 시:
   → data/inputs/src, ref 에서 자동 탐색
   → data/outputs/ 에 결과 저장
"""
import sys
import os
import yaml
import argparse
from pathlib import Path

# OpenMP 중복 로딩 에러 방지 (onnxruntime/rtmlib 충돌 회피)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pose_transfer.api import execute_pose_transfer


def main():
    parser = argparse.ArgumentParser(
        description='Pose Transfer - Source 체형에 Reference 포즈 적용',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 자동 탐색 모드 (io/inputs/ 폴더 사용)
  python PoseExtractor.py

  # 경로 직접 지정
  python PoseExtractor.py --source person.jpg --reference pose.jpg

  # 출력 경로 지정
  python PoseExtractor.py --source person.jpg --reference pose.jpg --output results/
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str, 
        default=None,
        help='Source 이미지 경로 (체형 기준)'
    )
    parser.add_argument(
        '--reference', '-r',
        type=str, 
        default=None,
        help='Reference 이미지 경로 (포즈 기준)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,  # None이면 api.py가 자동 결정
        help='출력 디렉토리 (기본: io/outputs 또는 data/outputs)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='pose_transfer/config/default.yaml',
        help='설정 파일 경로'
    )
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    config_path = Path(args.config)
    yaml_config = {}
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}
    else:
        print(f"⚠️ Config file not found: {config_path}")
        print("   Using default settings.")
    
    try:
        # 실행 (경로가 None이면 api.py가 자동 탐색)
        results = execute_pose_transfer(
            source_path=args.source,
            reference_path=args.reference,
            output_root=args.output,
            config_path=str(config_path) if config_path.exists() else None,
            explicit_config=yaml_config
        )
        
        print(f"\n{'='*50}")
        print("✅ 완료!")
        print(f"{'='*50}")
        print(f"📁 결과 폴더: {results.get('job_dir', 'N/A')}")
        print(f"🖼️  Skeleton: {results.get('skeleton', 'N/A')}")
        print(f"📄 JSON: {results.get('json', 'N/A')}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다:")
        print(f"   {e}")
        print(f"\n💡 해결 방법:")
        print(f"   1. io/inputs/src/ 폴더에 Source 이미지를 넣으세요")
        print(f"   2. io/inputs/ref/ 폴더에 Reference 이미지를 넣으세요")
        print(f"   3. 또는 --source, --reference 옵션으로 경로를 직접 지정하세요")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()