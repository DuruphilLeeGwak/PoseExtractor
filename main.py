"""
Pose Transfer CLI Entry Point
"""
import sys
import yaml
import argparse
from pathlib import Path

# 방금 만든 api 모듈에서 함수 가져오기
from pose_transfer.api import execute_pose_transfer, resolve_input_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Transfer Automation')
    parser.add_argument('--source', type=str, help='Source Image Path (Override)')
    parser.add_argument('--reference', type=str, help='Reference Image Path (Override)')
    parser.add_argument('--output', default='outputs', help='Output Root Directory')
    parser.add_argument('--config', default='pose_transfer/config/default.yaml', help='Config File Path')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    try:
        # 경로 결정 (CLI vs YAML)
        src_path, ref_path = resolve_input_paths(args, yaml_config)
        
        # 실행
        results = execute_pose_transfer(
            source_path=src_path,
            reference_path=ref_path,
            output_root=args.output,
            config_path=str(config_path),
            explicit_config=yaml_config
        )
        print(f"\n[Result] Skeleton Image: {results['skeleton']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("👉 default.yaml의 [input_mode] 설정을 확인하거나 CLI 인자를 제공해주세요.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        sys.exit(1)