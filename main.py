#!/usr/bin/env python
"""
Pose Transfer System - CLI
"""
import argparse
import sys
from pathlib import Path

# 설정 파일 경로
CONFIG_PATH = Path(__file__).parent / "pose_transfer" / "config" / "default.yaml"


def main():
    parser = argparse.ArgumentParser(
        description='Pose Transfer System - 포즈 전이 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # extract 명령어
    extract_parser = subparsers.add_parser('extract', help='단일 이미지에서 포즈 추출')
    extract_parser.add_argument('--input', '-i', required=True, help='입력 이미지 경로')
    extract_parser.add_argument('--output', '-o', required=True, help='출력 디렉토리')
    extract_parser.add_argument('--config', '-c', default=None, help='설정 파일 경로')
    
    # transfer 명령어
    transfer_parser = subparsers.add_parser('transfer', help='포즈 전이 수행')
    transfer_parser.add_argument('--source', '-s', required=True, help='원본 이미지')
    transfer_parser.add_argument('--reference', '-r', required=True, help='레퍼런스 이미지')
    transfer_parser.add_argument('--output', '-o', required=True, help='출력 디렉토리')
    transfer_parser.add_argument('--config', '-c', default=None, help='설정 파일 경로')
    
    # batch 명령어
    batch_parser = subparsers.add_parser('batch', help='배치 처리')
    batch_parser.add_argument('--input', '-i', required=True, help='입력 폴더')
    batch_parser.add_argument('--output', '-o', required=True, help='출력 폴더')
    batch_parser.add_argument('--config', '-c', default=None, help='설정 파일 경로')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 지연 임포트
    from pose_transfer import (
        PoseTransferPipeline,
        PipelineConfig,
        save_json,
        save_image,
        get_image_files
    )
    
    # 설정 파일 로드 (핵심!)
    config_path = args.config if args.config else CONFIG_PATH
    if Path(config_path).exists():
        print(f"[INFO] Loading config from: {config_path}")
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        print(f"[WARNING] Config not found: {config_path}, using defaults")
        config = PipelineConfig()
    
    pipeline = PoseTransferPipeline(config)
    
    if args.command == 'extract':
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = Path(args.input)
        print(f"Processing: {input_path}")
        
        json_data, skeleton_image, overlay_image = pipeline.extract_and_render(str(input_path))
        
        json_path = output_dir / f"{input_path.stem}_keypoints.json"
        skeleton_path = output_dir / f"{input_path.stem}_skeleton.png"
        overlay_path = output_dir / f"{input_path.stem}_rendered.png"
        
        save_json(json_data, json_path)
        save_image(skeleton_image, skeleton_path)
        save_image(overlay_image, overlay_path)
        
        print(f"Saved: {json_path}")
        print(f"Saved: {skeleton_path}")
        print(f"Saved: {overlay_path}")
    
    elif args.command == 'transfer':
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(args.source)
        ref_path = Path(args.reference)
        
        print(f"Source: {source_path}")
        print(f"Reference: {ref_path}")
        
        result = pipeline.transfer(str(source_path), str(ref_path))
        
        json_path = output_dir / f"{source_path.stem}_transferred_keypoints.json"
        skeleton_path = output_dir / f"{source_path.stem}_transferred_skeleton.png"
        
        save_json(result.to_json(), json_path)
        save_image(result.skeleton_image, skeleton_path)
        
        print(f"Saved: {json_path}")
        print(f"Saved: {skeleton_path}")
    
    elif args.command == 'batch':
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = get_image_files(input_dir)
        print(f"Found {len(image_files)} images")
        
        for i, img_path in enumerate(image_files):
            print(f"[{i+1}/{len(image_files)}] Processing: {img_path.name}")
            try:
                json_data, skeleton_image, overlay_image = pipeline.extract_and_render(str(img_path))
                
                json_path = output_dir / f"{img_path.stem}_keypoints.json"
                skeleton_path = output_dir / f"{img_path.stem}_skeleton.png"
                
                save_json(json_data, json_path)
                save_image(skeleton_image, skeleton_path)
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"Done! Output: {output_dir}")


if __name__ == '__main__':
    main()
