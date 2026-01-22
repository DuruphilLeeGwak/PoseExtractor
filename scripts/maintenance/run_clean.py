#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 캐시 완전 삭제 후 PoseExtractor 실행"""

import os
import shutil
import sys
from pathlib import Path

def clean_pycache(root_dir):
    """모든 __pycache__ 폴더와 .pyc 파일 삭제"""
    deleted_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        # __pycache__ 폴더 삭제
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"✓ 삭제: {pycache_path}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ 실패: {pycache_path} - {e}")
        
        # .pyc, .pyo 파일 삭제
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"✓ 삭제: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"✗ 실패: {file_path} - {e}")
    
    return deleted_count

if __name__ == "__main__":
    print("=" * 70)
    print("🗑️  Python 캐시 완전 삭제")
    print("=" * 70)
    
    root = Path(__file__).parent
    deleted = clean_pycache(root)
    
    print(f"\n✅ 캐시 삭제 완료: {deleted}개 항목")
    print("\n이제 다음 명령을 실행하세요:")
    print("  python PoseExtractor.py")
