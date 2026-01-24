"""
Face Rendering 설정 테스트 스크립트
- face_rendering.enabled=true일 때와 false일 때를 비교
"""
import sys
from pathlib import Path
import yaml
import tempfile
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_transfer.api import execute_pose_transfer

def test_face_rendering():
    """Face rendering 설정 테스트"""
    
    # 테스트 이미지 경로
    src_img = PROJECT_ROOT / "io" / "inputs" / "src" / "src.jpg"
    ref_img = PROJECT_ROOT / "io" / "inputs" / "ref" / "ref.jpg"
    
    if not src_img.exists() or not ref_img.exists():
        print("❌ io/inputs/src/src.jpg 또는 io/inputs/ref/ref.jpg가 없습니다.")
        return
    
    print(f"Source: {src_img}")
    print(f"Reference: {ref_img}")
    
    # 1. enabled=true 테스트
    print("\n" + "="*70)
    print("TEST 1: face_rendering.enabled = true")
    print("="*70)
    
    output_dir_enabled = PROJECT_ROOT / "test_io" / "face_rendering_test" / "enabled"
    if output_dir_enabled.exists():
        shutil.rmtree(output_dir_enabled)
    
    result_enabled = execute_pose_transfer(
        source_path=str(src_img),
        reference_path=str(ref_img),
        output_root=str(output_dir_enabled)
    )
    
    print(f"✅ Output: {output_dir_enabled}")
    
    # 2. enabled=false 테스트
    print("\n" + "="*70)
    print("TEST 2: face_rendering.enabled = false")
    print("="*70)
    
    # default.yaml을 읽어서 수정
    config_path = PROJECT_ROOT / "pose_transfer" / "config" / "default.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 백업
    original_face_enabled = config.get('face_rendering', {}).get('enabled', True)
    
    # false로 변경
    config['face_rendering']['enabled'] = False
    
    # 임시 config 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name
    
    try:
        output_dir_disabled = PROJECT_ROOT / "test_io" / "face_rendering_test" / "disabled"
        if output_dir_disabled.exists():
            shutil.rmtree(output_dir_disabled)
        
        # 임시 config로 실행
        result_disabled = execute_pose_transfer(
            source_path=str(src_img),
            reference_path=str(ref_img),
            output_root=str(output_dir_disabled),
            config_path=tmp_config_path
        )
        
        print(f"✅ Output: {output_dir_disabled}")
        
    finally:
        # 임시 파일 삭제
        Path(tmp_config_path).unlink()
    
    print("\n" + "="*70)
    print("✅ 테스트 완료")
    print("="*70)
    print(f"enabled=true 출력: {output_dir_enabled}")
    print(f"enabled=false 출력: {output_dir_disabled}")
    print("\n두 결과물을 비교하여 face landmarks가 제대로 제거되었는지 확인하세요.")

if __name__ == "__main__":
    test_face_rendering()
