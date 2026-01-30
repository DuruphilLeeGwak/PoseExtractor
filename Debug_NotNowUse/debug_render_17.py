"""17.jpg 렌더링 디버그"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pose_transfer import execute_pose_transfer

# 17.jpg만 처리
result = execute_pose_transfer(
    source_path="test_io/inputs/17.jpg",
    output_folder="test_io/debug_render_17",
    to_openpose=False,
    save_json=True,
    save_skeleton=True,
    save_debug=True
)

print("\n=== Debug Info ===")
print(f"Result keys: {result.keys()}")
if 'outputs' in result:
    for key, path in result['outputs'].items():
        print(f"  {key}: {path}")
