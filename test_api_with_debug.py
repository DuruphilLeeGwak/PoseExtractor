"""
Pose Transfer API 테스트 (Debug 정보 포함)
"""
from pose_transfer.api import execute_pose_transfer

# Source와 Reference 이미지로 Pose Transfer 실행
result = execute_pose_transfer(
    source_path='test_io/inputs/3.jpg',
    reference_path='test_io/inputs/2.jpg',
    output_root='io/outputs'
)

print(f"\n✅ 완료!")
print(f"📁 결과 폴더: {result['job_dir']}")
print(f"\n생성된 파일:")
for key, path in result.items():
    if key != 'job_dir':
        print(f"  - {key}: {path}")
