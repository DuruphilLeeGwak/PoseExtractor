"""
Pose Transfer System
====================

원본 이미지의 신체 비율을 유지하면서 레퍼런스 이미지의 포즈로 전이하는 시스템입니다.

Features:
- DWPose 기반 133 키포인트 추출 (Body + Face + Hands)
- 다중 인물 필터링 (이미지 중심 + 면적 기반)
- 벡터 기반 포즈 전이
- 누락 키포인트 폴백 (대칭 미러링, 계층적 추정)
- 손 영역 동적 업스케일

사용법:
```python
from pose_transfer import transfer_pose, extract_pose_from_image

# 단일 이미지 포즈 추출
json_data, skeleton_image = extract_pose_from_image("image.jpg")

# 포즈 전이
json_data, skeleton_image = transfer_pose(
    source_image="source.jpg",      # 신체 비율 소스
    reference_image="reference.jpg"  # 포즈 소스
)
```

고급 사용법:
```python
from pose_transfer import PoseTransferPipeline, PipelineConfig

# 설정 커스터마이징
config = PipelineConfig(
    backend='onnxruntime',
    device='cuda',
    mode='performance',
    filter_enabled=True,
    hand_refinement_enabled=True
)

pipeline = PoseTransferPipeline(config)
result = pipeline.transfer(source_image, reference_image)

# 결과 활용
result.transferred_keypoints  # 전이된 키포인트
result.skeleton_image         # 스켈레톤 이미지
result.to_json()              # OpenPose 호환 JSON
```
"""

__version__ = '0.1.0'
__author__ = '두루필'

from .pipeline import (
    PipelineConfig,
    PipelineResult,
    PoseTransferPipeline,
    get_pipeline,
    extract_pose_from_image,
    transfer_pose
)

from .extractors import (
    DWPoseExtractor,
    DWPoseExtractorFactory,
    PersonFilter,
    filter_main_person,
    RTMLIB_AVAILABLE
)

from .analyzers import (
    BoneCalculator,
    BodyProportions,
    DirectionExtractor,
    PoseDirections
)

from .transfer import (
    PoseTransferEngine,
    TransferConfig,
    FallbackStrategy
)

from .refiners import HandRefiner

from .renderers import SkeletonRenderer, render_skeleton

from .utils import (
    load_config,
    save_config,
    load_image,
    save_image,
    save_json,
    load_json,
    convert_to_openpose_format
)

__all__ = [
    # Pipeline
    'PipelineConfig',
    'PipelineResult', 
    'PoseTransferPipeline',
    'get_pipeline',
    'extract_pose_from_image',
    'transfer_pose',
    
    # Extractors
    'DWPoseExtractor',
    'DWPoseExtractorFactory',
    'PersonFilter',
    'filter_main_person',
    'RTMLIB_AVAILABLE',
    
    # Analyzers
    'BoneCalculator',
    'BodyProportions',
    'DirectionExtractor',
    'PoseDirections',
    
    # Transfer
    'PoseTransferEngine',
    'TransferConfig',
    'FallbackStrategy',
    
    # Refiners
    'HandRefiner',
    
    # Renderers
    'SkeletonRenderer',
    'render_skeleton',
    
    # Utils
    'load_config',
    'save_config',
    'load_image',
    'save_image',
    'save_json',
    'load_json',
    'convert_to_openpose_format',
]
