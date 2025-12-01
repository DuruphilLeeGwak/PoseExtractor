"""
Pose Transfer System - 사용 예시

이 스크립트는 포즈 전이 시스템의 주요 기능을 보여줍니다.
"""
import sys
from pathlib import Path

# 모듈 경로 추가 (개발 환경용)
sys.path.insert(0, str(Path(__file__).parent))


def example_single_extraction():
    """
    예시 1: 단일 이미지에서 포즈 추출
    """
    print("=" * 60)
    print("예시 1: 단일 이미지 포즈 추출")
    print("=" * 60)
    
    from pose_transfer import extract_pose_from_image, save_json, save_image
    
    # 이미지 경로 (실제 경로로 변경)
    image_path = "path/to/your/image.jpg"
    
    # 포즈 추출
    json_data, skeleton_image = extract_pose_from_image(image_path)
    
    # 저장
    save_json(json_data, "output_keypoints.json")
    save_image(skeleton_image, "output_skeleton.png")
    
    print("완료!")
    print(f"  - JSON: output_keypoints.json")
    print(f"  - 스켈레톤: output_skeleton.png")


def example_pose_transfer():
    """
    예시 2: 포즈 전이
    
    원본 이미지의 신체 비율 + 레퍼런스 이미지의 포즈
    """
    print("=" * 60)
    print("예시 2: 포즈 전이")
    print("=" * 60)
    
    from pose_transfer import transfer_pose, save_json, save_image
    
    # 이미지 경로 (실제 경로로 변경)
    source_image = "path/to/source.jpg"      # 신체 비율 소스
    reference_image = "path/to/reference.jpg"  # 포즈 소스
    
    # 포즈 전이
    json_data, skeleton_image = transfer_pose(source_image, reference_image)
    
    # 저장
    save_json(json_data, "transferred_keypoints.json")
    save_image(skeleton_image, "transferred_skeleton.png")
    
    print("완료!")
    print(f"  - JSON: transferred_keypoints.json")
    print(f"  - 스켈레톤: transferred_skeleton.png")


def example_advanced_pipeline():
    """
    예시 3: 고급 파이프라인 사용
    
    세부 설정 커스터마이징 및 중간 결과 접근
    """
    print("=" * 60)
    print("예시 3: 고급 파이프라인")
    print("=" * 60)
    
    from pose_transfer import (
        PoseTransferPipeline,
        PipelineConfig,
        save_json,
        save_image
    )
    
    # 설정 커스터마이징
    config = PipelineConfig(
        # 모델 설정
        backend='onnxruntime',
        device='cuda',           # 'cpu' for CPU only
        mode='performance',      # 'balanced', 'lightweight'
        
        # 다중 인물 필터링
        filter_enabled=True,
        area_weight=0.6,         # 면적 가중치
        center_weight=0.4,       # 중심 거리 가중치
        
        # 손 정밀화
        hand_refinement_enabled=True,
        min_hand_size=48,
        
        # 폴백
        fallback_enabled=True,
        
        # 신뢰도 임계값
        confidence_threshold=0.3,
        
        # 렌더링
        line_thickness=4,
        point_radius=4
    )
    
    # 파이프라인 생성
    pipeline = PoseTransferPipeline(config)
    
    # 이미지 경로 (실제 경로로 변경)
    source_image = "path/to/source.jpg"
    reference_image = "path/to/reference.jpg"
    
    # 포즈 전이
    result = pipeline.transfer(source_image, reference_image)
    
    # 결과 접근
    print(f"\n결과 정보:")
    print(f"  - 이미지 크기: {result.image_size}")
    print(f"  - 선택된 인물: {result.selected_person_idx}")
    print(f"  - 원본 본 길이: {len(result.source_bone_lengths)}개")
    
    # 전이된 키포인트 형태
    print(f"  - 전이된 키포인트 shape: {result.transferred_keypoints.shape}")
    
    # JSON으로 변환
    json_data = result.to_json()
    
    # 저장
    save_json(json_data, "advanced_output.json")
    save_image(result.skeleton_image, "advanced_skeleton.png")
    
    print("\n완료!")


def example_batch_processing():
    """
    예시 4: 배치 처리
    
    폴더 내 모든 이미지 처리
    """
    print("=" * 60)
    print("예시 4: 배치 처리")
    print("=" * 60)
    
    from pathlib import Path
    from pose_transfer import (
        PoseTransferPipeline,
        save_json,
        save_image
    )
    from pose_transfer.utils import get_image_files
    
    # 파이프라인 생성
    pipeline = PoseTransferPipeline()
    
    # 입출력 폴더
    input_folder = Path("path/to/input/images")
    output_folder = Path("path/to/output")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 목록
    image_files = get_image_files(input_folder)
    print(f"발견된 이미지: {len(image_files)}개")
    
    # 배치 처리
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}")
        
        try:
            json_data, skeleton_image = pipeline.extract_and_render(str(img_path))
            
            # 저장
            json_path = output_folder / f"{img_path.stem}_keypoints.json"
            skeleton_path = output_folder / f"{img_path.stem}_skeleton.png"
            
            save_json(json_data, json_path)
            save_image(skeleton_image, skeleton_path)
            
        except Exception as e:
            print(f"  오류: {e}")
    
    print(f"\n완료! 출력: {output_folder}")


def example_integration_with_controlnet():
    """
    예시 5: ControlNet 연동
    
    전이된 포즈 스켈레톤을 ControlNet 입력으로 사용
    """
    print("=" * 60)
    print("예시 5: ControlNet 연동 (개념)")
    print("=" * 60)
    
    print("""
    포즈 전이 결과를 ControlNet에 입력하는 방법:
    
    1. 포즈 전이 수행
       result = pipeline.transfer(source_image, reference_image)
    
    2. 스켈레톤 이미지 저장
       save_image(result.skeleton_image, "control_pose.png")
    
    3. ControlNet에 전달
       - Stable Diffusion WebUI: control_pose.png를 ControlNet 입력으로 사용
       - ComfyUI: LoadImage 노드로 control_pose.png 로드
       - Python API: 이미지를 직접 전달
    
    예시 (diffusers 사용):
    
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose"
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet
    )
    
    # 포즈 전이 결과 사용
    pose_image = Image.fromarray(result.skeleton_image[:, :, ::-1])  # BGR to RGB
    
    output = pipe(
        prompt="your prompt",
        image=pose_image,
        num_inference_steps=30
    ).images[0]
    """)


if __name__ == '__main__':
    print("\n포즈 전이 시스템 사용 예시")
    print("=" * 60)
    print("\n주의: 실제 이미지 경로로 변경 후 실행하세요.\n")
    
    # 예시 선택
    print("예시 목록:")
    print("  1. 단일 이미지 포즈 추출")
    print("  2. 포즈 전이")
    print("  3. 고급 파이프라인")
    print("  4. 배치 처리")
    print("  5. ControlNet 연동 (개념)")
    
    # example_single_extraction()
    # example_pose_transfer()
    # example_advanced_pipeline()
    # example_batch_processing()
    example_integration_with_controlnet()
