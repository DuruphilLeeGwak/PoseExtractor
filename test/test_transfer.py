"""
Pose Transfer Batch Test Script (Auto Clean)
- 목적: test_Inputs 폴더 내의 모든 이미지를 일괄 테스트
- 기능:
  1. 시작 시 기존 output 폴더 삭제 후 재생성 (Clean Start)
  2. 폴더 내 모든 이미지에 대해 키포인트 분석 (Reference 없을 때)
  3. 폴더 내 모든 이미지에 특정 Reference 포즈 전이 (Reference 있을 때)
"""
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Ensure local imports work regardless of cwd / launcher wrappers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 패키지 임포트
from pose_transfer.pipeline import PipelineConfig, PoseTransferPipeline
from pose_transfer.utils.io import save_json, save_image, load_image, convert_to_openpose_format
from pose_transfer.logic.debug_generator import generate_debug_info  # 디버그 정보 생성

# 이미지 확장자 목록
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


GHOSTFILTER_LAYERS_DEBUG_GLOSSARY = (
    "╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "║              GhostFilter v5.0 레이어 디버그 용어집 (Glossary)                ║\n"
    "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    "\n"
    "이 파일은 이미지 단위로 append(추가)됩니다. 헤더는 파일이 비어있을 때 1회만 기록됩니다.\n"
    "\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    " [1] 레이어 시스템 (Layer System)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "레이어 0 (정상): 이미지 안에서 정상적으로 보이는 키포인트\n"
    "              → 일반 렌더링 (불투명)\n"
    "\n"
    "레이어 -1 (폐색): 이미지 안에 있지만 옷/신체에 가려진 키포인트\n"
    "                → occluded_indices에 마킹\n"
    "                → score는 유지됨 (제거 안 됨!)\n"
    "                → 렌더링 시 투명도 50% 적용\n"
    "\n"
    "레이어 -2 (프레임 밖): 이미지 경계 밖으로 나간 키포인트\n"
    "                     → out_of_frame_indices에 마킹\n"
    "                     → 렌더링 시 포인트(원)는 안 그리고 라인만 그림\n"
    "\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    " [2] Step 3.5 (폐색/환각 억제) 변수\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "verdict: KEEP / OCCLUDED / SKIP\n"
    "  ✅ KEEP: 정상으로 판단, 레이어 0\n"
    "  🟡 OCCLUDED: 가려진/환각으로 판단, 레이어 -1 (occluded_indices에 마킹)\n"
    "              → 제거되지 않음! score는 유지됨\n"
    "              → 렌더링 시 투명도 50%로 표현\n"
    "  ⚪ SKIP: 사전 조건 실패로 Step3.5 실행 안 함\n"
    "\n"
    "주요 변수:\n"
    "- thr: 최소 confidence (키포인트가 'active'로 집계되기 위한 임계값)\n"
    "- active=N/21: confidence >= thr 인 키포인트 개수\n"
    "- min_pts: Step3.5 평가를 위한 최소 active 개수\n"
    "- wrist_score / elbow_score: 손목/팔꿈치 confidence\n"
    "- forearm: 팔꿈치-손목 거리 (픽셀)\n"
    "- base_r: 손목 중심 원 반지름 (near 판정용)\n"
    "- near: active 점 중 base_r 내에 있는 점 개수\n"
    "- near_ratio: near / active\n"
    "- min_near: 통과 기준 최소 near_ratio\n"
    "- far_r: far outlier 원 반지름\n"
    "- far: active 점 중 far_r 밖에 있는 점 개수\n"
    "- max_far: 허용 가능한 far outlier 최대 개수\n"
    "\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    " [3] 제거/마킹 사유 (Reasons)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "완전 제거 (filtered_scores = 0):\n"
    "- dummy_no_parent: 더미 좌표 + 부모 관절 없음\n"
    "- boundary_val: DWPose 감지 실패로 경계값(0, 0)에 찍힌 더미\n"
    "- clustered_hand: 손 점들이 한 점에 뭉침 (오검출)\n"
    "- orphan_node: 부모 관절 제거로 연쇄 제거 (Chain Kill)\n"
    "\n"
    "레이어 -1 마킹 (occluded_indices, 제거 안 됨):\n"
    "- occluded_LHand/RHand(마킹만): 가려진 것으로 판단\n"
    "  → near_ratio < min 또는 far_count > max\n"
    "  → 렌더링 시 투명도 50%\n"
    "\n"
    "레이어 -2 마킹 (out_of_frame_indices, 제거 안 됨):\n"
    "- dummy + parent valid: 더미 좌표지만 부모 유효\n"
    "  → 프레임 밖으로 마킹, 라인 형성 위해 유지\n"
    "\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    " [4] 출력 형식\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "각 이미지별로:\n"
    "- FILE: <이미지명>\n"
    "- [HAND][Step3.5] ... : Step 3.5 판정 결과\n"
    "- ┌───┐ 박스: 가려진 키포인트 정보\n"
    "- [HAND] 요약: 남은 키포인트 / 가려진 키포인트 / 제거 사유\n"
    "\n"
)


def _append_with_header_if_needed(path: Path, header: str, block_lines: List[str]) -> None:
    """텍스트를 append하고, 파일이 비어있으면 용어집 헤더를 먼저 기록합니다."""
    needs_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8") as f:
        if needs_header:
            f.write(header.rstrip("\n") + "\n")
            f.write("=" * 100 + "\n")
        for line in block_lines:
            f.write(line + "\n")

def get_image_files(directory: Path) -> List[Path]:
    """폴더 내 이미지 파일 목록 반환"""
    return [
        p for p in directory.iterdir() 
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]

def analyze_keypoints(name: str, scores: np.ndarray, threshold: float = 0.3):
    """키포인트 유효성 요약 출력"""
    total = len(scores)
    valid = np.sum(scores > threshold)
    pct = (valid / total) * 100
    print(f"   📊 [{name}] Valid Keypoints: {valid}/{total} ({pct:.1f}%)")

def process_image(
    pipeline: PoseTransferPipeline,
    src_path: Path,
    out_dir: Path,
    ref_data: Optional[dict] = None, # (ref_img, ref_path)
    config_threshold: float = 0.3,
    yaml_config: Optional[dict] = None  # 설정 딕셔너리 추가
):
    """단일 이미지 처리 함수"""
    file_stem = src_path.stem  # 확장자 뺀 파일명 (라벨링용)
    print(f"\nProcessing: {src_path.name} ...")

    try:
        # [Step 1] Source 추출
        src_img = load_image(src_path)
        src_kpts, src_scores, _, src_size = pipeline.extract_pose(src_img)
        # Cross-Filter가 extract_pose()에서 이미 적용됨
        src_scores_f = src_scores
        
        analyze_keypoints("Source", src_scores_f, config_threshold)

        # Source 결과 저장 (trans 출력 규칙과 유사하게: <name>_bg/_kp/_rend/_sk)
        # 0. Background (PoseExtractor 규칙: 원본/배경 그대로 저장)
        save_image(src_img, str(out_dir / f"{file_stem}_bg.jpg"))

        # 1. Keypoints JSON
        src_json = convert_to_openpose_format(src_kpts[None], src_scores_f[None], src_size)
        save_json(src_json, str(out_dir / f"{file_stem}_kp.json"))
        
        # 2. Skeleton
        src_skel = pipeline.renderer.render_skeleton_only(
            (src_size[0], src_size[1], 3), 
            src_kpts, 
            src_scores_f
        )
        save_image(src_skel, str(out_dir / f"{file_stem}_sk.jpg"))
        
        # 3. Render(Overlay)
        src_overlay = pipeline.renderer.render(
            src_img, 
            src_kpts, 
            src_scores_f
        )
        save_image(src_overlay, str(out_dir / f"{file_stem}_rend.jpg"))

        # [디버그 정보 생성] Cross-Filter 분석
        output_cfg = yaml_config.get('output', {}) if yaml_config else {}
        do_save_debug_txt = output_cfg.get('save_debug_txt', True)
        
        if do_save_debug_txt:
            try:
                debug_path = generate_debug_info(
                    src_path,
                    out_dir,
                    dw_extractor=pipeline.extractor,
                    body_extractor=None,  # 내부에서 생성
                    config=yaml_config
                )
                if debug_path:
                    print(f"   📝 Debug info: {Path(debug_path).name}")
            except Exception as e:
                print(f"   ⚠️ Debug 생성 중 오류 (무시): {e}")

        # [Step 2] 전이 (Reference가 있을 경우에만)
        if ref_data:
            ref_img, ref_path = ref_data

            # 전이 실행: 반드시 파이프라인을 사용 (Ghost Filter / Align / Canvas / Final Filter 포함)
            pipe_res = pipeline.transfer(src_img, ref_img)

            # 결과 저장 (라벨링: 원본명_transferred_*)
            bg = pipe_res.modified_source_image if pipe_res.modified_source_image is not None else src_img
            # transferred_bg: PoseExtractor 규칙대로 최종 배경(캔버스/크롭 적용 후)만 저장
            save_image(bg, str(out_dir / f"{file_stem}_transferred_bg.jpg"))
            save_json(pipe_res.to_json(), str(out_dir / f"{file_stem}_transferred_kp.json"))
            save_image(pipe_res.skeleton_image, str(out_dir / f"{file_stem}_transferred_sk.jpg"))
            overlay = pipeline.renderer.render(bg, pipe_res.transferred_keypoints, pipe_res.transferred_scores)
            save_image(overlay, str(out_dir / f"{file_stem}_transferred_rend.jpg"))

            print(f"   ✅ Transfer Complete -> {file_stem}_transferred_*.jpg/json")
        else:
            print(f"   ✅ Extraction Complete -> {file_stem}_*.jpg/json")

    except Exception as e:
        print(f"   ❌ Error processing {src_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Pose Transfer Batch Test')
    # 기본값: test_io/inputs 폴더
    parser.add_argument('--source', type=str, default='test_io/inputs', help='Input Directory or File')
    parser.add_argument('--reference', type=str, default=None, help='Reference Image Path (Optional)')
    # output은 "루트"로 받고, 실행 시각 폴더를 그 아래에 자동 생성
    parser.add_argument('--output', type=str, default='test_io/outputs', help='Output Root Directory')
    parser.add_argument('--config', type=str, default='pose_transfer/config/default.yaml', help='Config Path')
    
    args = parser.parse_args()
    
    # 1. 경로 설정
    source_input = Path(args.source)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # 실행마다 timestamp 폴더 생성
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output Run Dir: {out_dir}")

    # 소스 파일 목록 확보
    if source_input.is_dir():
        src_files = get_image_files(source_input)
        if not src_files:
            print(f"❌ '{source_input}' 폴더에 이미지 파일이 없습니다.")
            return
        print(f"📂 Batch Mode: '{source_input}' 폴더 내 {len(src_files)}개 이미지 처리")
    elif source_input.exists():
        src_files = [source_input]
        print(f"📄 Single Mode: {source_input} 처리")
    else:
        print(f"❌ Source 경로를 찾을 수 없습니다: {source_input}")
        return

    # 2. 파이프라인 초기화
    config_path = Path(args.config)
    yaml_config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        config = PipelineConfig()

    pipeline = PoseTransferPipeline(config, yaml_config=yaml_config)

    # 3. Reference 로드 (옵션)
    ref_data = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            print(f"💃 Reference Loading: {ref_path}")
            ref_img = load_image(ref_path)
            ref_data = (ref_img, ref_path)
            
            # Reference 분석 결과도 한 번 저장
            ref_kpts, ref_scores, _, ref_size = pipeline.extract_pose(ref_img)
            ref_scores_f = ref_scores
            r_skel = pipeline.renderer.render_skeleton_only((ref_size[0], ref_size[1], 3), ref_kpts, ref_scores_f)
            save_image(r_skel, str(out_dir / "reference_sk.jpg"))
        else:
            print(f"❌ Reference 파일을 찾을 수 없어 '추출 모드'로 진행합니다: {ref_path}")

    print("="*60)
    
    # 4. 일괄 처리 루프
    for src_path in src_files:
        process_image(
            pipeline, 
            src_path, 
            out_dir, 
            ref_data, 
            config.kpt_threshold,
            yaml_config  # 설정 전달
        )

    print("="*60)
    print(f"✨ 모든 작업 완료! 결과물 위치: {out_dir}")

if __name__ == "__main__":
    main()