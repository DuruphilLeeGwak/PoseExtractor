# 표준 라이브러리 임포트
import os
import sys
from pathlib import Path


def main() -> int:
    """
    GhostFilter에 의해 엉덩이(hip) 키포인트가 제거된 이미지를 스캔하는 메인 함수
    
    Returns:
        int: 프로그램 종료 코드 (0: 정상 종료)
    """
    # 프로젝트 루트 디렉토리 설정
    root = Path(r"d:\\2025\\pose_extractor")
    os.chdir(root)
    # 모듈 임포트를 위해 경로 추가
    sys.path.insert(0, str(root))

    # 포즈 전송 관련 모듈 임포트
    from pose_transfer.pipeline import PoseTransferPipeline
    from pose_transfer.config import load_config
    from pose_transfer.utils.io import load_image

    # 설정 파일 로드 및 파이프라인 초기화
    cfg, yaml_cfg = load_config(str(root / "pose_transfer" / "config" / "default.yaml"))
    pipe = PoseTransferPipeline(config=cfg, yaml_config=yaml_cfg)

    # 입력 이미지 디렉토리에서 이미지 파일 목록 가져오기
    inputs = root / "test_io" / "inputs"
    files = [p for p in inputs.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    # 파일명 기준으로 정렬
    files.sort(key=lambda p: p.name)

    # 엉덩이 키포인트 인덱스 정의 (왼쪽, 오른쪽)
    HIP_L, HIP_R = 11, 12
    # 신뢰도 임계값 (0.1 이하는 검출되지 않은 것으로 간주)
    thr = 0.1

    print(f"Scanning {len(files)} images for hip(11/12) removed by GhostFilter...")

    # 엉덩이가 제거된 이미지 개수 카운터
    found = 0
    # 각 이미지 파일에 대해 반복
    for p in files:
        # 이미지 로드
        img = load_image(str(p))
        # 포즈 추출 (사람 필터링 적용)
        kpts, scores, _, image_size = pipe.extract_pose(img, filter_person=True)

        # 초기에 엉덩이가 검출되지 않은 경우는 건너뛰기
        # (애초에 검출되지 않은 것은 제거된 것이 아니므로)
        if scores[HIP_L] <= thr and scores[HIP_R] <= thr:
            continue

        # GhostFilter 적용 후 결과 가져오기
        filt_scores, filt_res = pipe._apply_ghost_filter_single(kpts, scores, image_size)

        # 제거된 엉덩이 키포인트 목록
        removed = []
        # 왼쪽/오른쪽 엉덩이에 대해 각각 확인
        for idx, name in [(HIP_L, "Lhip"), (HIP_R, "Rhip")]:
            # 원본에서는 검출되었으나(thr 초과) 필터링 후 제거된 경우(거의 0)
            if scores[idx] > thr and filt_scores[idx] <= 1e-9:
                removed.append((idx, name, filt_res.removal_reasons.get(idx)))

        # 제거된 키포인트가 있는 경우 출력
        if removed:
            found += 1
            print(f"\n- {p.name}")
            # 각 제거된 키포인트의 상세 정보 출력
            for idx, nm, reason in removed:
                x, y = kpts[idx]
                print(
                    f"  {nm} idx={idx} orig={scores[idx]:.3f} -> 0 at ({x:.1f},{y:.1f}) reason={reason}"
                )

    # 최종 결과 출력
    print(f"\nDone. Found {found} images with hip removed.")
    return 0


if __name__ == "__main__":
    # 메인 함수 실행 및 종료 코드 반환
    raise SystemExit(main())
