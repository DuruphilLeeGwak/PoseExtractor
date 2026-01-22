from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

# Ensure local imports work regardless of cwd / launcher wrappers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_transfer.pipeline import PipelineConfig, PoseTransferPipeline
from pose_transfer.extractors.keypoint_constants import (
    BODY_KEYPOINTS,
    LEFT_HAND_START_IDX,
    RIGHT_HAND_START_IDX,
)
from pose_transfer.utils.io import load_image


def _debug_tools_enabled(yaml_cfg: dict, key: str) -> bool:
        tools = (yaml_cfg or {}).get("debug_tools", {})
        if tools.get("enabled", True) is False:
                return False
        return tools.get(key, True) is True


def _write_readme(out_dir: Path) -> None:
        readme = """# HandRefiner 디버그 출력 설명

이 폴더는 `HandRefiner`가 손을 어떻게 ROI로 잡고, 업스케일이 실제로 발동하는지, 그리고 refined 결과가 원본 대비 어떻게 달라지는지를 확인하기 위한 디버그 산출물입니다.

## 파일 종류 (프레임/손별)

- `*_roi.jpg`
    - 원본 전체 이미지 위에 손 ROI 박스를 그린 이미지입니다.
    - 흰 점: wrist(Body wrist), 회색 점: elbow(Body elbow)
    - 노란 사각형: HandRefiner가 추정한 ROI

- `*_crop.jpg`
    - ROI 영역만 crop한 이미지입니다.
    - 빨간 점: 원본 wholebody 추론에서의 손 21점(ROI 좌표계로 이동된 상태)

- `*_upscaled.jpg` (업스케일 발동 시에만 생성)
    - ROI crop을 업스케일한 이미지입니다.
    - 노란 점: 업스케일된 crop에서 재추론한 손 21점(업스케일 crop 좌표계)

- `*_compare.jpg` (업스케일 발동 시에만 생성)
    - 같은 ROI crop 위에 원본 손(빨강)과 refined 손(초록)을 함께 표시합니다.
    - 흰 선: 각 점의 이동(원본→refined)

- `*_info.json`
    - 수치 요약입니다.
    - 주요 필드:
        - `roi`: (x1,y1,x2,y2)
        - `needs_upscale`, `scale_factor`
        - `original_valid`, `refined_valid`, `used_refined`
        - `mean_displacement_px`: (유효점 기준) 평균 이동 거리

## 참고

이 디버그는 GhostFilter 판단이 아니라, HandRefiner 자체를 검증하기 위한 것입니다.
"""
        (out_dir / "README.md").write_text(readme, encoding="utf-8")


def _draw_points(
    img_bgr: np.ndarray,
    pts: np.ndarray,
    scores: np.ndarray | None,
    color: tuple[int, int, int],
    radius: int = 3,
    thr: float = 0.0,
) -> None:
    h, w = img_bgr.shape[:2]
    for i, (x, y) in enumerate(pts):
        if scores is not None and float(scores[i]) < thr:
            continue
        px, py = int(round(float(x))), int(round(float(y)))
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        cv2.circle(img_bgr, (px, py), radius, color, -1)


def _draw_lines_between(
    img_bgr: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[int, int, int],
    thr_a: np.ndarray | None = None,
    thr_b: np.ndarray | None = None,
    thr: float = 0.0,
) -> None:
    h, w = img_bgr.shape[:2]
    for i in range(min(len(a), len(b))):
        if thr_a is not None and float(thr_a[i]) < thr:
            continue
        if thr_b is not None and float(thr_b[i]) < thr:
            continue
        x1, y1 = a[i]
        x2, y2 = b[i]
        p1 = (int(round(float(x1))), int(round(float(y1))))
        p2 = (int(round(float(x2))), int(round(float(y2))))
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
            continue
        cv2.line(img_bgr, p1, p2, color, 1, cv2.LINE_AA)


def _safe_imwrite(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug HandRefiner: show ROI/crop/upscale and pre/post hand keypoints"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Output run folder containing *_bg.jpg",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=r"d:\2025\pose_extractor\pose_transfer\config\default.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--out_subdir",
        type=str,
        default="_debug_hand_refiner",
        help="Subfolder under run_dir to write debug artifacts",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10,
        help="Max number of *_bg frames to process (sorted by stem)",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.3,
        help="Threshold for counting/plotting 'valid' hand points",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not _debug_tools_enabled(yaml_cfg, "hand_refiner_debug"):
        print("[skip] debug_tools disabled: hand_refiner_debug")
        return

    _write_readme(out_dir)

    # Load pipeline/extractor
    pipe_cfg = PipelineConfig.from_yaml(str(config_path))
    pipeline = PoseTransferPipeline(pipe_cfg, yaml_config=yaml_cfg)

    # We want to inspect HandRefiner itself, not GhostFilter decisions.
    # HandRefiner is run inside extract_pose, so we will call extractor directly and run refiner manually.
    extractor = pipeline.extractor
    person_filter = pipeline.person_filter
    refiner = pipeline.hand_refiner

    stems = sorted({p.stem.replace("_bg", "") for p in run_dir.glob("*_bg.jpg")})
    if not stems:
        raise SystemExit(f"No *_bg.jpg found in {run_dir}")

    stems = stems[: max(1, int(args.max_frames))]

    sides = [
        ("LHand", True, LEFT_HAND_START_IDX, BODY_KEYPOINTS["left_wrist"], BODY_KEYPOINTS["left_elbow"]),
        ("RHand", False, RIGHT_HAND_START_IDX, BODY_KEYPOINTS["right_wrist"], BODY_KEYPOINTS["right_elbow"]),
    ]

    for stem in stems:
        bg_p = run_dir / f"{stem}_bg.jpg"
        img_rgb = load_image(bg_p)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_rgb.shape[:2]

        all_kpts, all_scores = extractor.extract(img_rgb)
        if len(all_kpts) == 0:
            continue

        if getattr(pipeline.config, "filter_enabled", True) and len(all_kpts) > 1:
            kpts, scores, person_idx, _ = person_filter.select_main_person(all_kpts, all_scores, (h, w))
        else:
            kpts, scores, person_idx = all_kpts[0], all_scores[0], 0

        for name, is_left, hand_start, wrist_idx, elbow_idx in sides:
            hand_kpts = kpts[hand_start : hand_start + 21]
            hand_scores = scores[hand_start : hand_start + 21]

            roi = refiner.estimate_hand_roi(kpts, scores, is_left=is_left, image_shape=(h, w))

            info: dict = {
                "stem": stem,
                "hand": name,
                "person_idx": int(person_idx),
                "image_wh": [int(w), int(h)],
                "wrist_score": float(scores[wrist_idx]),
                "elbow_score": float(scores[elbow_idx]),
                "roi": None,
                "needs_upscale": False,
                "scale_factor": 1.0,
                "original_valid": int(np.sum(hand_scores >= float(args.score_thr))),
                "refined_valid": None,
                "used_refined": False,
                "mean_displacement_px": None,
                "notes": [],
            }

            # Always write a ROI overlay on the full image so you can verify the ROI location.
            roi_overlay = img_bgr.copy()
            wx, wy = int(round(float(kpts[wrist_idx][0]))), int(round(float(kpts[wrist_idx][1])))
            ex, ey = int(round(float(kpts[elbow_idx][0]))), int(round(float(kpts[elbow_idx][1])))
            cv2.circle(roi_overlay, (wx, wy), 6, (255, 255, 255), -1)
            cv2.circle(roi_overlay, (ex, ey), 6, (200, 200, 200), -1)
            cv2.putText(
                roi_overlay,
                f"{name} wrist/elbow  person={person_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if roi is None:
                info["notes"].append("roi=None (wrist below confidence_threshold or ROI too small)")
                _safe_imwrite(out_dir / f"{stem}_{name}_roi.jpg", roi_overlay)
                (out_dir / f"{stem}_{name}_info.json").write_text(
                    json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                continue

            x1, y1, x2, y2 = roi
            info["roi"] = [int(x1), int(y1), int(x2), int(y2)]
            cv2.rectangle(roi_overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)

            needs_upscale, scale_factor = refiner.check_needs_upscale(roi)
            info["needs_upscale"] = bool(needs_upscale)
            info["scale_factor"] = float(scale_factor)

            _safe_imwrite(out_dir / f"{stem}_{name}_roi.jpg", roi_overlay)

            # Crop view (original scale) with original hand keypoints
            crop_bgr = img_bgr[y1:y2, x1:x2].copy()
            crop_hand_kpts = hand_kpts.copy()
            crop_hand_kpts[:, 0] -= x1
            crop_hand_kpts[:, 1] -= y1
            _draw_points(crop_bgr, crop_hand_kpts, hand_scores, (0, 0, 255), radius=3, thr=float(args.score_thr))
            _safe_imwrite(out_dir / f"{stem}_{name}_crop.jpg", crop_bgr)

            if not needs_upscale:
                info["notes"].append("no upscale needed (ROI >= min_hand_size)")
                (out_dir / f"{stem}_{name}_info.json").write_text(
                    json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                continue

            upscaled_crop, transform_info = refiner.crop_and_upscale(img_rgb, roi, scale_factor)
            up_bgr = cv2.cvtColor(upscaled_crop, cv2.COLOR_RGB2BGR)

            # Re-run whole-body extractor on the upscaled crop
            all_kpts2, all_scores2 = extractor.extract(upscaled_crop)
            if len(all_kpts2) == 0:
                info["notes"].append("extractor returned 0 people on upscaled crop")
                _safe_imwrite(out_dir / f"{stem}_{name}_upscaled.jpg", up_bgr)
                (out_dir / f"{stem}_{name}_info.json").write_text(
                    json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                continue

            new_hand_kpts = all_kpts2[0][hand_start : hand_start + 21]
            new_hand_scores = all_scores2[0][hand_start : hand_start + 21]

            # Visualize the extracted hand on the upscaled crop (crop coordinates)
            _draw_points(up_bgr, new_hand_kpts, new_hand_scores, (0, 255, 255), radius=3, thr=float(args.score_thr))
            _safe_imwrite(out_dir / f"{stem}_{name}_upscaled.jpg", up_bgr)

            refined_kpts, refined_scores = refiner.transform_keypoints_back(
                new_hand_kpts, new_hand_scores, transform_info, is_left=is_left
            )

            refined_valid = int(np.sum(refined_scores >= float(args.score_thr)))
            info["refined_valid"] = refined_valid
            info["used_refined"] = bool(refined_valid > info["original_valid"])

            # Compare overlay (in original crop coords)
            refined_in_crop = refined_kpts.copy()
            refined_in_crop[:, 0] -= x1
            refined_in_crop[:, 1] -= y1

            compare = crop_bgr.copy()
            # original = red, refined = green, connections = white
            _draw_points(compare, crop_hand_kpts, hand_scores, (0, 0, 255), radius=3, thr=float(args.score_thr))
            _draw_points(compare, refined_in_crop, refined_scores, (0, 255, 0), radius=3, thr=float(args.score_thr))
            _draw_lines_between(compare, crop_hand_kpts, refined_in_crop, (255, 255, 255), thr_a=hand_scores, thr_b=refined_scores, thr=float(args.score_thr))

            # displacement metric for points considered valid in either set
            mask = (hand_scores >= float(args.score_thr)) | (refined_scores >= float(args.score_thr))
            if np.any(mask):
                disp = np.linalg.norm((refined_kpts - hand_kpts)[mask], axis=1)
                info["mean_displacement_px"] = float(np.mean(disp))

            cv2.putText(
                compare,
                f"orig_valid={info['original_valid']} refined_valid={refined_valid} scale={scale_factor:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                compare,
                "RED=original  GREEN=refined",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            _safe_imwrite(out_dir / f"{stem}_{name}_compare.jpg", compare)

            (out_dir / f"{stem}_{name}_info.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        print(f"Wrote: {stem} (person_idx={person_idx})")


if __name__ == "__main__":
    main()
