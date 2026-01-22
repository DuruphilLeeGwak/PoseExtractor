from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from pose_transfer.extractors.keypoint_constants import BODY_KEYPOINTS
from pose_transfer.utils.io import load_image


@dataclass
class Step35Metrics:
    verdict: str
    active: int
    near: int
    far: int
    near_ratio: float | None
    wrist_score: float
    elbow_score: float
    forearm: float
    base_radius: float
    far_radius: float
    thr: float
    min_pts: int
    min_near_ratio: float
    max_far: int
    details: list[tuple[int, float, float]]  # (idx, score, dist)


def compute_step35(
    kpts: np.ndarray,
    scores: np.ndarray,
    *,
    name: str,
    elbow_idx: int,
    wrist_idx: int,
    start: int,
    end: int,
    thr: float,
    min_pts: int,
    min_near_ratio: float,
    radius_ratio: float,
    min_radius: float,
    far_ratio: float,
    max_far: int,
) -> Step35Metrics:
    wrist_score = float(scores[wrist_idx]) if wrist_idx < len(scores) else 0.0
    elbow_score = float(scores[elbow_idx]) if elbow_idx < len(scores) else 0.0

    if wrist_score < 0.3:
        return Step35Metrics(
            verdict="SKIP(wrist<0.3)",
            active=0,
            near=0,
            far=0,
            near_ratio=None,
            wrist_score=wrist_score,
            elbow_score=elbow_score,
            forearm=0.0,
            base_radius=float(min_radius),
            far_radius=float(min_radius) * 1.5,
            thr=thr,
            min_pts=min_pts,
            min_near_ratio=float(min_near_ratio),
            max_far=max_far,
            details=[],
        )

    wrist = kpts[wrist_idx]

    if elbow_score >= 0.3:
        forearm = float(np.linalg.norm(wrist - kpts[elbow_idx]))
    else:
        forearm = 0.0

    base_radius = max(float(min_radius), forearm * float(radius_ratio)) if forearm > 1.0 else float(min_radius)
    far_radius = max(base_radius * 1.5, forearm * float(far_ratio)) if forearm > 1.0 else base_radius * 1.5

    active = 0
    near = 0
    far = 0
    details: list[tuple[int, float, float]] = []

    for idx in range(start, end + 1):
        c = float(scores[idx]) if idx < len(scores) else 0.0
        if c < thr:
            continue
        active += 1
        d = float(np.linalg.norm(kpts[idx] - wrist))
        details.append((idx, c, d))
        if d <= base_radius:
            near += 1
        if d >= far_radius:
            far += 1

    if active < int(min_pts):
        return Step35Metrics(
            verdict="SKIP(active<min_pts)",
            active=active,
            near=near,
            far=far,
            near_ratio=None,
            wrist_score=wrist_score,
            elbow_score=elbow_score,
            forearm=forearm,
            base_radius=base_radius,
            far_radius=far_radius,
            thr=thr,
            min_pts=min_pts,
            min_near_ratio=float(min_near_ratio),
            max_far=max_far,
            details=details,
        )

    near_ratio = near / max(1, active)

    verdict = "KEEP"
    if near_ratio < float(min_near_ratio) or far > int(max_far):
        verdict = "REMOVE"

    return Step35Metrics(
        verdict=verdict,
        active=active,
        near=near,
        far=far,
        near_ratio=float(near_ratio),
        wrist_score=wrist_score,
        elbow_score=elbow_score,
        forearm=forearm,
        base_radius=base_radius,
        far_radius=far_radius,
        thr=thr,
        min_pts=min_pts,
        min_near_ratio=float(min_near_ratio),
        max_far=int(max_far),
        details=details,
    )


def draw_step35_overlay(
    img_bgr: np.ndarray,
    kpts: np.ndarray,
    scores: np.ndarray,
    metrics: Step35Metrics,
    *,
    wrist_idx: int,
    start: int,
    end: int,
    title: str,
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Wrist center
    if wrist_idx < len(kpts):
        wrist = kpts[wrist_idx]
        wx, wy = int(round(float(wrist[0]))), int(round(float(wrist[1])))
    else:
        wx, wy = w // 2, h // 2

    # Radii
    cv2.circle(out, (wx, wy), int(round(metrics.base_radius)), (0, 255, 255), 3)
    cv2.circle(out, (wx, wy), int(round(metrics.far_radius)), (0, 0, 255), 2)
    cv2.circle(out, (wx, wy), 6, (255, 255, 255), -1)

    # Finger points (only active points as Step3.5 sees)
    for idx in range(start, end + 1):
        c = float(scores[idx]) if idx < len(scores) else 0.0
        if c < metrics.thr:
            continue
        x, y = kpts[idx]
        px, py = int(round(float(x))), int(round(float(y)))
        if not (0 <= px < w and 0 <= py < h):
            continue
        d = float(np.linalg.norm(kpts[idx] - kpts[wrist_idx])) if wrist_idx < len(kpts) else 0.0
        if d <= metrics.base_radius:
            col = (0, 200, 0)  # near
        elif d >= metrics.far_radius:
            col = (0, 0, 255)  # far
        else:
            col = (0, 165, 255)  # mid
        cv2.circle(out, (px, py), 5, col, -1)

    # Text box
    lines: list[str] = [
        title,
        f"verdict={metrics.verdict}",
        f"active={metrics.active}/21 (thr={metrics.thr:.2f}, min_pts={metrics.min_pts})",
        f"near={metrics.near} far={metrics.far} max_far={metrics.max_far}",
        f"near_ratio={metrics.near_ratio if metrics.near_ratio is not None else 'NA'} min_near={metrics.min_near_ratio:.2f}",
        f"wrist_score={metrics.wrist_score:.2f} elbow_score={metrics.elbow_score:.2f}",
        f"forearm={metrics.forearm:.1f} base_r={metrics.base_radius:.1f} far_r={metrics.far_radius:.1f}",
    ]

    # Determine cause
    if metrics.verdict == "REMOVE" and metrics.near_ratio is not None:
        causes = []
        if metrics.near_ratio < metrics.min_near_ratio:
            causes.append("near_ratio low")
        if metrics.far > metrics.max_far:
            causes.append("too many far outliers")
        if causes:
            lines.append("cause=" + ", ".join(causes))

    # Draw black background
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7
    th = 2
    x0, y0 = 20, 30
    line_h = 26

    widths = []
    for s in lines:
        (tw, _), _b = cv2.getTextSize(s, font, fs, th)
        widths.append(tw)
    box_w = min(w - 40, max(widths) + 20)
    box_h = min(h - 40, len(lines) * line_h + 20)
    cv2.rectangle(out, (x0 - 10, y0 - 22), (x0 - 10 + box_w, y0 - 22 + box_h), (0, 0, 0), -1)

    y = y0
    for s in lines:
        cv2.putText(out, s, (x0, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        y += line_h

    return out


def crop_around(img: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return img[y1:y2, x1:x2].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize GhostFilter Step 3.5 occluded decision with concrete numbers, "
            "and compare pre/post HandRefiner to check correlation."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Run folder containing *_bg.jpg")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "pose_transfer" / "config" / "default.yaml"),
        help="YAML config path",
    )
    parser.add_argument("--out_subdir", type=str, default="_debug_occluded_explain", help="Output subdir under run_dir")
    parser.add_argument("--max_frames", type=int, default=50, help="Max frames to process")
    parser.add_argument("--only_removed", action="store_true", help="Only write outputs when verdict is REMOVE")
    parser.add_argument("--compare_hand_refiner", action="store_true", help="Write both pre/post HandRefiner comparisons")
    parser.add_argument("--crop_half", type=int, default=600, help="Crop half-size around wrist")
    parser.add_argument("--stem", type=str, default=None, help="Process only stems containing this substring")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pipe_cfg = PipelineConfig.from_yaml(str(config_path))
    pipeline = PoseTransferPipeline(pipe_cfg, yaml_config=yaml_cfg)

    gf = pipeline.ghost_filter
    # Step3.5 thresholds
    thr = float(gf.config.hand_finger_min_confidence)
    min_pts = int(gf.config.hand_min_finger_points)
    radius_ratio = float(gf.config.hand_wrist_radius_ratio)
    min_radius = float(gf.config.hand_wrist_min_radius_px)
    min_near_ratio = float(gf.config.hand_min_near_ratio)
    far_ratio = float(gf.config.hand_far_outlier_ratio)
    max_far = int(gf.config.hand_max_far_points)

    sides = [
        ("LHand", BODY_KEYPOINTS["left_elbow"], BODY_KEYPOINTS["left_wrist"], 91, 111),
        ("RHand", BODY_KEYPOINTS["right_elbow"], BODY_KEYPOINTS["right_wrist"], 112, 132),
    ]

    stems = sorted({p.stem.replace("_bg", "") for p in run_dir.glob("*_bg.jpg")})
    if args.stem:
        stems = [s for s in stems if args.stem in s]
    stems = stems[: max(1, int(args.max_frames))]

    report_txt = out_dir / "report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("GhostFilter Step 3.5 occluded 설명\n")
        f.write("\n")
        f.write("용어집(파일 상단에 1회 기록)\n")
        f.write("- verdict: KEEP / REMOVE / SKIP\n")
        f.write("  - SKIP: Step3.5 gating(사전 조건) 실패 (예: wrist_score < thr 또는 active < min_pts)\n")
        f.write("  - REMOVE: 손가락 점들을 신뢰 불가로 판단하여 제거(손가락 scores=0 처리)\n")
        f.write("- thr: 손 키포인트가 active로 집계되기 위한 최소 confidence\n")
        f.write("- active: confidence >= thr 인 손 키포인트 개수(0..21)\n")
        f.write("- min_pts: Step3.5를 수행하기 위한 최소 active 개수\n")
        f.write("- wrist_score / elbow_score: 바디 키포인트(손목/팔꿈치) confidence\n")
        f.write("- forearm: 팔꿈치-손목 거리(이미지 좌표계 픽셀)\n")
        f.write("- base_r: 손목 중심 near 판정 반지름\n")
        f.write("  - base_r = max(min_radius, forearm * radius_ratio)\n")
        f.write("- near: active 점 중 distance(wrist, point) <= base_r 인 점 개수\n")
        f.write("- near_ratio: near / active\n")
        f.write("- min_near_ratio: 통과에 필요한 최소 near_ratio\n")
        f.write("- far_r: far outlier 반지름 (far_r = base_r * far_ratio)\n")
        f.write("- far: active 점 중 distance >= far_r 인 점 개수\n")
        f.write("- max_far: 허용 가능한 far outlier 최대 개수\n")
        f.write("- farthest(active): 가장 멀리 있는 active 점 상위 목록 (idx<k>:d=<거리>,c=<confidence>)\n")
        f.write("\n")
        f.write("실행 설정\n")
        f.write(f"run_dir={run_dir}\n")
        f.write(
            f"thr={thr} min_pts={min_pts} min_near_ratio={min_near_ratio} max_far={max_far} "
            f"min_radius={min_radius} radius_ratio={radius_ratio} far_ratio={far_ratio}\n"
        )

    for stem in stems:
        bg_p = run_dir / f"{stem}_bg.jpg"
        img_rgb = load_image(bg_p)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        all_kpts, all_scores = pipeline.extractor.extract(img_rgb)
        if len(all_kpts) == 0:
            continue

        if pipe_cfg.filter_enabled and len(all_kpts) > 1:
            k_pre, s_pre, person_idx, _ = pipeline.person_filter.select_main_person(all_kpts, all_scores, (h, w))
        else:
            k_pre, s_pre, person_idx = all_kpts[0], all_scores[0], 0

        # Post HandRefiner (same as pipeline.extract_pose)
        if pipe_cfg.hand_refinement_enabled:
            k_post, s_post, _info = pipeline.hand_refiner.refine_both_hands(img_rgb, k_pre.copy(), s_pre.copy(), pipeline.extractor)
        else:
            k_post, s_post = k_pre.copy(), s_pre.copy()

        variants = [("post_refiner", k_post, s_post)]
        if args.compare_hand_refiner:
            variants.insert(0, ("pre_refiner", k_pre, s_pre))

        for tag, kpts, scores in variants:
            for name, elbow_idx, wrist_idx, start, end in sides:
                m = compute_step35(
                    kpts,
                    scores,
                    name=name,
                    elbow_idx=elbow_idx,
                    wrist_idx=wrist_idx,
                    start=start,
                    end=end,
                    thr=thr,
                    min_pts=min_pts,
                    radius_ratio=radius_ratio,
                    min_radius=min_radius,
                    far_ratio=far_ratio,
                    max_far=max_far,
                    min_near_ratio=min_near_ratio,
                )

                if args.only_removed and m.verdict != "REMOVE":
                    continue

                title = f"{stem} {name} ({tag}) person={person_idx}"
                vis = draw_step35_overlay(img_bgr, kpts, scores, m, wrist_idx=wrist_idx, start=start, end=end, title=title)

                # Crop around wrist
                wrist = kpts[wrist_idx]
                cx, cy = int(round(float(wrist[0]))), int(round(float(wrist[1])))
                crop = crop_around(vis, cx, cy, int(args.crop_half))

                out_img = out_dir / f"{stem}_{name}_{tag}_step35.jpg"
                out_crop = out_dir / f"{stem}_{name}_{tag}_step35_crop.jpg"
                cv2.imwrite(str(out_img), vis)
                cv2.imwrite(str(out_crop), crop)

                # Append report
                with open(report_txt, "a", encoding="utf-8") as f:
                    f.write("=" * 100 + "\n")
                    f.write(f"stem={stem} name={name} tag={tag} person={person_idx}\n")
                    f.write(
                        f"verdict={m.verdict} active={m.active} near={m.near} far={m.far} "
                        f"near_ratio={m.near_ratio} (min={m.min_near_ratio}) max_far={m.max_far}\n"
                    )
                    f.write(
                        f"wrist_score={m.wrist_score} elbow_score={m.elbow_score} forearm={m.forearm} "
                        f"base_r={m.base_radius} far_r={m.far_radius} thr={m.thr} min_pts={m.min_pts}\n"
                    )
                    if m.details:
                        m.details.sort(key=lambda t: t[2], reverse=True)
                        top = m.details[:10]
                        f.write("farthest(active): " + ", ".join([f"idx{ii}:d={dd:.1f},c={cc:.2f}" for ii, cc, dd in top]) + "\n")

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
