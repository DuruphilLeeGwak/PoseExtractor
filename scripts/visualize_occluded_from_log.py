from __future__ import annotations

import argparse
import re
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


@dataclass
class Step35Line:
    hand: str  # LHand or RHand
    verdict: str
    active: int
    near: int
    far: int
    near_ratio: float
    thr: float
    min_pts: int
    min_near: float
    max_far: int
    wrist_score: float
    elbow_score: float
    forearm: float
    base_r: float
    far_r: float


_STEP35_RE = re.compile(
    r"^\[HAND\]\[Step3\.5\]\s+(?P<hand>LHand|RHand)\s+verdict=(?P<verdict>\S+)\s+"
    r"active=(?P<active>\d+)/21\s+near=(?P<near>\d+)\s+far=(?P<far>\d+)\s+"
    r"near_ratio=(?P<near_ratio>[0-9.]+)\s+thr=(?P<thr>[0-9.]+)\s+min_pts=(?P<min_pts>\d+)\s+"
    r"min_near=(?P<min_near>[0-9.]+)\s+max_far=(?P<max_far>\d+)\s+"
    r"wrist_score=(?P<wrist_score>[0-9.]+)\s+elbow_score=(?P<elbow_score>[0-9.]+)\s+"
    r"forearm=(?P<forearm>[0-9.]+)\s+base_r=(?P<base_r>[0-9.]+)\s+far_r=(?P<far_r>[0-9.]+)\s*$"
)


def parse_step35_for_file(debug_txt: Path, filename: str) -> list[Step35Line]:
    lines = debug_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    out: list[Step35Line] = []
    in_block = False

    for line in lines:
        if line.startswith("FILE:"):
            in_block = line.strip() == f"FILE: {filename}"
            continue
        if not in_block:
            continue

        m = _STEP35_RE.match(line.strip())
        if not m:
            continue

        out.append(
            Step35Line(
                hand=m.group("hand"),
                verdict=m.group("verdict"),
                active=int(m.group("active")),
                near=int(m.group("near")),
                far=int(m.group("far")),
                near_ratio=float(m.group("near_ratio")),
                thr=float(m.group("thr")),
                min_pts=int(m.group("min_pts")),
                min_near=float(m.group("min_near")),
                max_far=int(m.group("max_far")),
                wrist_score=float(m.group("wrist_score")),
                elbow_score=float(m.group("elbow_score")),
                forearm=float(m.group("forearm")),
                base_r=float(m.group("base_r")),
                far_r=float(m.group("far_r")),
            )
        )

        # stop early if we already found both hands
        if len(out) >= 2:
            pass

    return out


def load_openpose_hand(kp_json: dict, hand: str) -> tuple[np.ndarray, np.ndarray]:
    people = kp_json.get("people") or []
    person = people[0] if people else {}

    key = "hand_left_keypoints_2d" if hand == "LHand" else "hand_right_keypoints_2d"
    arr = np.array(person.get(key, []), dtype=np.float32)
    if arr.size != 21 * 3:
        raise ValueError(f"Missing/invalid {key} in JSON")

    xs = arr[0::3]
    ys = arr[1::3]
    cs = arr[2::3]
    pts = np.stack([xs, ys], axis=1)
    return pts, cs


def load_openpose_body_point(kp_json: dict, body_idx: int) -> tuple[float, float, float]:
    people = kp_json.get("people") or []
    person = people[0] if people else {}
    arr = np.array(person.get("pose_keypoints_2d", []), dtype=np.float32)
    if arr.size < (body_idx + 1) * 3:
        raise ValueError("Missing/invalid pose_keypoints_2d in JSON")
    x = float(arr[body_idx * 3 + 0])
    y = float(arr[body_idx * 3 + 1])
    c = float(arr[body_idx * 3 + 2])
    return x, y, c


def draw_visual(
    bg_bgr: np.ndarray,
    hand_pts: np.ndarray,
    wrist_xy: tuple[float, float],
    step: Step35Line,
    *,
    min_hand_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    out = bg_bgr.copy()
    h, w = out.shape[:2]

    wx, wy = int(round(wrist_xy[0])), int(round(wrist_xy[1]))

    # Circles
    cv2.circle(out, (wx, wy), int(round(step.base_r)), (0, 255, 255), 3)
    cv2.circle(out, (wx, wy), int(round(step.far_r)), (0, 0, 255), 2)
    cv2.circle(out, (wx, wy), 6, (255, 255, 255), -1)

    # Points colored by distance to wrist
    near = 0
    far = 0
    active = 21  # in occluded cases we're visualizing, all 21 were active

    dists = np.linalg.norm(hand_pts - np.array([[wrist_xy[0], wrist_xy[1]]], dtype=np.float32), axis=1)
    for i, (pt, d) in enumerate(zip(hand_pts, dists)):
        px, py = int(round(float(pt[0]))), int(round(float(pt[1])))
        if not (0 <= px < w and 0 <= py < h):
            continue
        if d <= step.base_r:
            col = (0, 200, 0)
            near += 1
        elif d >= step.far_r:
            col = (0, 0, 255)
            far += 1
        else:
            col = (0, 165, 255)
        cv2.circle(out, (px, py), 5, col, -1)
        cv2.putText(out, str(i), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    near_ratio = near / max(1, active)

    # HandRefiner 관여 여부(대략)
    # HandRefiner ROI 크기 ~ forearm * 0.6 (elbow가 유효할 때)
    est_roi = step.forearm * 0.6 if step.elbow_score >= 0.3 else float(min_hand_size) * 1.5
    refiner_note = (
        "HandRefiner: 영향 적음(ROI 충분히 큼)" if est_roi >= min_hand_size else "HandRefiner: ROI가 작아 보정 가능성"
    )

    causes = []
    if near_ratio < step.min_near:
        causes.append(f"near_ratio {near_ratio:.2f} < 최소 {step.min_near:.2f}")
    if far > step.max_far:
        causes.append(f"far_count {far} > 최대 {step.max_far}")

    lines = [
        f"{step.hand} Step3.5 verdict={step.verdict}",
        f"active={active}/21 thr={step.thr:.2f} min_pts={step.min_pts}",
        f"near={near} far={far} near_ratio={near_ratio:.3f} (log={step.near_ratio:.3f})",
        f"base_r={step.base_r:.1f} far_r={step.far_r:.1f} max_far={step.max_far}",
        f"wrist_score={step.wrist_score:.2f} elbow_score={step.elbow_score:.2f} forearm={step.forearm:.1f}",
        f"{refiner_note} (추정 ROI~{est_roi:.0f}px, min_hand_size={min_hand_size})",
    ]
    if causes:
        lines.append("원인: " + ", ".join(causes))

    # Text box
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

    return out, dists


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
            "기존 출력물만으로 GhostFilter Step 3.5(occluded) 판정을 시각화합니다: "
            "*_bg.jpg, *_kp.json, _ghostfilter_hand_debug.txt (모델 재실행 없음)."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--stem", type=str, required=True, help="Image stem (without extension)")
    parser.add_argument("--hand", type=str, choices=["LHand", "RHand", "auto"], default="auto")
    parser.add_argument("--out_subdir", type=str, default="_debug_occluded_visual")
    parser.add_argument("--crop_half", type=int, default=600)
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "pose_transfer" / "config" / "default.yaml"),
        help="YAML config path (for min_hand_size)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    stem = args.stem

    bg_p = run_dir / f"{stem}_bg.jpg"
    kp_p = run_dir / f"{stem}_kp.json"
    dbg_p = run_dir / "_ghostfilter_hand_debug.txt"

    if not bg_p.exists():
        raise SystemExit(f"Missing bg: {bg_p}")
    if not kp_p.exists():
        raise SystemExit(f"Missing kp json: {kp_p}")
    if not dbg_p.exists():
        raise SystemExit(f"Missing debug txt: {dbg_p}")

    yaml_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    min_hand_size = int(((yaml_cfg or {}).get("hand_refinement") or {}).get("min_hand_size", 48))

    step_lines = parse_step35_for_file(dbg_p, f"{stem}.jpg")
    if not step_lines:
        raise SystemExit(f"No Step3.5 lines found for FILE: {stem}.jpg in {dbg_p}")

    if args.hand == "auto":
        # pick the hand with verdict REMOVE if present, else first
        pick = None
        for s in step_lines:
            if s.verdict.startswith("REMOVE"):
                pick = s
                break
        step = pick or step_lines[0]
    else:
        matches = [s for s in step_lines if s.hand == args.hand]
        if not matches:
            raise SystemExit(f"No Step3.5 line for hand={args.hand}")
        step = matches[0]

    img = cv2.imread(str(bg_p), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Unreadable image: {bg_p}")

    kp_json = yaml.safe_load(kp_p.read_text(encoding="utf-8")) if kp_p.suffix.lower() in {".yaml", ".yml"} else None
    if kp_json is None:
        import json
        kp_json = json.loads(kp_p.read_text(encoding="utf-8"))

    hand_pts, _hand_scores = load_openpose_hand(kp_json, step.hand)

    # Body indices: COCO body (OpenPose): left_elbow=7, left_wrist=9, right_elbow=8, right_wrist=10
    wrist_idx = 9 if step.hand == "LHand" else 10
    wx, wy, _wc = load_openpose_body_point(kp_json, wrist_idx)

    vis, _ = draw_visual(img, hand_pts, (wx, wy), step, min_hand_size=min_hand_size)
    crop = crop_around(vis, int(round(wx)), int(round(wy)), int(args.crop_half))

    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_img = out_dir / f"{stem}_{step.hand}_step35_from_log.jpg"
    out_crop = out_dir / f"{stem}_{step.hand}_step35_from_log_crop.jpg"

    cv2.imwrite(str(out_img), vis)
    cv2.imwrite(str(out_crop), crop)

    print(f"Wrote: {out_img}")
    print(f"Wrote: {out_crop}")


if __name__ == "__main__":
    main()
