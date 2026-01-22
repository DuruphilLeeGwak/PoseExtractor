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
from pose_transfer.utils.io import load_image


# COCO-WholeBody / OpenPose-style hand keypoint naming (relative 0..20)
# See: pose_transfer.extractors.keypoint_constants.HAND_BONES
HAND_KEYPOINT_NAMES: list[str] = [
    "wrist",  # 0
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_4",
    "index_1",
    "index_2",
    "index_3",
    "index_4",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_4",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_4",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_4",
]


def hand_point_name(metrics: HandOcclusionMetrics, abs_idx: int) -> str:
    rel = abs_idx - metrics.start
    if 0 <= rel < len(HAND_KEYPOINT_NAMES):
        return HAND_KEYPOINT_NAMES[rel]
    return f"hand_{rel}"


@dataclass
class HandOcclusionMetrics:
    name: str
    elbow_idx: int
    wrist_idx: int
    start: int
    end: int
    wrist_score: float
    elbow_score: float
    forearm: float
    base_radius: float
    far_radius: float
    active: int
    near: int
    far: int
    near_ratio: float


def compute_hand_metrics(
    kpts: np.ndarray,
    scores: np.ndarray,
    name: str,
    elbow_idx: int,
    wrist_idx: int,
    start: int,
    end: int,
    thr: float,
    min_pts: int,
    radius_ratio: float,
    min_radius: float,
    far_ratio: float,
) -> HandOcclusionMetrics:
    wrist = kpts[wrist_idx]
    wrist_score = float(scores[wrist_idx])
    elbow_score = float(scores[elbow_idx]) if elbow_idx < len(scores) else 0.0

    if elbow_idx < len(kpts) and elbow_score >= 0.3:
        forearm = float(np.linalg.norm(wrist - kpts[elbow_idx]))
    else:
        forearm = 0.0

    base_radius = max(min_radius, forearm * radius_ratio) if forearm > 1.0 else min_radius
    far_radius = max(base_radius * 1.5, forearm * far_ratio) if forearm > 1.0 else base_radius * 1.5

    active = 0
    near = 0
    far = 0

    for i in range(start, end + 1):
        c = float(scores[i])
        if c < thr:
            continue
        active += 1
        d = float(np.linalg.norm(kpts[i] - wrist))
        if d <= base_radius:
            near += 1
        if d >= far_radius:
            far += 1

    near_ratio = near / max(1, active)

    return HandOcclusionMetrics(
        name=name,
        elbow_idx=elbow_idx,
        wrist_idx=wrist_idx,
        start=start,
        end=end,
        wrist_score=wrist_score,
        elbow_score=elbow_score,
        forearm=forearm,
        base_radius=base_radius,
        far_radius=far_radius,
        active=active,
        near=near,
        far=far,
        near_ratio=near_ratio,
    )


def draw_hand_debug(
    img_bgr: np.ndarray,
    kpts: np.ndarray,
    scores: np.ndarray,
    metrics: HandOcclusionMetrics,
    thr: float,
    min_pts: int,
    min_near_ratio: float,
    max_far: int,
) -> np.ndarray:
    out = img_bgr.copy()

    wrist = kpts[metrics.wrist_idx]
    wx, wy = int(round(wrist[0])), int(round(wrist[1]))

    # Circles: base (near) and far
    cv2.circle(out, (wx, wy), int(round(metrics.base_radius)), (0, 255, 255), 3)  # yellow
    cv2.circle(out, (wx, wy), int(round(metrics.far_radius)), (0, 0, 255), 2)  # red
    cv2.circle(out, (wx, wy), 6, (255, 255, 255), -1)

    # Finger points
    for i in range(metrics.start, metrics.end + 1):
        c = float(scores[i])
        if c < thr:
            continue
        x, y = kpts[i]
        px, py = int(round(x)), int(round(y))
        d = float(np.linalg.norm(kpts[i] - wrist))

        if d <= metrics.base_radius:
            col = (0, 200, 0)  # green: near
        elif d >= metrics.far_radius:
            col = (0, 0, 255)  # red: far outlier
        else:
            col = (0, 165, 255)  # orange: mid

        cv2.circle(out, (px, py), 5, col, -1)

        # Label each keypoint with its name (relative 0..20)
        label = hand_point_name(metrics, i)
        # Keep labels readable but not huge; decouple from the large overlay text scale.
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = float(getattr(draw_hand_debug, "_kp_label_scale", 0.55))
        label_thickness = int(getattr(draw_hand_debug, "_kp_label_thickness", 2))

        h, w = out.shape[:2]
        tx = min(w - 10, px + 8)
        ty = max(18, py - 8)
        # Outline for contrast
        cv2.putText(out, label, (tx, ty), font, label_scale, (0, 0, 0), max(1, label_thickness + 2), cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), font, label_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)

    # Match GhostFilter Step 3.5 behavior as closely as possible:
    # - If wrist score is too low, Step 3.5 is skipped.
    # - If not enough active finger points, Step 3.5 is skipped.
    if metrics.wrist_score < 0.3:
        verdict = "SKIP(wrist<0.3)"
    elif metrics.active < int(min_pts):
        verdict = "SKIP(active<min_pts)"
    else:
        verdict = "KEEP"
        if metrics.near_ratio < min_near_ratio or metrics.far > max_far:
            verdict = "REMOVE"

    lines = [
        f"{metrics.name} verdict={verdict}",
        f"near_ratio={metrics.near_ratio:.2f} (near={metrics.near}/active={metrics.active})  min={min_near_ratio:.2f}",
        f"far_count={metrics.far}  max_far={max_far}",
        f"wrist_score={metrics.wrist_score:.2f} elbow_score={metrics.elbow_score:.2f}",
        f"forearm={metrics.forearm:.1f}  base_r={metrics.base_radius:.1f}  far_r={metrics.far_radius:.1f}",
        f"thr={thr:.2f}",
    ]

    def _wrap_line(text: str, max_px: int, font_scale: float, thickness: int) -> list[tuple[str, int]]:
        """Return [(segment, indent_px)] with continuation lines indented."""
        if not text:
            return [("", 0)]

        font = cv2.FONT_HERSHEY_SIMPLEX

        def text_width(s: str) -> int:
            (w, _), _b = cv2.getTextSize(s, font, font_scale, thickness)
            return int(w)

        # Fast path
        if text_width(text) <= max_px:
            return [(text, 0)]

        words = text.split(" ")
        if len(words) == 1:
            # No spaces: hard wrap by chars
            segs: list[str] = []
            cur = ""
            for ch in text:
                nxt = cur + ch
                if cur and text_width(nxt) > max_px:
                    segs.append(cur)
                    cur = ch
                else:
                    cur = nxt
            if cur:
                segs.append(cur)
            out_segs: list[tuple[str, int]] = []
            for i, s in enumerate(segs):
                out_segs.append((s, 0 if i == 0 else 40))
            return out_segs

        segs2: list[str] = []
        cur = words[0]
        for w in words[1:]:
            nxt = cur + " " + w
            if text_width(nxt) <= max_px:
                cur = nxt
            else:
                segs2.append(cur)
                cur = w
        if cur:
            segs2.append(cur)

        out_segs2: list[tuple[str, int]] = []
        for i, s in enumerate(segs2):
            out_segs2.append((s, 0 if i == 0 else 40))
        return out_segs2

    def _draw_text_block(text_lines: list[str]) -> None:
        h, w = out.shape[:2]
        margin = 20
        # Keep the info box in the top-left (original behavior).
        x0 = 30

        # Text scale: default is intentionally large for high-res frames
        font_scale0 = float(getattr(draw_hand_debug, "_font_scale", 0.8))
        thickness0 = int(getattr(draw_hand_debug, "_thickness", 2))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Keep the box inside the image
        # Ensure we leave at least a minimal width for wrapping.
        min_box_w = 260
        x0 = min(max(margin + 10, x0), max(margin + 10, w - margin - min_box_w))
        max_box_w = max(min_box_w, w - (x0 + margin))

        font_scale = font_scale0
        thickness = max(1, thickness0)

        # Wrap first, then if still too tall, reduce scale a bit until it fits.
        for _ in range(10):
            wrapped: list[tuple[str, int]] = []
            for line in text_lines:
                wrapped.extend(_wrap_line(line, max_box_w - 20, font_scale, thickness))

            # Measure heights. OpenCV uses a baseline Y: glyphs extend up by `th`.
            (tw, th), base = cv2.getTextSize("Ag", font, font_scale, thickness)
            extra_gap = max(6, int(round(6 * (font_scale / 0.8))))
            line_h = int(th + base + extra_gap)

            # Baseline for first line; ensure the top of the glyphs is inside the black box
            # and the box is inside the image.
            pad_top = 10
            pad_bottom = 10
            pad_left = 10
            pad_right = 10

            y0 = max(40, margin + th + pad_top)

            # Compute rectangle bounds from baseline geometry so text never renders above the box.
            box_top = int(y0 - th - pad_top)
            box_bottom = int(y0 + (len(wrapped) - 1) * line_h + base + pad_bottom)

            # Clamp vertically if needed (rare unless image is tiny)
            if box_bottom > h - margin:
                shift = box_bottom - (h - margin)
                y0 = max(margin + th + pad_top, y0 - shift)
                box_top = int(y0 - th - pad_top)
                box_bottom = int(y0 + (len(wrapped) - 1) * line_h + base + pad_bottom)

            if box_bottom <= h - margin and box_top >= margin:
                # Draw
                cv2.rectangle(
                    out,
                    (x0 - pad_left, box_top),
                    (x0 + max_box_w + pad_right, box_bottom),
                    (0, 0, 0),
                    -1,
                )
                y = y0
                for seg, indent_px in wrapped:
                    cv2.putText(
                        out,
                        seg,
                        (x0 + int(indent_px), y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                    y += line_h
                return

            # Too tall: reduce scale and retry
            font_scale *= 0.85
            thickness = max(1, int(round(thickness * 0.85)))

        # Last resort: draw without box (should rarely happen)
        y = y0
        for line in text_lines:
            cv2.putText(out, line, (x0, y), font, max(0.5, font_scale0 * 0.5), (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

    _draw_text_block(lines)

    return out


def crop_around_point(img: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return img[y1:y2, x1:x2].copy()


def hstack_same_height(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Horizontally stack two images, resizing right to match left height."""
    lh = left.shape[0]
    rh = right.shape[0]
    if rh != lh:
        scale = lh / max(1, rh)
        new_w = int(round(right.shape[1] * scale))
        right = cv2.resize(right, (new_w, lh), interpolation=cv2.INTER_AREA)
    return np.concatenate([left, right], axis=1)


def _debug_tools_enabled(yaml_cfg: dict, key: str) -> bool:
        tools = (yaml_cfg or {}).get("debug_tools", {})
        if tools.get("enabled", True) is False:
                return False
        return tools.get(key, True) is True


def _write_readme(out_dir: Path) -> None:
        readme = """# near_ratio 비교 이미지 설명

이 폴더의 `*_near_ratio*.jpg` 파일은 GhostFilter Step 3.5(손 occlusion/hallucination 억제)가 손을 왜 살리거나(KEEP) 왜 자르는지(REMOVE)를 **손 중심으로 시각화**한 디버그 이미지입니다.

## 파일 종류

- `*_near_ratio.jpg`
    - RAW(pre-filter) 상태에서의 시각화 이미지입니다.
- `*_near_ratio_crop.jpg`
    - 위 이미지를 손목(wrist) 주변으로 크롭한 버전입니다.
- `*_near_ratio_compare.jpg` (옵션: `--write_compare`)
    - 좌: RAW(pre-filter) / 우: FILTERED(post-filter) 비교 이미지입니다.
- `*_near_ratio_compare_crop.jpg` (옵션: `--write_compare`)
    - 위 compare 이미지를 손목 주변으로 크롭한 버전입니다.

## 그림 요소(손 위주)

- 흰 점: 손목(wrist)
- 노란 원: `base_radius` (손목 주변 "근접" 반경)
    - 손가락 점이 이 원 안에 들어오면 near로 카운트됩니다.
- 빨간 원: `far_radius` (손목에서 너무 멀리 튄 outlier 반경)
    - 손가락 점이 이 원 밖이면 far(outlier)로 카운트됩니다.

- 손가락 점 색:
    - 초록: near (`d <= base_radius`)
    - 주황: 중간 (`base_radius < d < far_radius`)
    - 빨강: far outlier (`d >= far_radius`)

## 키포인트 라벨

각 손가락 점 옆에 `wrist`, `thumb_1..4`, `index_1..4`, `middle_1..4`, `ring_1..4`, `pinky_1..4` 라벨이 표시됩니다.

## outlier 정의(중요)

이 디버그에서 말하는 **outlier는 “far outlier”**를 의미합니다.

- 각 손가락 점에 대해 손목(wrist)까지의 거리 $d$를 계산합니다.
- **far outlier 조건**: $d \ge \\texttt{far_radius}$ 인 경우
    - 이때 해당 점은 `far_count`에 1개로 카운트됩니다.

### far_radius 계산(코드 기준)

- `forearm = ||wrist - elbow||` (단, elbow score가 충분할 때만)
- `base_radius = max(hand_wrist_min_radius_px, forearm * hand_wrist_radius_ratio)`
- `far_radius = max(base_radius * 1.5, forearm * hand_far_outlier_ratio)`

## Step 3.5 제거 조건(요약)

- `near_ratio < hand_min_near_ratio` 또는
- `far_count > hand_max_far_points`

즉, 손가락 점이 손목 주변에 충분히 모여있지 않거나(near_ratio 낮음), 손목에서 너무 멀리 튄 점이 많으면(far_count 많음) 환각/가림으로 판단하고 손을 자릅니다.

## 상단 검정 오버레이 텍스트(각 줄 의미)

near_ratio 이미지 상단의 검정 박스에는 Step 3.5 판단에 필요한 핵심 수치가 줄 단위로 표시됩니다.

- `LHand/RHand verdict=KEEP|REMOVE`
    - 현재 프레임/손에 대해 Step 3.5가 "살릴지"(KEEP) "자를지"(REMOVE) 를 나타냅니다.
    - 판정 조건(이 스크립트 기준):
        - `wrist_score < 0.3` 이면 Step 3.5는 SKIP 됩니다(손목이 anchor 역할이라).
        - 먼저 `active >= hand_min_finger_points` 이어야 평가합니다.
        - 그 다음 `near_ratio < hand_min_near_ratio` 이거나 `far_count > hand_max_far_points`이면 REMOVE, 아니면 KEEP.

- `near_ratio=... (near=.../active=...) min=...`
    - `active`: 손 21점 중에서 `score >= thr` 를 만족하는 "활성" 점 개수
    - `near`: active 점 중에서 손목과의 거리 $d$가 `base_radius` 이하인 점 개수
    - `near_ratio = near / active`
    - `min=...` 는 설정값 `hand_min_near_ratio` 입니다.

- `far_count=... max_far=...`
    - `far_count`: active 점 중에서 $d \ge \\texttt{far_radius}$ 인 "far outlier" 개수
    - `max_far=...` 는 설정값 `hand_max_far_points` 입니다.

- `wrist_score=... elbow_score=...`
    - 여기서 wrist/elbow는 "손 21점"이 아니라 몸통 키포인트(왼/오른 손목, 팔꿈치)의 신뢰도입니다.
    - 팔꿈치 점수가 낮으면 아래 forearm/base_radius/far_radius 계산이 보수적으로(작게/기본값) 잡힐 수 있습니다.

- `forearm=... base_r=... far_r=...`
    - `forearm`: 팔꿈치→손목 거리(단, elbow score가 충분할 때만 의미 있게 계산)
    - `base_r`: 손목 주변 "근접" 반경 (near 판단에 사용)
    - `far_r`: 손목에서 "너무 멀다"를 판단하는 반경 (far outlier 판단에 사용)

- `thr=...`
    - 손 21점의 "활성(active)" 여부와 점 표시 여부를 결정하는 점수 임계값입니다.
    - `thr` 미만인 점은 active에 포함되지 않고, 그림에서도 그리지 않습니다.
"""
        (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GhostFilter Step 3.5 near_ratio around hands")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=r"d:\2025\pose_extractor\test_io\outputs\20260104_223045",
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
        default="_debug_near_ratio",
        help="Subfolder under run_dir to write debug artifacts",
    )
    parser.add_argument("--crop_half", type=int, default=600, help="Half-size of crop around wrist")
    parser.add_argument(
        "--write_compare",
        action="store_true",
        help="Write side-by-side raw vs filtered comparison images",
    )
    parser.add_argument(
        "--text_scale_mult",
        type=float,
        default=5.0,
        help="Multiply overlay text size (default 5x)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = Path(args.config)

    yaml_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not _debug_tools_enabled(yaml_cfg, "hand_near_ratio_visualization"):
        print("[skip] debug_tools disabled: hand_near_ratio_visualization")
        return

    pipe_cfg = PipelineConfig.from_yaml(str(config_path))
    pipeline = PoseTransferPipeline(pipe_cfg, yaml_config=yaml_cfg)
    gf = pipeline.ghost_filter

    thr = float(gf.config.hand_finger_min_confidence)
    min_pts = int(gf.config.hand_min_finger_points)
    radius_ratio = float(gf.config.hand_wrist_radius_ratio)
    min_radius = float(gf.config.hand_wrist_min_radius_px)
    min_near_ratio = float(gf.config.hand_min_near_ratio)
    far_ratio = float(gf.config.hand_far_outlier_ratio)
    max_far = int(gf.config.hand_max_far_points)

    out_dir = run_dir / str(args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_readme(out_dir)

    stems = sorted({p.stem.replace("_bg", "") for p in run_dir.glob("*_bg.jpg")})
    if not stems:
        raise SystemExit(f"No *_bg.jpg found in {run_dir}")

    print("Config:", {
        "thr": thr,
        "min_pts": min_pts,
        "min_near_ratio": min_near_ratio,
        "far_ratio": far_ratio,
        "max_far": max_far,
        "radius_ratio": radius_ratio,
        "min_radius": min_radius,
    })

    sides = [
        ("LHand", 7, 9, 91, 111),
        ("RHand", 8, 10, 112, 132),
    ]

    for stem in stems:
        bg_p = run_dir / f"{stem}_bg.jpg"
        img_bgr = cv2.imread(str(bg_p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("Skip unreadable:", bg_p)
            continue

        # run extractor on this frame
        img_rgb = load_image(bg_p)
        kpts, raw_scores, person_idx, size = pipeline.extract_pose(img_rgb, filter_person=True)

        # Apply ghost filter once (final filtered view)
        filtered_scores = gf.filter_single(kpts, raw_scores, size).filtered_scores

        # We want to visualize what Step 3.5 sees BEFORE it removes anything
        # Configure overlay text size (large by default)
        draw_hand_debug._font_scale = 0.8 * float(args.text_scale_mult)
        draw_hand_debug._thickness = max(2, int(2 * float(args.text_scale_mult)))

        for name, elbow_idx, wrist_idx, start, end in sides:
            m_raw = compute_hand_metrics(
                kpts,
                raw_scores,
                name,
                elbow_idx,
                wrist_idx,
                start,
                end,
                thr,
                min_pts,
                radius_ratio,
                min_radius,
                far_ratio,
            )

            m_filtered = compute_hand_metrics(
                kpts,
                filtered_scores,
                name,
                elbow_idx,
                wrist_idx,
                start,
                end,
                thr,
                min_pts,
                radius_ratio,
                min_radius,
                far_ratio,
            )

            # if wrist is too low-confidence, Step 3.5 would skip; still useful to visualize
            debug_raw = draw_hand_debug(img_bgr, kpts, raw_scores, m_raw, thr, min_pts, min_near_ratio, max_far)
            debug_filt = draw_hand_debug(img_bgr, kpts, filtered_scores, m_filtered, thr, min_pts, min_near_ratio, max_far)

            label_scale = 1.0 * float(args.text_scale_mult)
            label_thickness = max(2, int(2 * float(args.text_scale_mult)))

            cv2.putText(
                debug_raw,
                "RAW (pre-filter)",
                (30, debug_raw.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_filt,
                "FILTERED (post-filter)",
                (30, debug_filt.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA,
            )

            # crop around wrist to focus on hand
            cx, cy = int(round(kpts[wrist_idx][0])), int(round(kpts[wrist_idx][1]))
            crop_raw = crop_around_point(debug_raw, cx, cy, args.crop_half)
            crop_filt = crop_around_point(debug_filt, cx, cy, args.crop_half)

            out_p = out_dir / f"{stem}_{name}_near_ratio.jpg"
            out_crop_p = out_dir / f"{stem}_{name}_near_ratio_crop.jpg"

            cv2.imwrite(str(out_p), debug_raw)
            cv2.imwrite(str(out_crop_p), crop_raw)

            if args.write_compare:
                compare = hstack_same_height(debug_raw, debug_filt)
                compare_crop = hstack_same_height(crop_raw, crop_filt)
                out_cmp = out_dir / f"{stem}_{name}_near_ratio_compare.jpg"
                out_cmp_crop = out_dir / f"{stem}_{name}_near_ratio_compare_crop.jpg"
                cv2.imwrite(str(out_cmp), compare)
                cv2.imwrite(str(out_cmp_crop), compare_crop)

        print(f"Wrote: {stem} (person_idx={person_idx})")


if __name__ == "__main__":
    main()
