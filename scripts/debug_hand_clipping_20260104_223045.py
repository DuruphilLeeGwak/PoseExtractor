from __future__ import annotations

import argparse
from pathlib import Path
import json
from collections import Counter
import sys

import numpy as np
import yaml

# Ensure local imports work regardless of cwd / launcher wrappers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_transfer.pipeline import PipelineConfig, PoseTransferPipeline
from pose_transfer.utils.io import load_image


def _debug_tools_enabled(yaml_cfg: dict, key: str) -> bool:
    tools = (yaml_cfg or {}).get("debug_tools", {})
    if tools.get("enabled", True) is False:
        return False
    return tools.get(key, True) is True


def _write_readme(out_dir: Path) -> None:
    readme = """# Hand clipping 디버그 출력

이 폴더는 GhostFilter Step 3.5(손 occlusion/hallucination 억제) 때문에 손이 잘리는지 여부를 프레임별로 확인하기 위한 리포트입니다.

## 생성 파일

- `report.json`
  - 프레임별로 다음 정보를 담습니다.
    - 저장된 OpenPose JSON에서 hand_left/right valid 개수
    - Step 3.5 근사 지표(active/near/far/base_radius/far_radius/verdict)
    - GhostFilter removal_reasons 집계

## 참고

이 스크립트는 콘솔에도 요약을 출력합니다.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def summarize_openpose_hand(person: dict, key: str) -> str:
    arr = np.array(person.get(key, []), dtype=float)
    if arr.size == 0:
        return f"{key}: (missing)"
    scores = arr[2::3]
    valid = int(np.sum(scores > 0))
    return f"{key}: valid {valid}/21, min={scores.min():.3f}, max={scores.max():.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug why hands are removed by GhostFilter (Step 3.5)")
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
        default="_debug_hand_clipping",
        help="Subfolder under run_dir to write report.json and README.md",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    out_dir = run_dir / args.out_subdir

    yaml_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not _debug_tools_enabled(yaml_cfg, "hand_clipping_debug"):
        print("[skip] debug_tools disabled: hand_clipping_debug")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_readme(out_dir)

    pipe_cfg = PipelineConfig.from_yaml(str(config_path))
    pipeline = PoseTransferPipeline(pipe_cfg, yaml_config=yaml_cfg)
    gf = pipeline.ghost_filter

    print("CONFIG hand occlusion:")
    print(
        {
            "check_hand_occlusion": gf.config.check_hand_occlusion,
            "thr": gf.config.hand_finger_min_confidence,
            "min_pts": gf.config.hand_min_finger_points,
            "min_near_ratio": gf.config.hand_min_near_ratio,
            "far_ratio": gf.config.hand_far_outlier_ratio,
            "max_far": gf.config.hand_max_far_points,
            "radius_ratio": gf.config.hand_wrist_radius_ratio,
            "min_radius": gf.config.hand_wrist_min_radius_px,
            "check_clustering": gf.config.check_clustering,
            "min_cluster_spread": gf.config.min_cluster_spread,
            "check_boundary_values": gf.config.check_boundary_values,
            "boundary_tolerance": gf.config.boundary_tolerance,
            "check_bounds": gf.config.check_bounds,
        }
    )

    stems = sorted({p.stem.replace("_bg", "") for p in run_dir.glob("*_bg.jpg")})
    if not stems:
        raise SystemExit(f"No *_bg.jpg found in {run_dir}")

    report: list[dict] = []

    for stem in stems:
        bg_p = run_dir / f"{stem}_bg.jpg"
        kp_p = run_dir / f"{stem}_kp.json"
        print("\n" + "=" * 100)
        print("FRAME", stem)
        if kp_p.exists():
            data = json.loads(kp_p.read_text(encoding="utf-8"))
            person = data.get("people", [{}])[0]
            print("Saved JSON:")
            print(" ", summarize_openpose_hand(person, "hand_left_keypoints_2d"))
            print(" ", summarize_openpose_hand(person, "hand_right_keypoints_2d"))
        else:
            print("Saved JSON: (missing)")

        img = load_image(bg_p)
        kpts, raw_scores, person_idx, size = pipeline.extract_pose(img, filter_person=True)
        res = gf.filter_single(kpts, raw_scores, size)

        # Mirror Step 3.5 math for explanation
        thr = float(gf.config.hand_finger_min_confidence)
        min_pts = int(gf.config.hand_min_finger_points)
        radius_ratio = float(gf.config.hand_wrist_radius_ratio)
        min_radius = float(gf.config.hand_wrist_min_radius_px)
        min_near_ratio = float(gf.config.hand_min_near_ratio)
        far_ratio = float(gf.config.hand_far_outlier_ratio)
        max_far = int(gf.config.hand_max_far_points)

        def occlusion_metrics(name: str, elbow_idx: int, wrist_idx: int, start: int, end: int) -> dict:
            if wrist_idx >= len(kpts):
                return {"name": name, "skip": True}
            wrist = kpts[wrist_idx]
            elbow_score = float(raw_scores[elbow_idx]) if elbow_idx < len(raw_scores) else 0.0
            wrist_score = float(raw_scores[wrist_idx])
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
                c = float(raw_scores[i])
                if c < thr:
                    continue
                active += 1
                d = float(np.linalg.norm(kpts[i] - wrist))
                if d <= base_radius:
                    near += 1
                if d >= far_radius:
                    far += 1
            if active < min_pts:
                verdict = "SKIP(active<min_pts)"
                nr = None
            else:
                nr = near / max(1, active)
                verdict = "REMOVE" if (nr < min_near_ratio or far > max_far) else "KEEP"

            print(
                f"  [Step3.5] {name}: wrist_score={wrist_score:.3f} elbow_score={elbow_score:.3f} "
                f"forearm={forearm:.1f} base_r={base_radius:.1f} far_r={far_radius:.1f} "
                f"active={active}/21 near={near} far={far} verdict={verdict}"
            )

            return {
                "name": name,
                "wrist_score": wrist_score,
                "elbow_score": elbow_score,
                "forearm": forearm,
                "base_radius": base_radius,
                "far_radius": far_radius,
                "active": active,
                "near": near,
                "far": far,
                "near_ratio": nr,
                "verdict": verdict,
            }

        m_l = occlusion_metrics("LHand", 7, 9, 91, 111)
        m_r = occlusion_metrics("RHand", 8, 10, 112, 132)

        def reasons_for(start: int, end: int) -> Counter:
            c = Counter()
            for i in range(start, end + 1):
                r = res.removal_reasons.get(i)
                if r:
                    c[r] += 1
            return c

        lw, rw = 9, 10
        print(f"RAW wrists: Lw={raw_scores[lw]:.3f} Rw={raw_scores[rw]:.3f} person_idx={person_idx}")

        lh_reason = reasons_for(91, 111)
        rh_reason = reasons_for(112, 132)
        print("Removed LH reasons:")
        print(" ", dict(lh_reason) if lh_reason else "(none)")
        print("Removed RH reasons:")
        print(" ", dict(rh_reason) if rh_reason else "(none)")

        if lw in res.removal_reasons:
            print("Lw removed:", res.removal_reasons[lw])
        if rw in res.removal_reasons:
            print("Rw removed:", res.removal_reasons[rw])

        lh_post = int(np.sum(res.filtered_scores[91:112] > 0))
        rh_post = int(np.sum(res.filtered_scores[112:133] > 0))
        print(f"Post-filter: LH={lh_post}/21 RH={rh_post}/21")

        entry: dict = {
            "stem": stem,
            "person_idx": int(person_idx),
            "saved_json": None,
            "step3_5": {"LHand": m_l, "RHand": m_r},
            "removal_reasons": {
                "LHand": dict(lh_reason) if lh_reason else {},
                "RHand": dict(rh_reason) if rh_reason else {},
            },
            "post_filter_valid": {"LHand": lh_post, "RHand": rh_post},
        }

        if kp_p.exists():
            data = json.loads(kp_p.read_text(encoding="utf-8"))
            person = data.get("people", [{}])[0]
            entry["saved_json"] = {
                "hand_left": summarize_openpose_hand(person, "hand_left_keypoints_2d"),
                "hand_right": summarize_openpose_hand(person, "hand_right_keypoints_2d"),
            }

        report.append(entry)

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
