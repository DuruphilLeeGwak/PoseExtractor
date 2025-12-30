import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(r"d:\\2025\\pose_extractor")
    os.chdir(root)
    sys.path.insert(0, str(root))

    from pose_transfer.pipeline import PoseTransferPipeline
    from pose_transfer.config import load_config
    from pose_transfer.utils.io import load_image

    cfg, yaml_cfg = load_config(str(root / "pose_transfer" / "config" / "default.yaml"))
    pipe = PoseTransferPipeline(config=cfg, yaml_config=yaml_cfg)

    inputs = root / "test_io" / "inputs"
    files = [p for p in inputs.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    files.sort(key=lambda p: p.name)

    HIP_L, HIP_R = 11, 12
    thr = 0.1

    print(f"Scanning {len(files)} images for hip(11/12) removed by GhostFilter...")

    found = 0
    for p in files:
        img = load_image(str(p))
        kpts, scores, _, image_size = pipe.extract_pose(img, filter_person=True)

        # only care about cases where hip was initially detected
        if scores[HIP_L] <= thr and scores[HIP_R] <= thr:
            continue

        filt_scores, filt_res = pipe._apply_ghost_filter_single(kpts, scores, image_size)

        removed = []
        for idx, name in [(HIP_L, "Lhip"), (HIP_R, "Rhip")]:
            if scores[idx] > thr and filt_scores[idx] <= 1e-9:
                removed.append((idx, name, filt_res.removal_reasons.get(idx)))

        if removed:
            found += 1
            print(f"\n- {p.name}")
            for idx, nm, reason in removed:
                x, y = kpts[idx]
                print(
                    f"  {nm} idx={idx} orig={scores[idx]:.3f} -> 0 at ({x:.1f},{y:.1f}) reason={reason}"
                )

    print(f"\nDone. Found {found} images with hip removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
