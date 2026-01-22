from __future__ import annotations

import argparse
from datetime import datetime
import re
import subprocess
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _script_accepts_arg(script_text: str, flag: str) -> bool:
    # Simple heuristic: the argparse flag literal appears in the source.
    return flag in script_text


def _script_requires_run_dir(script_text: str) -> bool:
    # Heuristic: if --run_dir is present and it's marked required=True near it.
    # (Keep it lenient; we only use this to avoid a guaranteed failure.)
    return re.search(r"--run_dir[\s\S]{0,250}required\s*=\s*True", script_text) is not None


def _discover_scripts(script_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    candidates = list(script_dir.rglob(pattern) if recursive else script_dir.glob(pattern))
    out: list[Path] = []

    this_file = Path(__file__).resolve()
    for p in candidates:
        if not p.is_file() or p.suffix.lower() != ".py":
            continue
        if p.resolve() == this_file:
            continue
        out.append(p)

    # Deterministic order
    out.sort(key=lambda x: (x.name.lower(), str(x).lower()))
    return out


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_latest_run_dir(project_root: Path) -> Path | None:
    base = project_root / "test_io" / "outputs"
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    # Folder names are like YYYYMMDD_HHMMSS, lexicographic sort works.
    return sorted(dirs, key=lambda p: p.name)[-1]


def _default_config_path(project_root: Path) -> Path | None:
    p = project_root / "pose_transfer" / "config" / "default.yaml"
    return p if p.exists() else None


def _base_out_subdir_for(script_path: Path) -> str:
    name = script_path.name
    if name.startswith("debug_hand_refiner"):
        return "_debug_hand_refiner"
    if name.startswith("debug_hand_clipping"):
        return "_debug_hand_clipping"
    if name.startswith("visualize_hand_near_ratio"):
        return "_debug_near_ratio"
    return f"_{script_path.stem}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all hand-related scripts in scripts/ at once (sequentially). "
            "By default it runs files matching '*hand*.py'."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="If provided, pass --run_dir to scripts that support it. If omitted, uses latest test_io/outputs/*.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="If provided, pass --config to scripts that support it. If omitted, uses pose_transfer/config/default.yaml when present.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Append suffix to each script's output subdir (when the script supports --out_subdir).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*hand*.py",
        help="Glob pattern to select scripts (default: *hand*.py).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders under scripts/ as well.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining scripts even if one fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional per-script timeout (seconds).",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra args appended to EACH script. Prefix with -- to disambiguate.",
    )

    args = parser.parse_args()

    suffix = args.suffix
    if suffix is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = Path(__file__).resolve().parent
    scripts = _discover_scripts(script_dir, pattern=args.pattern, recursive=bool(args.recursive))

    if not scripts:
        print(f"No scripts matched pattern={args.pattern!r} under {script_dir}")
        return 2

    # Normalize pass-through args: argparse.REMAINDER will include leading '--' if provided.
    passthrough = list(args.script_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    project_root = _default_project_root()

    resolved_run_dir: Path | None
    if args.run_dir:
        resolved_run_dir = Path(args.run_dir).resolve()
    else:
        resolved_run_dir = _find_latest_run_dir(project_root)

    resolved_config: Path | None
    if args.config:
        resolved_config = Path(args.config).resolve()
    else:
        resolved_config = _default_config_path(project_root)

    run_dir = str(resolved_run_dir) if resolved_run_dir else None
    config = str(resolved_config) if resolved_config else None

    failures: list[tuple[Path, int]] = []
    skipped: list[Path] = []

    print(f"Discovered {len(scripts)} script(s):")
    for p in scripts:
        print(f" - {p.relative_to(script_dir)}")

    print(f"Suffix: {suffix}")

    for script_path in scripts:
        text = _read_text(script_path)

        # If no run_dir was provided but the script likely requires it, skip (or fail-fast).
        if run_dir is None and _script_requires_run_dir(text):
            msg = f"[skip] {script_path.name}: requires --run_dir"
            print(msg)
            skipped.append(script_path)
            if args.dry_run or args.continue_on_error:
                continue
            print("Tip: pass --run_dir <folder> or use --continue_on_error")
            return 2

        cmd: list[str] = [args.python, str(script_path)]

        if run_dir is not None and _script_accepts_arg(text, "--run_dir"):
            cmd += ["--run_dir", run_dir]
        if config is not None and _script_accepts_arg(text, "--config"):
            cmd += ["--config", config]

        if suffix and _script_accepts_arg(text, "--out_subdir"):
            base = _base_out_subdir_for(script_path)
            cmd += ["--out_subdir", f"{base}_{suffix}"]

        cmd += passthrough

        print("\n" + "=" * 100)
        print("RUN:")
        print(" ", subprocess.list2cmdline(cmd))

        if args.dry_run:
            continue

        try:
            completed = subprocess.run(cmd, check=False, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[timeout] {script_path.name} exceeded {args.timeout}s")
            failures.append((script_path, 124))
            if not args.continue_on_error:
                return 124
            continue

        if completed.returncode != 0:
            print(f"[fail] {script_path.name} exit_code={completed.returncode}")
            failures.append((script_path, int(completed.returncode)))
            if not args.continue_on_error:
                return int(completed.returncode)
        else:
            print(f"[ok] {script_path.name}")

    print("\n" + "=" * 100)
    if skipped:
        print(f"Skipped {len(skipped)} script(s) (missing required args).")
    if failures:
        print(f"Failed {len(failures)} script(s):")
        for p, code in failures:
            print(f" - {p.name}: {code}")
        return 1

    if args.dry_run:
        print("Dry-run completed.")
        return 0

    print("All scripts finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
