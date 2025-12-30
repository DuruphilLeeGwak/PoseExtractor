"""Legacy entrypoint.

This repository historically had two batch scripts:
- test_transfer.py (root)
- test/test_transfer.py

The root script previously bypassed PoseTransferPipeline.transfer(), which caused
different outputs from PoseExtractor (ghost filter / intersection / align / canvas
steps were skipped).

To keep behavior consistent, this file now forwards execution to test/test_transfer.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    target = project_root / "test" / "test_transfer.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing forwarded script: {target}")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()