#!/usr/bin/env python3
"""Thin entrypoint for the exact-image historical stage reproduction."""

# ruff: noqa: E402, I001

import sys
from pathlib import Path


INPUT = Path("/kaggle/input")
matches = sorted(INPUT.glob("**/src/rlvr_safety/__init__.py"))
if len(matches) != 1:
    raise RuntimeError(f"expected one bundled rlvr_safety package, found {matches}")
PACKAGE_SOURCE = matches[0].parents[1]
sys.path.insert(0, str(PACKAGE_SOURCE))

from rlvr_safety.kaggle_reproduction import main  # noqa: E402


if __name__ == "__main__":
    main(
        profile="stage",
        script_path=Path(__file__).resolve(),
        package_source=PACKAGE_SOURCE,
    )
