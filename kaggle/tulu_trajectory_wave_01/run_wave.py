#!/usr/bin/env python3
"""Kaggle entry point for Tülu structured trajectory wave 01."""

from __future__ import annotations

import sys
from pathlib import Path


INPUT = Path("/kaggle/input")
matches = sorted(INPUT.glob("**/src/rlvr_safety/__init__.py"))
if len(matches) != 1:
    raise RuntimeError(f"expected one bundled rlvr_safety package, found {matches}")
sys.path.insert(0, str(matches[0].parents[1]))

from rlvr_safety.kaggle_trajectory import main  # noqa: E402


if __name__ == "__main__":
    main(Path(__file__).resolve())
