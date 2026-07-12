#!/usr/bin/env python3
"""Compatibility wrapper for the packaged structured-choice scorer."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlvr_safety.cli.score_choices import main  # noqa: E402


if __name__ == "__main__":
    main()
