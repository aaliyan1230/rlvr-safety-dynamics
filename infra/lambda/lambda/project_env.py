"""Load only Lambda settings from the ignored project environment file."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

PROJECT_ENV = Path(__file__).resolve().parents[3] / ".env"
VARIABLES = {"LAMBDA_API_KEY", "LAMBDA_API_KEY_FILE", "LAMBDA_FORWARD_API_KEY"}


def load_project_env(path: Path = PROJECT_ENV) -> None:
    if not path.exists():
        return
    if path.stat().st_mode & 0o077:
        raise ValueError(f"Restrict {path} to owner access with chmod 600 before using Lambda")
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in VARIABLES or name in os.environ:
            continue
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError(f"Invalid Lambda setting on line {number} of {path}") from None
        if len(parts) > 1:
            raise ValueError(f"Quote the Lambda setting on line {number} of {path}")
        os.environ[name] = parts[0] if parts else ""
