"""Parse and verify the complete frozen Python environment before inference."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from ..io import sha256_file


def package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_versions(text: str) -> dict[str, str]:
    """Exact, normalized name==version pins; reject ambiguous or editable entries."""
    versions: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+_-]*)", line)
        if match is None:
            raise ValueError(f"invalid runtime lock line: {line}")
        name, version = package_name(match[1]), match[2]
        if name in versions:
            raise ValueError(f"duplicate runtime lock entry: {name}")
        versions[name] = version
    if not versions:
        raise ValueError("runtime lock has no pinned packages")
    return versions


def compare_runtime(locked_text: str, installed_text: str) -> dict:
    locked, installed = lock_versions(locked_text), lock_versions(installed_text)
    missing = sorted(locked.keys() - installed.keys())
    unexpected = sorted(installed.keys() - locked.keys())
    mismatched = {
        name: {"expected": locked[name], "observed": installed[name]}
        for name in sorted(locked.keys() & installed.keys())
        if locked[name] != installed[name]
    }
    return {
        "passed": not (missing or unexpected or mismatched),
        "locked_packages": len(locked),
        "installed_packages": len(installed),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def verify_runtime_lock(lock_path: Path, report_path: Path) -> dict:
    installed = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, timeout=60
    )
    report = {
        "lock_sha256": sha256_file(lock_path),
        **compare_runtime(lock_path.read_text(), installed),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(f"runtime differs from frozen lock; inspect {report_path}")
    return report
