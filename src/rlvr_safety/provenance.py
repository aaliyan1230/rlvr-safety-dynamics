"""Checksum verification for tracked evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path

from .io import sha256_file


class ProvenanceError(ValueError):
    pass


def verify_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ProvenanceError(f"{manifest_path}: non-empty files mapping is required")
    for relative, expected in files.items():
        path = manifest_path.parent / relative
        if not path.is_file():
            raise ProvenanceError(f"missing artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ProvenanceError(
                f"checksum mismatch for {path}: expected {expected}, observed {actual}"
            )
    return manifest
