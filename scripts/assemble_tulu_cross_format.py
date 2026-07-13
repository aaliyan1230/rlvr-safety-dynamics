#!/usr/bin/env python3
"""Assemble six frozen Tülu cross-format waves without inspecting outcomes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlvr_safety.io import read_jsonl, sha256_file  # noqa: E402


STEPS = [0, 40, 80, 160, 320, 640, 960, 1280, 1600, 1920, 2240, 2440]
MILESTONES = {0, 320, 960, 1600, 1920, 2440}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=REPO / "artifacts")
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/tulu_cross_format_v1")
    args = parser.parse_args()

    rows: list[dict] = []
    source_manifests = []
    quality_exceptions = []
    for wave in range(1, 7):
        source = args.artifacts_root / f"tulu_cross_format_wave_{wave:02d}"
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source_manifests.append(
            {
                "path": str(manifest_path.relative_to(REPO)),
                "sha256": sha256_file(manifest_path),
            }
        )
        if not manifest["completion"]["quality_gate_passed"]:
            quality_exceptions.append(
                {
                    "wave": wave,
                    "token_capped": manifest["completion"]["token_capped"],
                    "disposition": manifest["completion"]["quality_exception"],
                }
            )
        rows.extend(read_jsonl(source / "cross_format_generations.jsonl"))

    keys = {(int(row["checkpoint"]), row["panel"], row["id"]) for row in rows}
    if len(rows) != 504 or len(keys) != 504:
        raise SystemExit(f"expected 504 unique cells, got rows={len(rows)}, keys={len(keys)}")
    capability = sorted(
        (row for row in rows if row["panel"] == "capability"),
        key=lambda row: (int(row["checkpoint"]), str(row["id"])),
    )
    freeform = sorted(
        (row for row in rows if row["panel"] == "freeform"),
        key=lambda row: (int(row["checkpoint"]), str(row["id"])),
    )
    if len(capability) != 360 or Counter(int(row["checkpoint"]) for row in capability) != Counter(
        {step: 30 for step in STEPS}
    ):
        raise SystemExit("capability schedule is incomplete")
    if len(freeform) != 144 or Counter(int(row["checkpoint"]) for row in freeform) != Counter(
        {step: 24 for step in MILESTONES}
    ):
        raise SystemExit("free-form milestone schedule is incomplete")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "cross_format_generations.jsonl", sorted(rows, key=lambda row: (int(row["checkpoint"]), row["panel"], row["id"])))
    write_jsonl(args.out_dir / "capability_generations.jsonl", capability)
    write_jsonl(args.out_dir / "freeform_generations.jsonl", freeform)
    shutil.copy2(
        REPO / "src/rlvr_safety/cross_format_analysis.py",
        args.out_dir / "cross_format_analysis.py",
    )
    shutil.copy2(
        REPO / "src/rlvr_safety/cli/score_capability.py",
        args.out_dir / "score_capability.py",
    )
    shutil.copy2(
        REPO / "scripts/score_tulu_freeform_ai.py",
        args.out_dir / "score_tulu_freeform_ai.py",
    )
    capped = [
        row
        for row in rows
        if int(row.get("generated_tokens", 0)) >= int(row["row_max_new_tokens"])
    ]
    files = {
        path.name: sha256_file(path)
        for path in sorted(args.out_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "tulu_cross_format_v1",
        "role": "Complete frozen capability and blinded free-form generation panel.",
        "completion": {
            "rows": 504,
            "unique_cells": 504,
            "capability_rows": 360,
            "freeform_rows": 144,
            "token_capped": len(capped),
            "quality_exceptions": quality_exceptions,
            "scientific_interpretation_allowed": True,
        },
        "source_manifests": source_manifests,
        "files": files,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Assembled 504 cells: capability=360, freeform=144, capped={len(capped)}, "
        f"quality_exceptions={len(quality_exceptions)}"
    )


if __name__ == "__main__":
    main()
