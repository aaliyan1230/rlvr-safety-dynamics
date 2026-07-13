#!/usr/bin/env python3
"""Validate and package one completed cross-format Tülu wave."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlvr_safety.io import read_jsonl, sha256_file  # noqa: E402


SNAPSHOTS = [
    "cross_format_generations.jsonl",
    "model_0_metadata.json",
    "model_1_metadata.json",
    "run_metadata.json",
    "run_summary.json",
]


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    args = parser.parse_args()

    wave = f"{args.wave:02d}"
    experiment_id = f"tulu_cross_format_wave_{wave}"
    source = args.source_dir.resolve()
    summary = json.loads((source / "run_summary.json").read_text())
    metadata = json.loads((source / "run_metadata.json").read_text())
    config_path = REPO / f"configs/experiments/{experiment_id}.json"
    config = json.loads(config_path.read_text())
    rows = list(read_jsonl(source / "cross_format_generations.jsonl"))
    expected = int(config["gates"]["expected_rows"])
    cells = {(row["model"], row["panel"], row["id"]) for row in rows}
    capped = [
        row
        for row in rows
        if int(row.get("generated_tokens", 0)) >= int(row["row_max_new_tokens"])
    ]
    cap_rate = len(capped) / expected
    cap_limit = float(config["gates"]["max_token_capped_rate"])
    quality_exception = cap_rate > cap_limit
    allowed_errors = [{"token_capped": len(capped), "rows": expected}] if quality_exception else []

    if summary.get("experiment_id") != experiment_id:
        raise SystemExit("mismatched experiment id")
    if len(rows) != expected or len(cells) != expected:
        raise SystemExit(f"incomplete wave: rows={len(rows)}, cells={len(cells)}, expected={expected}")
    if bool(summary.get("had_error")) != quality_exception or summary.get("errors", []) != allowed_errors:
        raise SystemExit(f"unexpected integrity errors: {summary.get('errors')}")
    models = summary.get("models", [])
    configured = {row["id"]: row for row in config["models"]}
    if len(models) != 2 or any(
        row.get("model_revision_requested") != configured.get(row.get("model"), {}).get("revision")
        or row.get("model_revision_resolved") != row.get("model_revision_requested")
        or row.get("attention_mask_mode") != "explicit_all_ones"
        or row.get("model_is_quantized") is not True
        for row in models
    ):
        raise SystemExit("revision, mask, or quantization integrity failed")
    if metadata.get("gpus") != ["Tesla T4", "Tesla T4"]:
        raise SystemExit(f"unexpected GPUs: {metadata.get('gpus')}")

    destination = REPO / f"artifacts/{experiment_id}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOTS:
        shutil.copy2(source / name, destination / name)
    shutil.copy2(config_path, destination / "config.json")
    shutil.copy2(REPO / "src/rlvr_safety/kaggle_cross_format.py", destination / "kaggle_cross_format.py")
    shutil.copy2(REPO / f"kaggle/{experiment_id}/run_wave.py", destination / "run_wave.py")
    shutil.copy2(
        REPO / f"kaggle/{experiment_id}/kernel-metadata.json",
        destination / "kernel-metadata.json",
    )

    started = parse_utc(metadata["started_utc"])
    completed = parse_utc(summary["completed_utc"])
    wall_minutes = round((completed - started).total_seconds() / 60, 2)
    attempts = {
        "schema_version": 1,
        "remote_kernel": f"aaliyanshaikh/rlvr-tulu-cross-format-wave-{wave}",
        "attempts": [
            {
                "kernel_version": args.kernel_version,
                "completed_utc": summary["completed_utc"],
                "disposition": (
                    "retained_complete_primary_with_quality_exception"
                    if quality_exception
                    else "accepted_integrity_artifact"
                ),
                "rows": expected,
                "token_capped": len(capped),
                "quality_gate_passed": not quality_exception,
                "corrective_rerun": False,
            }
        ],
    }
    (destination / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    files = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "role": "Frozen cross-format primary artifact; capped rows remain censored or incorrect.",
        "remote_kernel": attempts["remote_kernel"],
        "accepted_kernel_version": args.kernel_version,
        "completion": {
            "rows": expected,
            "expected_rows": expected,
            "unique_cells": expected,
            "capability_rows": sum(row["panel"] == "capability" for row in rows),
            "freeform_rows": sum(row["panel"] == "freeform" for row in rows),
            "token_capped": len(capped),
            "token_capped_rate": cap_rate,
            "quality_gate_passed": not quality_exception,
            "quality_exception": (
                "token-cap rate exceeded the one-percent wave gate; rows retained under frozen censoring rules"
                if quality_exception
                else None
            ),
            "wall_clock_minutes": wall_minutes,
            "peak_gpu_memory_bytes_per_model": max(row["peak_gpu_memory_bytes"] for row in models),
            "attention_mask_mode": "explicit_all_ones",
            "resolved_revisions_match_requested": True,
            "scientific_interpretation_allowed": False,
        },
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    gate = "QUALITY EXCEPTION" if quality_exception else "PASS"
    report = (
        f"# Tülu cross-format wave {wave} integrity report\n\n"
        f"**Gate result: {gate}. Scientific interpretation: withheld until all waves and blinded scoring are complete.**\n\n"
        f"The pinned T4×2 run produced {expected}/{expected} unique cells in {wall_minutes:.2f} minutes, "
        f"with {len(capped)} token-capped row(s). "
        + (
            "The capped capability row is retained and scored incorrect under the frozen protocol; no replacement generation was run. "
            if quality_exception
            else "The frozen one-percent token-cap gate passed. "
        )
        + "Requested and resolved revisions matched, quantization was active, and explicit attention masks were used.\n"
    )
    (REPO / f"reports/{experiment_id}_integrity.md").write_text(report)
    print(f"Packaged wave {wave}: {expected}/{expected} rows, gate={gate}, capped={len(capped)}")


if __name__ == "__main__":
    main()
