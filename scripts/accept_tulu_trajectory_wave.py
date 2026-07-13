#!/usr/bin/env python3
"""Validate and package one completed structured Tülu wave without reading outcomes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlvr_safety.io import sha256_file  # noqa: E402


SNAPSHOTS = [
    "choice_scores.csv",
    "model_0_metadata.json",
    "model_1_metadata.json",
    "run_metadata.json",
    "run_summary.json",
]


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--kind", choices=["primary", "refinement"], default="primary")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    parser.add_argument("--prior-failure-version", type=int)
    parser.add_argument("--prior-failure-reason")
    args = parser.parse_args()
    wave = f"{args.wave:02d}"
    experiment_stem = (
        "tulu_grpo_trajectory_wave_"
        if args.kind == "primary"
        else "tulu_grpo_trajectory_refinement_wave_"
    )
    artifact_stem = (
        "tulu_trajectory_wave_" if args.kind == "primary" else "tulu_trajectory_refinement_wave_"
    )
    kernel_stem = (
        "rlvr-tulu-grpo-trajectory-wave-"
        if args.kind == "primary"
        else "rlvr-tulu-trajectory-refinement-wave-"
    )
    source = args.source_dir.resolve()
    summary = json.loads((source / "run_summary.json").read_text())
    metadata = json.loads((source / "run_metadata.json").read_text())
    config_path = REPO / f"configs/experiments/{experiment_stem}{wave}.json"
    config_data = json.loads(config_path.read_text())
    expected_id = f"{experiment_stem}{wave}"
    if summary.get("experiment_id") != expected_id or summary.get("had_error") is not False:
        raise SystemExit(f"failed or mismatched run summary: {summary}")
    if summary.get("rows") != 1152 or summary.get("unique_cells") != 1152:
        raise SystemExit("wave is not 1,152/1,152 complete")
    quality = summary.get("quality_by_model", {})
    maximum_rate = float(config_data["gates"]["max_malformed_or_capped_rate"])
    if len(quality) != 2 or any(
        row.get("rows") != 576
        or (int(row.get("malformed", 0)) + int(row.get("token_capped", 0))) / 576
        > maximum_rate
        for row in quality.values()
    ):
        raise SystemExit(f"wave quality gate failed: {quality}")
    malformed = sum(int(row["malformed"]) for row in quality.values())
    token_capped = sum(int(row["token_capped"]) for row in quality.values())
    models = summary.get("models", [])
    if len(models) != 2 or any(
        row.get("model_revision_requested") != row.get("model_revision_resolved")
        or row.get("attention_mask_mode") != "explicit_all_ones"
        or row.get("model_is_quantized") is not True
        for row in models
    ):
        raise SystemExit("revision, mask, or quantization integrity failed")
    if metadata.get("gpus") != ["Tesla T4", "Tesla T4"]:
        raise SystemExit(f"unexpected GPUs: {metadata.get('gpus')}")
    destination = REPO / f"artifacts/{artifact_stem}{wave}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOTS:
        shutil.copy2(source / name, destination / name)
    runner = REPO / f"kaggle/{artifact_stem}{wave}/run_wave.py"
    kernel_metadata = REPO / f"kaggle/{artifact_stem}{wave}/kernel-metadata.json"
    shutil.copy2(config_path, destination / "config.json")
    shutil.copy2(REPO / "src/rlvr_safety/kaggle_trajectory.py", destination / "kaggle_trajectory.py")
    shutil.copy2(runner, destination / "run_wave.py")
    shutil.copy2(kernel_metadata, destination / "kernel-metadata.json")
    attempt_rows = []
    if args.prior_failure_version is not None:
        if not args.prior_failure_reason:
            raise SystemExit("--prior-failure-reason is required with --prior-failure-version")
        attempt_rows.append(
            {
                "kernel_version": args.prior_failure_version,
                "disposition": "failed_before_generation",
                "rows": 0,
                "reason": args.prior_failure_reason,
                "outcomes_inspected": False,
            }
        )
    attempt_rows.append({
        "kernel_version": args.kernel_version,
        "completed_utc": summary["completed_utc"],
        "disposition": "accepted_integrity_artifact",
        "rows": 1152,
        "malformed": malformed,
        "token_capped": token_capped,
        "outcomes_inspected": False,
    })
    attempts = {
        "schema_version": 1,
        "remote_kernel": f"aaliyanshaikh/{kernel_stem}{wave}",
        "attempts": attempt_rows,
    }
    (destination / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    started = parse_utc(metadata["started_utc"])
    completed = parse_utc(summary["completed_utc"])
    wall_minutes = round((completed - started).total_seconds() / 60, 2)
    files = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": expected_id,
        "role": "Accepted integrity-only artifact; outcomes withheld until the complete trajectory panel is available.",
        "remote_kernel": attempts["remote_kernel"],
        "accepted_kernel_version": args.kernel_version,
        "completion": {
            "rows": 1152,
            "expected_rows": 1152,
            "unique_cells": 1152,
            "rows_per_checkpoint": 576,
            "strict_parse_rows": 1152 - malformed - token_capped,
            "malformed": malformed,
            "token_capped": token_capped,
            "wall_clock_minutes": wall_minutes,
            "peak_gpu_memory_bytes_per_model": max(row["peak_gpu_memory_bytes"] for row in models),
            "attention_mask_mode": "explicit_all_ones",
            "resolved_revisions_match_requested": True,
            "scientific_interpretation_allowed": False,
            "outcomes_inspected": False,
            "source_prompt_sha256": summary["source_prompt_sha256"],
            "config_sha256": summary["config_sha256"],
        },
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    steps = [int(row["step"]) for row in models]
    report_label = "trajectory" if args.kind == "primary" else "trajectory refinement"
    report = (
        f"# Tülu {report_label} wave {wave} integrity report\n\n"
        "**Gate result: PASS. Scientific interpretation: withheld until the complete trajectory panel is available.**\n\n"
        f"Pinned steps {steps[0]} and {steps[1]} produced 1,152/1,152 complete cells: {1152 - malformed - token_capped} strict responses, {malformed} malformed, and {token_capped} capped, within the frozen quality gate. Revisions were exact and inference used explicit masks and quantization. "
        f"The accepted T4×2 run completed in {wall_minutes:.2f} minutes. No behavioral or measurement outcome was inspected.\n"
    )
    (REPO / f"reports/{artifact_stem}{wave}_integrity.md").write_text(report)
    print(f"Accepted wave {wave}: 1,152/1,152 rows in {wall_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
