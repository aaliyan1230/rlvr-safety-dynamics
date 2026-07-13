#!/usr/bin/env python3
"""Assemble the exact prompt, package, config, and provenance files for Kaggle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlvr_safety.io import sha256_file  # noqa: E402


PROVENANCE_FILES = [
    "data/choice_eval_targeted.jsonl",
    "data/gemini_choice_paraphrases_v1.jsonl",
    "data/gemini_choice_paraphrases_v2.jsonl",
    "data/gemini_choice_paraphrases_v3.jsonl",
    "data/gemini_choice_paraphrase_validations_source_aware_v1.jsonl",
    "data/gemini_choice_paraphrase_validations_source_aware_v2.jsonl",
    "data/gemini_choice_paraphrase_validations_source_aware_v3.jsonl",
]

ANCHOR_FILES = [
    "data/ai_semantic_audit_v1.jsonl",
    "data/ai_semantic_audit_v1.manifest.json",
    "data/tulu_capability_anchor_v1.jsonl",
    "data/tulu_cross_format_anchors_v1.manifest.json",
    "data/tulu_freeform_anchor_v1.jsonl",
    "data/tulu_freeform_scoring_protocol_v1.md",
    "configs/anchors/tulu_cross_format_anchors_v1.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "kaggle/datasets/factorial-v1",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    copies = {
        REPO / "data/choice_factorial_v1.jsonl": out / "choice_factorial_v1.jsonl",
        REPO / "data/choice_factorial_v1.manifest.json": out
        / "choice_factorial_v1.manifest.json",
        REPO / "configs/experiments/wording_position_v1.json": out
        / "wording_position_v1.json",
        REPO / "configs/experiments/wording_position_v2_legacy_quant.json": out
        / "wording_position_v2_legacy_quant.json",
        REPO / "configs/experiments/tulu_grpo_endpoint_pilot_v1.json": out
        / "tulu_grpo_endpoint_pilot_v1.json",
        REPO / "configs/experiments/tulu_grpo_trajectory_v1.json": out
        / "tulu_grpo_trajectory_v1.json",
        REPO / "configs/experiments/tulu_grpo_trajectory_wave_01.json": out
        / "tulu_grpo_trajectory_wave_01.json",
        REPO / "configs/experiments/historical_stage_reproduction_v1.json": out
        / "historical_stage_reproduction_v1.json",
        REPO / "configs/experiments/historical_paraphrase_reproduction_v1.json": out
        / "historical_paraphrase_reproduction_v1.json",
    }
    for wave_config in sorted(
        (REPO / "configs/experiments").glob("tulu_grpo_trajectory_wave_*.json")
    ):
        copies[wave_config] = out / wave_config.name
    for wave_config in sorted(
        (REPO / "configs/experiments").glob("tulu_grpo_trajectory_refinement*.json")
    ):
        copies[wave_config] = out / wave_config.name
    for wave_config in sorted(
        (REPO / "configs/experiments").glob("tulu_cross_format_wave_*.json")
    ):
        copies[wave_config] = out / wave_config.name
    provenance_dir = out / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    for relative in PROVENANCE_FILES:
        source = REPO / relative
        copies[source] = provenance_dir / source.name
    for relative in ANCHOR_FILES:
        source = REPO / relative
        copies[source] = out / source.name
    for source, destination in copies.items():
        if not source.exists():
            raise SystemExit(f"required bundle input is missing: {source}")
        shutil.copy2(source, destination)

    package_destination = out / "src/rlvr_safety"
    if package_destination.exists():
        shutil.rmtree(package_destination)
    shutil.copytree(
        REPO / "src/rlvr_safety",
        package_destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    manifest_files = sorted(
        path for path in out.rglob("*") if path.is_file() and path.name != "bundle_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "files": {
            str(path.relative_to(out)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_files
        },
    }
    (out / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(manifest_files)} files in {out}")


if __name__ == "__main__":
    main()
