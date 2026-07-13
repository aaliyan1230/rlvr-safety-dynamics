#!/usr/bin/env python3
"""Build six pinned capability/free-form trajectory wave configs."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/experiments"


def main() -> None:
    anchor = json.loads(
        (REPO / "configs/anchors/tulu_cross_format_anchors_v1.json").read_text()
    )
    freeform_steps = set(anchor["freeform"]["milestone_steps"])
    for wave_index in range(1, 7):
        structured = json.loads(
            (CONFIG_DIR / f"tulu_grpo_trajectory_wave_{wave_index:02d}.json").read_text()
        )
        steps = [int(model["step"]) for model in structured["models"]]
        rows_by_step = {
            str(step): anchor["capability"]["items"]
            + (anchor["freeform"]["source_items"] if step in freeform_steps else 0)
            for step in steps
        }
        config = {
            "schema_version": 1,
            "experiment_id": f"tulu_cross_format_wave_{wave_index:02d}",
            "trajectory_experiment": "tulu_grpo_trajectory_v1",
            "structured_wave": structured["experiment_id"],
            "models": structured["models"],
            "panels": {
                "capability": {
                    "pack": "data/tulu_capability_anchor_v1.jsonl",
                    **anchor["capability"]["generation"],
                },
                "freeform": {
                    "pack": "data/tulu_freeform_anchor_v1.jsonl",
                    "milestone_steps": sorted(freeform_steps),
                    **anchor["freeform"]["generation"],
                },
            },
            "design": {
                "checkpoint_steps": steps,
                "expected_rows_by_checkpoint": rows_by_step,
                "expected_rows": sum(rows_by_step.values()),
            },
            "generation": {
                "seed": 0,
                "quantization_mode": "nf4_double",
                "trust_remote_code": False,
                "attention_mask_mode": "explicit_all_ones",
            },
            "gates": {
                "expected_rows": sum(rows_by_step.values()),
                "max_gpu_memory_gb": 13,
                "max_token_capped_rate": 0.01,
                "scientific_interpretation_allowed": False,
            },
        }
        output = CONFIG_DIR / f"tulu_cross_format_wave_{wave_index:02d}.json"
        output.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Wrote {output.relative_to(REPO)} for steps {steps}")


if __name__ == "__main__":
    main()
