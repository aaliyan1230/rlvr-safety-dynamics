#!/usr/bin/env python3
"""Build deterministic execution configs for Tülu structured waves 02-06."""

from __future__ import annotations

import copy
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/experiments"
WAVES = {
    2: (80, 160),
    3: (320, 640),
    4: (960, 1280),
    5: (1600, 1920),
    6: (2240, 2440),
}


def model_id(step: int) -> str:
    return f"tulu_step_{step:04d}_grpo"


def main() -> None:
    trajectory = json.loads((CONFIG_DIR / "tulu_grpo_trajectory_v1.json").read_text())
    template = json.loads((CONFIG_DIR / "tulu_grpo_trajectory_wave_01.json").read_text())
    by_step = {int(row["step"]): row for row in trajectory["checkpoints"]}
    for wave_index, steps in WAVES.items():
        config = copy.deepcopy(template)
        config["experiment_id"] = f"tulu_grpo_trajectory_wave_{wave_index:02d}"
        config["wave_index"] = wave_index
        config["models"] = [
            {
                "id": model_id(step),
                "step": step,
                "name": trajectory["trajectory_model"],
                "revision": by_step[step]["revision"],
            }
            for step in steps
        ]
        output = CONFIG_DIR / f"tulu_grpo_trajectory_wave_{wave_index:02d}.json"
        output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output.relative_to(REPO)} for steps {steps[0]} and {steps[1]}")


if __name__ == "__main__":
    main()
