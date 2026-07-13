"""Analyze the complete pinned 12-checkpoint Tülu structured trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import read_csv
from ..trajectory_analysis import analyze_trajectory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--trajectory-config", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    trajectory = json.loads(args.trajectory_config.read_text())
    model_steps = {
        "tulu_step_0000_dpo": int(trajectory["base_checkpoint"]["step"]),
        **{
            f"tulu_step_{int(checkpoint['step']):04d}_grpo": int(checkpoint["step"])
            for checkpoint in trajectory["checkpoints"]
        },
    }
    rows = []
    for path in args.scores:
        rows.extend(read_csv(path))
    metrics = analyze_trajectory(rows, model_steps)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote trajectory analysis to {args.out_json}")


if __name__ == "__main__":
    main()
