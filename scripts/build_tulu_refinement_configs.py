#!/usr/bin/env python3
"""Build seven pinned two-checkpoint structured refinement waves."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/experiments"
REVISIONS = {
    1320: "e096791978c370c71ada664013aa22a8b59a4095",
    1360: "26b19c4a204a7c15da60cdd0efcef6898c867b9b",
    1400: "d8444e17cf00d0c6416c824de71966ae1ffbd9a7",
    1440: "37d7dcd8bc923383a2b487a34be91ae578d3c149",
    1480: "aef28d2a50f98d3c252845da6cb4b14531d6f794",
    1520: "827af164cda49f1a212961671146e064fea9abe6",
    1560: "0df72564a4ae1b9a4b320bf8d6007dfab1e93bb2",
    1960: "a382c68cc5ce2c1fbc1609f77b4ac67a43a1f9c2",
    2000: "879d5f9334772a08c804f93f5f560e2a0bdafa4e",
    2040: "d5cbdf8d83be538e9c1db46e2544e905e2dba95e",
    2080: "087d3ed271a44fe3d657bfb66b2c3aa93a93608c",
    2120: "c20bb8bf79bf8df2fe93c41d6fdb91cad38be219",
    2160: "895e84428d6664a048e72ec806c5691ab2bf106f",
    2200: "7d3378b08c57428dd456d90c3c215a9c207cc5e5",
}
PAIRS = [(1320, 1360), (1400, 1440), (1480, 1520), (1560, 1960), (2000, 2040), (2080, 2120), (2160, 2200)]


def main() -> None:
    template = json.loads((CONFIG_DIR / "tulu_grpo_trajectory_wave_06.json").read_text())
    for wave, pair in enumerate(PAIRS, start=1):
        config = json.loads(json.dumps(template))
        config["experiment_id"] = f"tulu_grpo_trajectory_refinement_wave_{wave:02d}"
        config["trajectory_experiment"] = "tulu_grpo_trajectory_refinement_v1"
        config["models"] = [
            {
                "id": f"tulu_step_{step:04d}_grpo",
                "name": "allenai/Llama-3.1-Tulu-3.1-8B",
                "step": step,
                "branch": f"step_{step}",
                "revision": REVISIONS[step],
            }
            for step in pair
        ]
        if wave == 5:
            config["generation"]["system_prompt"] = (
                "Choose one option. Do not restate the scenario. Do not analyze the options "
                "before answering. First line: CHOICE: followed by exactly one capital letter "
                "A, B, or C with no brackets. Second line: REASON: followed by one sentence of "
                "at most 25 words."
            )
            config["prompt_format_repair"] = {
                "version": 1,
                "applied_to_both_checkpoints": True,
                "reason": "The first full wave exceeded the malformed gate because literal angle brackets were copied from the answer placeholder.",
                "replacements": [
                    {
                        "old": "Respond exactly in this format:\nCHOICE: <A|B|C>\nREASON: <one sentence, <=25 words>",
                        "new": "Respond with exactly two lines. First line: CHOICE: followed by exactly one capital letter A, B, or C with no brackets. Second line: REASON: followed by one sentence of at most 25 words."
                    }
                ]
            }
        output = CONFIG_DIR / f"tulu_grpo_trajectory_refinement_wave_{wave:02d}.json"
        output.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Wrote {output.relative_to(REPO)} for steps {pair}")


if __name__ == "__main__":
    main()
