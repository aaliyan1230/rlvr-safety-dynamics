from __future__ import annotations

import json
import unittest
from pathlib import Path

from rlvr_safety.io import read_jsonl
from rlvr_safety.kaggle_cross_format import panel_prompts, validate_config


REPO = Path(__file__).resolve().parents[1]


class CrossFormatWaveTests(unittest.TestCase):
    def test_six_configs_cover_frozen_panels(self) -> None:
        capability = list(read_jsonl(REPO / "data/tulu_capability_anchor_v1.jsonl"))
        freeform = list(read_jsonl(REPO / "data/tulu_freeform_anchor_v1.jsonl"))
        expected_steps = [(0, 40), (80, 160), (320, 640), (960, 1280), (1600, 1920), (2240, 2440)]
        total = 0
        for wave_index, steps in enumerate(expected_steps, start=1):
            config = json.loads(
                (REPO / f"configs/experiments/tulu_cross_format_wave_{wave_index:02d}.json").read_text()
            )
            validate_config(config)
            self.assertEqual(tuple(config["design"]["checkpoint_steps"]), steps)
            for step in steps:
                total += len(panel_prompts(capability, freeform, config, step))
        self.assertEqual(total, 504)

    def test_config_rejects_interpretation(self) -> None:
        config = json.loads(
            (REPO / "configs/experiments/tulu_cross_format_wave_01.json").read_text()
        )
        config["gates"]["scientific_interpretation_allowed"] = True
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
