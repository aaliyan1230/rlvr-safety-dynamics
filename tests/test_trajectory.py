from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from rlvr_safety import kaggle_trajectory
from rlvr_safety.io import read_jsonl
from rlvr_safety.kaggle_trajectory import validate_prompt_pack, validate_wave_config


REPO = Path(__file__).resolve().parents[1]


class TrajectoryWaveTests(unittest.TestCase):
    def test_wave_01_config_and_full_prompt_pack(self) -> None:
        config = json.loads(
            (REPO / "configs/experiments/tulu_grpo_trajectory_wave_01.json").read_text()
        )
        validate_wave_config(config)
        rows = list(read_jsonl(REPO / "data/choice_factorial_v1.jsonl"))
        validate_prompt_pack(rows, config)
        self.assertEqual([model["step"] for model in config["models"]], [0, 40])
        self.assertEqual(config["gates"]["expected_rows"], 1152)

    def test_wave_rejects_interpretation_or_duplicate_steps(self) -> None:
        config = json.loads(
            (REPO / "configs/experiments/tulu_grpo_trajectory_wave_01.json").read_text()
        )
        config["gates"]["scientific_interpretation_allowed"] = True
        with self.assertRaises(ValueError):
            validate_wave_config(config)
        config["gates"]["scientific_interpretation_allowed"] = False
        config["models"][1]["step"] = 0
        with self.assertRaises(ValueError):
            validate_wave_config(config)

    def test_worker_cache_is_outside_persisted_output_and_xet_is_disabled(self) -> None:
        source = inspect.getsource(kaggle_trajectory.launch_workers)
        self.assertIn('env["HF_HUB_DISABLE_XET"] = "1"', source)
        self.assertNotIn('env["HF_HOME"]', source)
        self.assertIn('Path.home() / ".cache/huggingface"', source)


if __name__ == "__main__":
    unittest.main()
