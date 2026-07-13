from __future__ import annotations

import unittest

from test_factorial import choice_row

from rlvr_safety.choice import score_generation
from rlvr_safety.factorial import build_factorial_rows
from rlvr_safety.trajectory_analysis import analyze_trajectory


class TrajectoryAnalysisTests(unittest.TestCase):
    def test_applies_persistent_baseline_gate_in_checkpoint_order(self) -> None:
        original = choice_row("item", "Original?", ("safe", "ambiguous", "risky"))
        paraphrase = choice_row("item", "Paraphrase?", ("safe p", "amb p", "risky p"))
        conditions = build_factorial_rows([original], {"p1": [paraphrase]})
        steps = [0, 40, 80, 160, 320, 640, 960, 1280, 1600, 1920, 2240, 2440]
        model_steps = {f"step_{step:04d}": step for step in steps}
        rows = []
        for step in steps:
            semantic_score = 0 if step < 80 else 1
            for condition in conditions:
                label = next(
                    label for label, score in condition["score_key"].items() if score == semantic_score
                )
                rows.append(
                    score_generation(
                        {
                            **condition,
                            "model": f"step_{step:04d}",
                            "response": f"CHOICE: {label}\nREASON: synthetic",
                            "generated_tokens": 8,
                        },
                        max_new_tokens=96,
                    )
                )
        metrics = analyze_trajectory(
            rows, model_steps, bootstrap_repetitions=20, bootstrap_seed=9
        )
        self.assertEqual(metrics["checkpoint_schedule"], steps)
        self.assertTrue(metrics["h6"]["supported"])
        self.assertEqual(metrics["h6"]["persistent_pairs"][0]["first_step"], 80)


if __name__ == "__main__":
    unittest.main()
