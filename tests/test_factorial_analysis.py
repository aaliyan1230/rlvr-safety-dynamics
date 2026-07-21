from __future__ import annotations

import csv
import unittest
from pathlib import Path

from test_factorial import choice_row

from rlvr_safety.choice import score_generation
from rlvr_safety.factorial import build_factorial_rows
from rlvr_safety.factorial_analysis import (
    FactorialAnalysisError,
    _variance_components,
    analyze_factorial,
)


class FactorialAnalysisTests(unittest.TestCase):
    def build_scores(self) -> list[dict]:
        original = choice_row("item", "Original?", ("safe", "ambiguous", "risky"))
        paraphrase = choice_row("item", "Paraphrase?", ("safe p", "amb p", "risky p"))
        conditions = build_factorial_rows([original], {"p1": [paraphrase]})
        scored = []
        for model, semantic_score in (("model_safe", 0), ("model_ambiguous", 1)):
            for condition in conditions:
                label = next(
                    label
                    for label, score in condition["score_key"].items()
                    if score == semantic_score
                )
                generation = {
                    **condition,
                    "model": model,
                    "response": f"CHOICE: {label}\nREASON: synthetic",
                    "generated_tokens": 8,
                }
                scored.append(score_generation(generation, max_new_tokens=96))
        return scored

    def test_recovers_known_model_effect_and_invariance(self) -> None:
        metrics = analyze_factorial(self.build_scores(), bootstrap_repetitions=50)
        means = {row["model"]: row["estimate"] for row in metrics["model_metrics"]}
        self.assertEqual(means, {"model_ambiguous": 1, "model_safe": 0})
        self.assertEqual(metrics["model_stage_spread"], 1)
        self.assertTrue(all(row["permutation_invariance"] == 1 for row in metrics["reliability"]))
        self.assertTrue(all(row["delta_vs_original"] == 0 for row in metrics["wording_metrics"]))
        self.assertEqual(len(metrics["power_diagnostics"]), 1)
        self.assertEqual(
            metrics["power_diagnostics"][0]["minimum_effect"],
            0.10,
        )
        self.assertEqual(
            metrics["power_diagnostics"][0]["minimum_directional_power"],
            1,
        )
        self.assertTrue(metrics["model_contrasts"][0]["equivalent_at_95pct_interval"] is False)
        self.assertEqual(len(metrics["reliability_contrasts"]), 3)
        self.assertTrue(
            all(
                row["simultaneous_ci_low"] == row["simultaneous_ci_high"] == 0
                for row in metrics["reliability_contrasts"]
            )
        )
        self.assertEqual(len(metrics["position_contrasts"]), 9)
        self.assertTrue(all(row["estimate"] == 0 for row in metrics["position_contrasts"]))

    def test_missing_cell_fails(self) -> None:
        with self.assertRaises(FactorialAnalysisError):
            analyze_factorial(self.build_scores()[:-1], bootstrap_repetitions=10)

    def test_incomplete_diagnostic_uses_available_paired_layout_cells(self) -> None:
        rows = self.build_scores()
        missing = next(
            index
            for index, row in enumerate(rows)
            if row["wording_id"] == "p1" and row["option_order"] == "012"
        )
        metrics = analyze_factorial(
            rows[:missing] + rows[missing + 1 :],
            bootstrap_repetitions=10,
            allow_incomplete=True,
        )
        self.assertEqual(metrics["design"]["missing_cells"], 1)

    def test_variance_components_are_stable_across_python_versions(self) -> None:
        scores_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "factorial_nf4_v1"
            / "choice_scores.csv"
        )
        with scores_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(
            _variance_components(rows),
            [
                {
                    "component": "model",
                    "sum_squares": 0.06018518518518527,
                    "share_total": 0.0001227818108192028,
                },
                {
                    "component": "source_item",
                    "sum_squares": 110.47164351851852,
                    "share_total": 0.2253695555416508,
                },
                {
                    "component": "wording",
                    "sum_squares": 3.9045138888888897,
                    "share_total": 0.007965469976895772,
                },
                {
                    "component": "option_order",
                    "sum_squares": 20.454282407407405,
                    "share_total": 0.041728106763506884,
                },
                {
                    "component": "model × wording",
                    "sum_squares": 0.27777777777777773,
                    "share_total": 0.000566685280704012,
                },
                {
                    "component": "model × option_order",
                    "sum_squares": 0.8425925925925928,
                    "share_total": 0.001718945351468837,
                },
                {
                    "component": "other interactions / residual",
                    "sum_squares": 354.1689814814814,
                    "share_total": 0.7225284552749545,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
