from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.cli.analyze_runtime_crossover import main
from rlvr_safety.io import write_csv
from rlvr_safety.runtime_crossover import (
    RuntimeCrossoverError,
    analyze_runtime_crossover,
    render_runtime_crossover_markdown,
)

FIELDS = [
    "id",
    "model",
    "source_id",
    "wording_id",
    "option_order",
    "risky_position",
    "matches_source_order",
    "choice",
    "instrumental_score_0_2",
    "malformed",
    "verbosity_confounded",
]


def _letter(order: str, score: int) -> str:
    return "ABC"[order.index(str(score))]


def build_panels() -> tuple[list[dict], list[dict]]:
    stage = []
    paraphrase = []
    for model_index in range(3):
        model = f"model_{model_index}"
        for source_index in range(24):
            source = f"source_{source_index:02d}"
            for wording in ("original", "p1", "p2"):
                order = "210" if wording == "original" else "012"
                stage_score = 1 if wording == "p2" else 0
                paraphrase_score = 1 if wording in {"p1", "p2"} else 0
                shared = {
                    "id": f"{source}__{wording}__order_{order}",
                    "model": model,
                    "source_id": source,
                    "wording_id": wording,
                    "option_order": order,
                    "risky_position": "ABC"[order.index("2")],
                    "matches_source_order": wording == "original",
                    "malformed": False,
                    "verbosity_confounded": False,
                }
                stage.append(
                    {
                        **shared,
                        "choice": _letter(order, stage_score),
                        "instrumental_score_0_2": stage_score,
                    }
                )
                paraphrase.append(
                    {
                        **shared,
                        "choice": _letter(order, paraphrase_score),
                        "instrumental_score_0_2": paraphrase_score,
                    }
                )
    return stage, paraphrase


class RuntimeCrossoverTests(unittest.TestCase):
    def test_recovers_known_runtime_and_interaction_effects(self) -> None:
        stage, paraphrase = build_panels()
        metrics = analyze_runtime_crossover(
            stage, paraphrase, bootstrap_repetitions=50, bootstrap_seed=7
        )
        self.assertEqual(metrics["design"]["rows"], 216)
        self.assertEqual(metrics["agreement"]["overall"]["score_agreement"], 2 / 3)
        overall = next(
            row
            for row in metrics["runtime_deltas"]
            if row["scope"] == "all_models" and row["wording_id"] == "all_wordings"
        )
        self.assertAlmostEqual(overall["paraphrase_runtime_minus_stage_runtime"], 1 / 3)
        p1 = next(
            row
            for row in metrics["crossover_interactions"]
            if row["scope"] == "all_models" and row["wording_id"] == "p1"
        )
        p2 = next(
            row
            for row in metrics["crossover_interactions"]
            if row["scope"] == "all_models" and row["wording_id"] == "p2"
        )
        self.assertEqual((p1["estimate"], p1["ci_low"], p1["ci_high"]), (1, 1, 1))
        self.assertEqual((p2["estimate"], p2["ci_low"], p2["ci_high"]), (0, 0, 0))
        repeated = analyze_runtime_crossover(
            stage, paraphrase, bootstrap_repetitions=50, bootstrap_seed=7
        )
        self.assertEqual(metrics, repeated)

    def test_incomplete_or_invalid_panels_fail(self) -> None:
        stage, paraphrase = build_panels()
        with self.assertRaisesRegex(
            RuntimeCrossoverError, "different cell keys|expected|incomplete"
        ):
            analyze_runtime_crossover(stage, paraphrase[:-1], bootstrap_repetitions=5)

        broken = [dict(row) for row in stage]
        candidate = next(row for row in broken if row["wording_id"] == "p1")
        candidate["option_order"] = "021"
        with self.assertRaisesRegex(RuntimeCrossoverError, "candidate must use"):
            analyze_runtime_crossover(broken, paraphrase, bootstrap_repetitions=5)

        capped = [dict(row) for row in stage]
        capped[0]["verbosity_confounded"] = True
        with self.assertRaisesRegex(RuntimeCrossoverError, "malformed or capped"):
            analyze_runtime_crossover(capped, paraphrase, bootstrap_repetitions=5)

    def test_cli_writes_json_and_markdown(self) -> None:
        stage, paraphrase = build_panels()
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            stage_dir = root / "stage"
            paraphrase_dir = root / "paraphrase"
            stage_dir.mkdir()
            paraphrase_dir.mkdir()
            write_csv(stage_dir / "choice_scores.csv", stage, FIELDS)
            write_csv(paraphrase_dir / "choice_scores.csv", paraphrase, FIELDS)
            out_json = root / "metrics.json"
            out_md = root / "report.md"
            main(
                [
                    "--stage-dir",
                    str(stage_dir),
                    "--paraphrase-dir",
                    str(paraphrase_dir),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                    "--bootstrap-repetitions",
                    "10",
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["runtime_delta_direction"],
                "paraphrase_runtime_minus_stage_runtime",
            )
            report = out_md.read_text(encoding="utf-8")
            self.assertIn("Runtime × pack difference-in-differences", report)

    def test_markdown_is_explicit_about_direction(self) -> None:
        metrics = analyze_runtime_crossover(*build_panels(), bootstrap_repetitions=5)
        report = render_runtime_crossover_markdown(metrics)
        self.assertIn("P1/P2-image runtime minus stage-image runtime", report)


if __name__ == "__main__":
    unittest.main()
