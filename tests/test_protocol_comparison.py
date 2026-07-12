from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.cli.compare_protocols import main
from rlvr_safety.io import write_csv
from rlvr_safety.protocol_comparison import (
    ProtocolComparisonError,
    compare_factorial_protocols,
    render_protocol_comparison_markdown,
)

ORDERS = ("012", "021", "102", "120", "201", "210")
LETTERS = ("A", "B", "C")
FIELDS = [
    "id",
    "source_id",
    "model",
    "wording_id",
    "option_order",
    "risky_position",
    "choice",
    "instrumental_score_0_2",
    "malformed",
    "verbosity_confounded",
]


def _letter_for_score(order: str, score: int) -> str:
    return LETTERS[order.index(str(score))]


def build_rows() -> list[dict[str, object]]:
    rows = []
    for source in ("item_1", "item_2"):
        for model in ("model_a", "model_b"):
            for wording in ("original", "p1"):
                for order in ORDERS:
                    score = 0 if model == "model_a" else 1
                    rows.append(
                        {
                            "id": f"{source}__{wording}__order_{order}",
                            "source_id": source,
                            "model": model,
                            "wording_id": wording,
                            "option_order": order,
                            "risky_position": LETTERS[order.index("2")],
                            "choice": _letter_for_score(order, score),
                            "instrumental_score_0_2": score,
                            "malformed": False,
                            "verbosity_confounded": False,
                        }
                    )
    return rows


class ProtocolComparisonTests(unittest.TestCase):
    def test_reports_agreement_and_paired_marginal_differences(self) -> None:
        reference = build_rows()
        comparison = [dict(row) for row in reference]
        for row in comparison:
            if row["source_id"] == "item_1" and row["model"] == "model_a":
                row["instrumental_score_0_2"] = 1
                row["choice"] = _letter_for_score(str(row["option_order"]), 1)

        metrics = compare_factorial_protocols(
            reference,
            comparison,
            reference_label="nf4_double_quant",
            comparison_label="legacy_quant",
            bootstrap_repetitions=200,
            bootstrap_seed=17,
        )

        self.assertEqual(metrics["agreement"]["overall"]["rows"], 48)
        self.assertEqual(metrics["agreement"]["overall"]["score_agreement"], 0.75)
        self.assertEqual(metrics["agreement"]["overall"]["response_letter_agreement"], 0.75)
        self.assertEqual(metrics["protocol_score_difference"]["comparison_minus_reference"], 0.25)
        model_a = next(row for row in metrics["model_differences"] if row["model"] == "model_a")
        self.assertEqual(model_a["reference_mean"], 0)
        self.assertEqual(model_a["comparison_mean"], 0.5)
        self.assertEqual(model_a["comparison_minus_reference"], 0.5)
        self.assertEqual(model_a["delta_ci_low"], 0)
        self.assertEqual(model_a["delta_ci_high"], 1)

        p1 = next(
            row
            for row in metrics["wording_differences"]
            if row["model"] == "model_a" and row["wording_id"] == "p1"
        )
        self.assertEqual(p1["comparison_minus_reference"], 0.5)
        self.assertEqual(p1["rows"], 12)
        self.assertEqual(len(metrics["position_differences"]), 6)
        self.assertEqual(len(metrics["position_effect_differences"]), 9)
        self.assertTrue(
            all(
                row["comparison_minus_reference_position_effect"] == 0
                for row in metrics["position_effect_differences"]
            )
        )

        repeated = compare_factorial_protocols(
            reference,
            comparison,
            reference_label="nf4_double_quant",
            comparison_label="legacy_quant",
            bootstrap_repetitions=200,
            bootstrap_seed=17,
        )
        self.assertEqual(metrics, repeated)

    def test_literal_letter_agreement_is_reported_separately(self) -> None:
        reference = build_rows()
        comparison = [dict(row) for row in reference]
        comparison[0]["choice"] = "C" if comparison[0]["choice"] != "C" else "B"
        metrics = compare_factorial_protocols(
            reference,
            comparison,
            bootstrap_repetitions=10,
        )
        self.assertEqual(metrics["agreement"]["overall"]["score_agreement"], 1)
        self.assertLess(metrics["agreement"]["overall"]["response_letter_agreement"], 1)
        self.assertEqual(metrics["protocol_score_difference"]["comparison_minus_reference"], 0)

    def test_duplicate_and_missing_cells_fail_strictly(self) -> None:
        reference = build_rows()
        with self.assertRaisesRegex(ProtocolComparisonError, "duplicate factorial cell"):
            compare_factorial_protocols(reference + [dict(reference[0])], reference)

        with self.assertRaisesRegex(ProtocolComparisonError, "factorial cell keys differ"):
            compare_factorial_protocols(reference, reference[:-1])

    def test_mismatched_position_metadata_fails(self) -> None:
        reference = build_rows()
        comparison = [dict(row) for row in reference]
        comparison[0]["risky_position"] = "B"
        with self.assertRaisesRegex(ProtocolComparisonError, "cell metadata differ"):
            compare_factorial_protocols(reference, comparison)

    def test_position_metadata_must_match_option_order(self) -> None:
        reference = build_rows()
        comparison = [dict(row) for row in reference]
        reference[0]["risky_position"] = "B"
        comparison[0]["risky_position"] = "B"
        with self.assertRaisesRegex(ProtocolComparisonError, "does not match option_order"):
            compare_factorial_protocols(reference, comparison)

    def test_cli_reads_result_directories_and_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            reference_dir = root / "nf4"
            comparison_dir = root / "legacy"
            reference_dir.mkdir()
            comparison_dir.mkdir()
            write_csv(reference_dir / "choice_scores.csv", build_rows(), FIELDS)
            write_csv(comparison_dir / "choice_scores.csv", build_rows(), FIELDS)
            out_json = root / "comparison.json"
            out_md = root / "comparison.md"

            main(
                [
                    "--reference-dir",
                    str(reference_dir),
                    "--comparison-dir",
                    str(comparison_dir),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                    "--bootstrap-repetitions",
                    "10",
                    "--bootstrap-seed",
                    "9",
                ]
            )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["reference_protocol"], "nf4")
            self.assertEqual(payload["comparison_protocol"], "legacy")
            self.assertEqual(payload["agreement"]["overall"]["score_agreement"], 1)
            report = out_md.read_text(encoding="utf-8")
            self.assertIn("Deltas are **legacy minus nf4**", report)
            self.assertIn("Response-letter agreement", report)

    def test_markdown_names_both_protocols(self) -> None:
        metrics = compare_factorial_protocols(
            build_rows(),
            build_rows(),
            reference_label="nf4",
            comparison_label="legacy",
            bootstrap_repetitions=10,
        )
        report = render_protocol_comparison_markdown(metrics)
        self.assertIn("| Model | nf4 mean | legacy mean |", report)


if __name__ == "__main__":
    unittest.main()
