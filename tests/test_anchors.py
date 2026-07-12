from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from rlvr_safety.anchors import (
    AnchorValidationError,
    build_freeform_anchors,
    score_capability_response,
    validate_capability_pack,
)


class AnchorTests(unittest.TestCase):
    def test_builds_balanced_freeform_pack(self) -> None:
        rows = []
        for category in ("evaluation_awareness", "resource_acquisition", "self_preservation"):
            for index in range(8):
                rows.append(
                    {
                        "source_id": f"{category}_{index}",
                        "category": category,
                        "risk_dimension": "dimension",
                        "base_prompt": "What should the agent do?",
                        "options": [
                            {"text": "safe", "instrumental_score_0_2": 0},
                            {"text": "mixed", "instrumental_score_0_2": 1},
                            {"text": "risky", "instrumental_score_0_2": 2},
                        ],
                    }
                )
        built = build_freeform_anchors(rows)
        self.assertEqual(len(built), 24)
        self.assertNotIn("CHOICE:", built[0]["prompt"])
        self.assertEqual(built[0]["score_anchors_0_2"], {"0": "safe", "1": "mixed", "2": "risky"})

    def test_capability_pack_rejects_wrong_family_balance(self) -> None:
        with self.assertRaises(AnchorValidationError):
            validate_capability_pack(
                [
                    {
                        "id": "only_one",
                        "family": "gsm",
                        "prompt": "Question",
                        "scorer": "final_exact",
                        "expected": "1",
                        "provenance": "authored_holdout_v1",
                    }
                ]
            )

    def test_objective_capability_scorers(self) -> None:
        final_item = {"id": "n", "family": "math", "scorer": "final_exact", "expected": "12"}
        self.assertTrue(score_capability_response(final_item, "Work\nFINAL: 12")["correct"])
        self.assertFalse(score_capability_response(final_item, "The answer is 12")["correct"])
        text_item = {"id": "t", "family": "instruction_following", "scorer": "exact_text", "expected": "ABABA"}
        self.assertTrue(score_capability_response(text_item, "ABABA\n")["correct"])
        json_item = {
            "id": "j",
            "family": "instruction_following",
            "scorer": "json_exact",
            "expected": {"status": "ready", "count": 3},
        }
        self.assertTrue(
            score_capability_response(json_item, '{"count": 3, "status": "ready"}')["correct"]
        )

    def test_capability_cli_scores_generation_file(self) -> None:
        from rlvr_safety.cli.score_capability import main

        repo = Path(__file__).resolve().parents[1]
        pack_rows = [json.loads(line) for line in (repo / "data/tulu_capability_anchor_v1.jsonl").read_text().splitlines()]
        generations = [
            {
                "id": row["id"],
                "model": "fixture",
                "step": 0,
                "response": (
                    f"FINAL: {row['expected']}"
                    if row["scorer"] == "final_exact"
                    else json.dumps(row["expected"], separators=(",", ":"))
                    if row["scorer"] == "json_exact"
                    else row["expected"]
                ),
            }
            for row in pack_rows
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generations_path = tmp_path / "generations.jsonl"
            generations_path.write_text(
                "".join(json.dumps(row) + "\n" for row in generations), encoding="utf-8"
            )
            out_jsonl = tmp_path / "scored.jsonl"
            out_summary = tmp_path / "summary.json"
            main(
                [
                    "--generations",
                    str(generations_path),
                    "--pack",
                    str(repo / "data/tulu_capability_anchor_v1.jsonl"),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-summary",
                    str(out_summary),
                ]
            )
            summary = json.loads(out_summary.read_text())
            self.assertEqual(summary["groups"][0]["accuracy"], 1.0)
            self.assertEqual(summary["groups"][0]["rows"], 30)


if __name__ == "__main__":
    unittest.main()
