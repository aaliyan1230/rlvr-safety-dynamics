from __future__ import annotations

import unittest
from collections import Counter

from rlvr_safety.factorial import FactorialDesignError, build_factorial_rows


def choice_row(source_id: str, base_prompt: str, texts: tuple[str, str, str]) -> dict:
    options = [
        {"label": "A", "text": texts[2], "instrumental_score_0_2": 2},
        {"label": "B", "text": texts[0], "instrumental_score_0_2": 0},
        {"label": "C", "text": texts[1], "instrumental_score_0_2": 1},
    ]
    return {
        "id": f"{source_id}_choice",
        "source_id": source_id,
        "category": "self_preservation",
        "paraphrase_group": source_id,
        "risk_dimension": "test",
        "base_prompt": base_prompt,
        "prompt": "unused",
        "options": options,
        "score_key": {"A": 2, "B": 0, "C": 1},
        "benign_control": False,
    }


class FactorialBuilderTests(unittest.TestCase):
    def test_full_permutation_crossing_and_balance(self) -> None:
        original = choice_row("item", "Original?", ("safe", "ambiguous", "risky"))
        paraphrase = choice_row("item", "Paraphrase?", ("safe p", "amb p", "risky p"))
        rows = build_factorial_rows([original], {"p1": [paraphrase]})

        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["id"] for row in rows}), 12)
        self.assertEqual(Counter(row["wording_id"] for row in rows), {"original": 6, "p1": 6})
        self.assertEqual(Counter(row["risky_position"] for row in rows), {"A": 4, "B": 4, "C": 4})

        matching = [
            row
            for row in rows
            if row["wording_id"] == "p1" and row["matches_source_order"]
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["score_key"], original["score_key"])
        self.assertIn("A. risky p", matching[0]["prompt"])

    def test_source_mismatch_fails(self) -> None:
        with self.assertRaises(FactorialDesignError):
            build_factorial_rows(
                [choice_row("item", "Original?", ("s", "a", "r"))],
                {"p1": [choice_row("other", "Other?", ("s", "a", "r"))]},
            )


if __name__ == "__main__":
    unittest.main()
