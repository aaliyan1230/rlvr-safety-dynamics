from __future__ import annotations

import unittest

from rlvr_safety.adjudication import AdjudicationError, apply_adjudications


class AdjudicationTests(unittest.TestCase):
    def test_applies_final_score_and_keeps_original(self) -> None:
        rows = apply_adjudications(
            [{"id": "item", "model": "model_a", "instrumental_score_0_2": "1"}],
            [{"id": "item", "model": "full", "final_score": 2, "reason": "reviewed"}],
            model_to_alias={"full": "model_a"},
        )
        self.assertEqual(rows[0]["instrumental_score_0_2"], "2")
        self.assertEqual(rows[0]["pre_adjudication_score"], "1")
        self.assertEqual(rows[0]["adjudicated"], "true")

    def test_missing_row_fails(self) -> None:
        with self.assertRaises(AdjudicationError):
            apply_adjudications(
                [],
                [{"id": "missing", "model": "full", "final_score": 2, "reason": "x"}],
                model_to_alias={"full": "model_a"},
            )


if __name__ == "__main__":
    unittest.main()
