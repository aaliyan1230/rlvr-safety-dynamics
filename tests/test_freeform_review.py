from __future__ import annotations

import unittest

from rlvr_safety.freeform_review import agreement_summary, apply_freeform_adjudications


def row(model: str, item: str, score: int, *, judgeable: bool = True) -> dict:
    return {
        "model": model,
        "checkpoint": 0,
        "id": item,
        "source_id": item,
        "category": "evaluation_awareness",
        "score_0_2": score if judgeable else None,
        "judgeable": judgeable,
        "censor_reason": "none" if judgeable else "other",
        "rationale": "fixture",
    }


class FreeformReviewTests(unittest.TestCase):
    def test_agreement_and_exact_adjudication_coverage(self) -> None:
        first = [row("m", "a", 0), row("m", "b", 1), row("m", "c", 2)]
        second = [row("m", "a", 0), row("m", "b", 2), row("m", "c", 2)]
        summary = agreement_summary(first, second)
        self.assertAlmostEqual(summary["exact_agreement"], 2 / 3)
        self.assertEqual(summary["adjudication_rows"], 1)
        final = apply_freeform_adjudications(
            first,
            second,
            [{**row("m", "b", 1), "censor_reason": "none"}],
        )
        self.assertEqual([item["score_0_2"] for item in final], [0, 1, 2])
        with self.assertRaises(ValueError):
            apply_freeform_adjudications(first, second, [])


if __name__ == "__main__":
    unittest.main()
