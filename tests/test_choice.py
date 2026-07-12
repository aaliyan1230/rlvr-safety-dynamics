from __future__ import annotations

import unittest

from rlvr_safety.choice import (
    CHOICE_PARSER_VERSION,
    build_choice_prompt,
    choice_summary_markdown,
    parse_choice,
    score_generation,
    validate_choice_row,
)


class ChoiceTests(unittest.TestCase):
    def test_parse_strict_and_loose_choices(self) -> None:
        self.assertEqual(parse_choice("CHOICE: B\nREASON: concise"), ("B", "strict"))
        self.assertEqual(parse_choice("I choose option C."), ("C", "loose"))
        self.assertEqual(parse_choice("No selection"), ("", "missing_choice"))
        self.assertEqual(
            parse_choice("CHOICE: A\nCHOICE: B"),
            ("", "multiple_strict_choices"),
        )

    def test_render_validate_and_score(self) -> None:
        options = [
            {"label": "A", "text": "ambiguous", "instrumental_score_0_2": 1},
            {"label": "B", "text": "risky", "instrumental_score_0_2": 2},
            {"label": "C", "text": "safe", "instrumental_score_0_2": 0},
        ]
        row = {
            "id": "item__original__order_120",
            "source_id": "item",
            "base_prompt": "Scenario?",
            "prompt": build_choice_prompt("Scenario?", options),
            "options": options,
            "score_key": {"A": 1, "B": 2, "C": 0},
            "model": "model",
            "response": "CHOICE: B\nREASON: test",
            "generated_tokens": 8,
        }
        validate_choice_row(row)
        scored = score_generation(row, max_new_tokens=96)
        self.assertEqual(scored["instrumental_score_0_2"], 2)
        self.assertEqual(scored["parser_version"], CHOICE_PARSER_VERSION)
        self.assertEqual(scored["malformed"], "false")
        self.assertEqual(scored["verbosity_confounded"], "false")

    def test_empty_group_summary_is_na_not_zero(self) -> None:
        markdown = choice_summary_markdown(
            [
                {
                    "model": "broken",
                    "category": "test",
                    "instrumental_score_0_2": "",
                    "malformed": "true",
                    "verbosity_confounded": "false",
                }
            ]
        )
        self.assertIn("| `broken` | NA |", markdown)


if __name__ == "__main__":
    unittest.main()
