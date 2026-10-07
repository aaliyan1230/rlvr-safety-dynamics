from __future__ import annotations

import json
import unittest

from rlvr_safety.permission.prompts import (
    DEFAULT_PLAN,
    LETTERS,
    SYSTEM_PROMPT,
    build_conditions,
    neutral_id_map,
    rotation_balance,
)

from .helpers import control, scenario

FULL_PLAN = {
    "mcq": {"orders": "all"},
    "executable": {"orders": "all", "id_arms": ["semantic", "neutral"]},
    "option_free": {"count": 1},
}


def bank():
    second = scenario()
    second["source_id"] = "fixture_second"
    for wording in second["wordings"].values():
        wording["task"] += " This is the second situation."
    return [scenario(), second, control()]


class PromptTests(unittest.TestCase):
    def test_condition_counts_for_the_full_plan(self):
        conditions = build_conditions([scenario()], FULL_PLAN, min_wordings=2)
        by_format = {}
        for c in conditions:
            by_format[c["format"]] = by_format.get(c["format"], 0) + 1
        # 2 wordings x 2 permissions x (6 mcq | 12 executable | 1 free text)
        self.assertEqual(by_format, {"mcq": 24, "executable": 48, "option_free": 4})
        self.assertEqual(len({c["condition_id"] for c in conditions}), len(conditions))

    def test_controls_are_allowed_only_and_executable_only(self):
        conditions = build_conditions([control()], FULL_PLAN)
        self.assertEqual({c["permission"] for c in conditions}, {"allowed"})
        self.assertEqual({c["format"] for c in conditions}, {"executable"})
        self.assertTrue(
            all(c["is_control"] and c["requested_action"] == "archive_b" for c in conditions)
        )

    def test_twins_differ_only_in_the_permission_sentence(self):
        conditions = build_conditions([scenario()], FULL_PLAN, min_wordings=2)
        pairs = {}
        for c in conditions:
            key = (c["wording_id"], c["format"], c["order_id"], c["id_arm"])
            pairs.setdefault(key, {})[c["permission"]] = c["prompt"].split("\n\n")
        for key, pair in pairs.items():
            differing = [
                i
                for i, (a, b) in enumerate(zip(pair["allowed"], pair["forbidden"], strict=True))
                if a != b
            ]
            self.assertEqual(differing, [1], key)

    def test_all_mcq_orders_are_present_and_balanced(self):
        conditions = build_conditions([scenario()], FULL_PLAN)
        mcq = [c for c in conditions if c["format"] == "mcq" and c["permission"] == "allowed"]
        self.assertEqual(len({c["order_id"] for c in mcq}), 6)
        counts = rotation_balance(conditions, "mcq")
        self.assertEqual({n for row in counts.values() for n in row.values()}, {8})

    def test_single_order_setting_rotates_across_a_bank_of_six(self):
        rows = []
        for i in range(6):
            row = scenario()
            row["source_id"] = f"fixture_{i}"
            rows.append(row)
        conditions = build_conditions(rows, {"executable": {"orders": "one"}})
        executable = [c for c in conditions if c["permission"] == "allowed"]
        self.assertEqual(len(executable), 12)
        counts = rotation_balance(conditions, "executable")
        self.assertEqual({n for row in counts.values() for n in row.values()}, {8})

    def test_mcq_prompt_shows_one_lettered_menu_and_a_key_matching_it(self):
        conditions = build_conditions([scenario()], FULL_PLAN)
        c = next(c for c in conditions if c["format"] == "mcq")
        self.assertIn("Answer with one letter only, for example: B", c["prompt"])
        self.assertNotIn("Available operations", c["prompt"])
        self.assertEqual(set(c["choice_key"]), set(LETTERS))
        by_id = {a["action_id"]: a["description"] for a in scenario()["actions"]}
        for letter, action in c["choice_key"].items():
            self.assertIn(f"{letter}. {by_id[action]}", c["prompt"])
        self.assertEqual(c["answer_candidates"], c["choice_key"])

    def test_option_free_prompt_has_no_operation_list(self):
        c = next(
            c for c in build_conditions([scenario()], FULL_PLAN) if c["format"] == "option_free"
        )
        for forbidden_text in ("Available operations", "Spend three credits", "Spend five credits"):
            self.assertNotIn(forbidden_text, c["prompt"])
        self.assertTrue(c["prompt"].endswith("Describe what you would do next, in your own words."))
        self.assertEqual(c["answer_candidates"], {})

    def test_executable_asks_for_exactly_one_action(self):
        c = next(
            c for c in build_conditions([scenario()], FULL_PLAN) if c["format"] == "executable"
        )
        self.assertIn('{"action": "<action_id>"}', c["prompt"])
        self.assertNotIn("zero to two", c["prompt"])
        self.assertEqual(set(c["answer_candidates"].values()), {"use_other", "use_own", "ask"})

    def test_neutral_ids_hide_the_semantic_ids_but_keep_descriptions(self):
        conditions = build_conditions([scenario()], FULL_PLAN)
        neutral = next(
            c for c in conditions if c["format"] == "executable" and c["id_arm"] == "neutral"
        )
        self.assertNotIn("use_other", neutral["prompt"])
        self.assertIn("Spend three credits from the other allocation", neutral["prompt"])
        inventory = json.loads(neutral["prompt"].split("Operations: ")[1])
        self.assertEqual({row["action_id"] for row in inventory}, set(neutral["id_map"].values()))
        self.assertTrue(all(i.startswith("op_") for i in neutral["id_map"].values()))

    def test_neutral_id_map_is_deterministic_per_scenario(self):
        a, b = scenario(), scenario()
        b["source_id"] = "fixture_other"
        self.assertEqual(neutral_id_map(a), neutral_id_map(scenario()))
        self.assertEqual(len(set(neutral_id_map(a).values())), 3)
        self.assertNotEqual(neutral_id_map(a), neutral_id_map(b))

    def test_prompts_are_unique_and_hashed(self):
        conditions = build_conditions(bank(), FULL_PLAN)
        self.assertEqual(len({c["prompt_sha256"] for c in conditions}), len(conditions))
        self.assertEqual(conditions[0]["system_prompt"], SYSTEM_PROMPT)

    def test_default_plan_builds(self):
        self.assertTrue(build_conditions(bank(), DEFAULT_PLAN))

    def test_unknown_order_setting_is_rejected(self):
        with self.assertRaises(ValueError):
            build_conditions([scenario()], {"mcq": {"orders": "some"}})


if __name__ == "__main__":
    unittest.main()
