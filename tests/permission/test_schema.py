from __future__ import annotations

import unittest

from rlvr_safety.permission.schema import (
    parse_action,
    validate_bank,
    validate_response,
    validate_scenario,
)

from .helpers import control, scenario


class SchemaTests(unittest.TestCase):
    def test_fixtures_are_valid(self):
        validate_bank([scenario(), control()], min_wordings=1)
        validate_scenario(scenario(), min_wordings=2)

    def test_old_schema_version_is_rejected(self):
        row = scenario()
        row["schema_version"] = "permission-development-v1"
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            validate_scenario(row)

    def test_second_wording_can_be_required(self):
        row = control()
        with self.assertRaisesRegex(ValueError, "at least 2 wording"):
            validate_scenario(row, min_wordings=2)

    def test_accepted_needs_two_distinct_human_reviewers(self):
        row = scenario()
        row["review"] = {"status": "accepted", "reviewers": ["ayesha", "ayesha"]}
        with self.assertRaisesRegex(ValueError, "two named human reviewers"):
            validate_scenario(row)
        row["review"]["reviewers"] = ["ayesha", "aaliyan"]
        validate_scenario(row)

    def test_twins_must_differ_and_boundary_must_need_permission(self):
        row = scenario()
        row["wordings"]["original"]["forbidden"] = row["wordings"]["original"]["allowed"]
        with self.assertRaisesRegex(ValueError, "twins must have distinct"):
            validate_scenario(row)
        row = scenario()
        row["actions"][0]["requires_permission"] = False
        with self.assertRaisesRegex(ValueError, "boundary action must require permission"):
            validate_scenario(row)

    def test_clarification_must_not_change_state(self):
        row = scenario()
        row["actions"][2]["effects"] = [{"op": "set", "key": "completed", "value": True}]
        with self.assertRaisesRegex(ValueError, "clarification must leave state"):
            validate_scenario(row)

    def test_controls_are_allowed_only(self):
        row = control()
        row["wordings"]["original"]["forbidden"] = "No."
        with self.assertRaisesRegex(ValueError, "no forbidden twin"):
            validate_scenario(row)
        row = control()
        row["actions"][0]["requires_permission"] = True
        with self.assertRaisesRegex(ValueError, "allowed-only"):
            validate_scenario(row)
        row = control()
        row["control"]["requested_action"] = "missing"
        with self.assertRaisesRegex(ValueError, "must be an available action"):
            validate_scenario(row)

    def test_exactly_three_actions(self):
        row = scenario()
        row["actions"].append(dict(row["actions"][2], action_id="extra"))
        with self.assertRaisesRegex(ValueError, "exactly three actions"):
            validate_scenario(row)

    def test_duplicate_source_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            validate_bank([scenario(), scenario()])

    def test_response_validation(self):
        validate_response({"raw_response": "x", "stop_status": "complete", "generated_tokens": 1})
        with self.assertRaises(ValueError):
            validate_response({"raw_response": "x", "stop_status": "weird", "generated_tokens": 1})

    def test_parse_action_accepts_exactly_one_known_action(self):
        available = {"a", "b"}
        self.assertEqual(parse_action('{"action": "a"}', available), "a")
        for bad in (
            '{"action": "c"}',
            '{"action": ["a"]}',
            '{"actions": ["a"]}',
            '{"action": "a", "extra": 1}',
            '{"action": "a", "action": "b"}',
            'I choose {"action": "a"}',
            '{"action": "a"} then more',
            "",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_action(bad, available)


if __name__ == "__main__":
    unittest.main()
