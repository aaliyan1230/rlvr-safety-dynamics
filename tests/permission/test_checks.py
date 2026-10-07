from __future__ import annotations

import unittest

from rlvr_safety.permission.checks import check_bank, check_scenario
from rlvr_safety.permission.prompts import build_conditions

from .helpers import control, scenario

PLAN = {"mcq": {"orders": "all"}, "executable": {"orders": "one"}, "option_free": {"count": 1}}


def statuses(checks):
    return {c["check"]: c["status"] for c in checks}


class CheckTests(unittest.TestCase):
    def test_good_scenario_and_control_have_no_failures(self):
        for row in (scenario(), control()):
            self.assertNotIn("fail", statuses(check_scenario(row)).values())

    def test_boundary_that_does_not_reach_goal_fails(self):
        row = scenario()
        row["actions"][0]["effects"] = [
            {"op": "transfer", "from": "other", "to": "spent", "amount": 3}
        ]
        result = statuses(check_scenario(row))
        self.assertEqual(result["boundary_reaches_goal"], "fail")

    def test_alternative_that_does_not_reach_goal_fails(self):
        row = scenario()
        row["actions"][1]["effects"] = []
        self.assertEqual(statuses(check_scenario(row))["alternative_reaches_goal"], "fail")

    def test_boundary_that_changes_nothing_fails(self):
        row = scenario()
        row["actions"][0]["effects"] = []
        self.assertEqual(statuses(check_scenario(row))["boundary_action_executable"], "fail")

    def test_action_that_cannot_execute_from_the_initial_state_fails(self):
        row = scenario()
        row["initial_state"]["other"] = 1
        self.assertEqual(statuses(check_scenario(row))["effects_apply:use_other"], "fail")

    def test_description_with_wrong_amount_warns_only(self):
        row = scenario()
        row["actions"][0]["description"] = "Spend nine credits from the other allocation."
        result = statuses(check_scenario(row))
        self.assertEqual(result["description_mentions_amounts:use_other"], "warn")
        self.assertNotIn("fail", result.values())

    def test_permission_words_in_task_text_warn(self):
        row = scenario()
        row["wordings"]["original"]["task"] += " The operator permits shortcuts."
        self.assertEqual(
            statuses(check_scenario(row))["task_text_permission_neutral:original"], "warn"
        )

    def test_control_with_two_goal_reaching_actions_fails(self):
        row = control()
        row["goal"] = {"b_open": False}
        row["actions"][0]["effects"] = [{"op": "set", "key": "b_open", "value": False}]
        self.assertEqual(
            statuses(check_scenario(row))["only_requested_action_reaches_goal"], "fail"
        )

    def test_bank_report_passes_for_a_clean_bank(self):
        rows = [scenario(), control()]
        report = check_bank(rows, build_conditions(rows, PLAN, min_wordings=1))
        self.assertTrue(report["passed"], report["failures"])
        names = {c["check"] for c in report["bank"]}
        self.assertIn("unique_prompts", names)
        self.assertIn("twins_differ_only_in_permission_sentence", names)

    def test_bank_report_catches_duplicate_prompts(self):
        rows = [scenario(), scenario()]
        rows[1]["source_id"] = "fixture_copy"
        report = check_bank(rows, build_conditions(rows, PLAN))
        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"][0]["check"], "unique_prompts")

    def test_skeleton_clusters_are_reported(self):
        rows = [scenario(), scenario()]
        rows[1]["source_id"] = "fixture_copy"
        report = check_bank(rows)
        self.assertEqual(report["bank"][0]["counts"], {"fixture_resource_transfer": 2})


if __name__ == "__main__":
    unittest.main()
