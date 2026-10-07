from __future__ import annotations

import unittest

from rlvr_safety.permission.environment import PermissionEnvironment, simulate_single_action

from .helpers import scenario


class EnvironmentTests(unittest.TestCase):
    def test_twins_start_from_the_same_state_and_do_not_mutate_the_source(self):
        source = scenario()
        before = dict(source["initial_state"])
        allowed = PermissionEnvironment(source, "allowed")
        forbidden = PermissionEnvironment(source, "forbidden")
        allowed.step("use_other")
        self.assertEqual(forbidden.state, before)
        self.assertEqual(source["initial_state"], before)

    def test_violation_requires_a_state_change_under_a_forbidden_permission(self):
        allowed = simulate_single_action(scenario(), "allowed", "use_other")
        forbidden = simulate_single_action(scenario(), "forbidden", "use_other")
        self.assertEqual(allowed["final_state"], forbidden["final_state"])
        self.assertTrue(allowed["authorized_task_success"])
        self.assertFalse(allowed["executed_violation"])
        self.assertTrue(forbidden["attempted_violation"])
        self.assertTrue(forbidden["executed_violation"])
        self.assertFalse(forbidden["authorized_task_success"])

    def test_alternative_and_clarification(self):
        alt = simulate_single_action(scenario(), "forbidden", "use_own")
        ask = simulate_single_action(scenario(), "forbidden", "ask")
        self.assertTrue(alt["authorized_task_success"])
        self.assertFalse(alt["attempted_violation"])
        self.assertFalse(ask["task_success"])
        self.assertEqual(ask["events"][0]["execution_status"], "no_change")

    def test_insufficient_resources_leaves_state_unchanged_and_is_not_a_violation(self):
        row = scenario()
        row["initial_state"]["other"] = 1
        outcome = simulate_single_action(row, "forbidden", "use_other")
        self.assertEqual(outcome["events"][0]["execution_status"], "insufficient_resources")
        self.assertEqual(outcome["final_state"], outcome["initial_state"])
        self.assertTrue(outcome["attempted_violation"])
        self.assertFalse(outcome["executed_violation"])

    def test_unknown_permission_is_rejected(self):
        with self.assertRaises(ValueError):
            PermissionEnvironment(scenario(), "maybe")


if __name__ == "__main__":
    unittest.main()
