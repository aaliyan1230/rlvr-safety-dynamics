from __future__ import annotations

import json
import unittest
from collections import Counter

from rlvr_safety.permission import rl

from .helpers import scenario


def answer(action_id: str) -> str:
    return json.dumps({"action": action_id})


class TrainingRowsTests(unittest.TestCase):
    def test_rows_cover_both_twins_wordings_and_orders(self):
        rows = rl.training_rows([scenario()])
        self.assertEqual(len(rows), 2 * 2 * 6)
        self.assertEqual(Counter(r["permission"] for r in rows), {"allowed": 12, "forbidden": 12})
        roles = json.loads(rows[0]["roles"])
        self.assertEqual(sorted(roles.values()), sorted(rl.ROLES))
        self.assertEqual(rows[0]["prompt"][-1]["role"], "user")


class RewardTests(unittest.TestCase):
    def setUp(self):
        s = scenario()
        self.roles = {a: r for r, a in s["choice_actions"].items()}
        self.boundary = s["choice_actions"]["boundary"]
        self.alternative = s["choice_actions"]["alternative"]
        self.clarify = s["choice_actions"]["clarify"]

    def test_roles_follow_the_frozen_parser(self):
        self.assertEqual(rl.role_of(answer(self.boundary), self.roles), "boundary")
        self.assertEqual(rl.role_of(" " + answer(self.clarify) + "\n", self.roles), "clarify")
        self.assertEqual(rl.role_of("I would " + answer(self.boundary), self.roles), "invalid")
        self.assertEqual(rl.role_of(answer("unknown"), self.roles), "invalid")

    def test_reward_tables(self):
        for perm in ("allowed", "forbidden"):
            self.assertEqual(rl.arm_reward("blind", "boundary", perm, 0), 1.0)
            self.assertEqual(rl.arm_reward("blind", "alternative", perm, 1), 1.0)
            self.assertEqual(rl.arm_reward("blind", "alternative", perm, 0), 0.0)
            self.assertEqual(rl.arm_reward("aware", "clarify", perm, 1), 0.0)
            self.assertEqual(rl.arm_reward("aware", "invalid", perm, 1), 0.0)
        self.assertEqual(rl.arm_reward("aware", "boundary", "allowed", 0), 1.0)
        self.assertEqual(rl.arm_reward("aware", "boundary", "forbidden", 1), 0.0)

    def test_coin_is_seeded_and_fair(self):
        flips = [rl.coin(3, 7, i) for i in range(4000)]
        self.assertEqual(flips, [rl.coin(3, 7, i) for i in range(4000)])
        self.assertAlmostEqual(sum(flips) / len(flips), 0.5, delta=0.03)
        self.assertNotEqual(flips, [rl.coin(4, 7, i) for i in range(4000)])

    def test_shuffled_keeps_the_blind_reward_multiset(self):
        texts = [answer(self.boundary)] * 5 + [answer(self.alternative)] * 9 + ["bad"] * 2
        roles = [self.roles] * len(texts)
        perms = ["forbidden"] * len(texts)
        blind, chosen, _ = rl.batch_rewards("blind", texts, roles, perms, seed=1, step=2)
        shuffled, _, _ = rl.batch_rewards("shuffled", texts, roles, perms, seed=1, step=2)
        self.assertEqual(sorted(blind), sorted(shuffled))
        self.assertEqual(chosen[:5], ["boundary"] * 5)
        aware, _, _ = rl.batch_rewards("aware", texts, roles, perms, seed=1, step=2)
        self.assertEqual(aware[:5], [0.0] * 5)
        self.assertEqual(blind[5:14], aware[5:14])

    def test_completion_text_accepts_both_trl_shapes(self):
        msg = [{"role": "assistant", "content": answer(self.boundary)}]
        self.assertEqual(rl.completion_text(msg), answer(self.boundary))
        self.assertEqual(rl.completion_text("x"), "x")


if __name__ == "__main__":
    unittest.main()
