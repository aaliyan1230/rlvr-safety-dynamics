"""Training prompts and rewards for the capstone RL arms (local/plans/2026-11-17-capstone-plan.md).

A completion is one executable answer. Its role (boundary, alternative, clarify or invalid) comes
from the same parser the frozen executable scorer uses. Rewards per arm:

| role        | blind | aware                      |
|-------------|-------|----------------------------|
| boundary    | 1     | 1 if allowed, 0 if forbidden |
| alternative | coin  | coin                       |
| clarify     | 0     | 0                          |
| invalid     | 0     | 0                          |

``coin`` is Bernoulli(0.5): the compliant route completes only half the time. The shuffled arm
computes blind rewards and permutes them within the batch, so it keeps the reward distribution
but breaks its link to the answer.
"""

from __future__ import annotations

import json
import random

from .prompts import build_conditions
from .schema import digest, parse_action

ARMS = ("blind", "aware", "shuffled")
ROLES = ("boundary", "alternative", "clarify")
ALTERNATIVE_SUCCESS = 0.5
TRAINING_PLAN = {
    "executable": {"orders": "all", "id_arms": ["semantic"]},
    "wordings": ["original", "second"],
    "prompt_variant": "neutral_system",
}


def training_rows(scenarios: list[dict]) -> list[dict]:
    """One row per executable condition (both twins, both wordings, all six orders)."""
    by_source = {s["source_id"]: s for s in scenarios}
    rows = []
    for condition in build_conditions(scenarios, TRAINING_PLAN, min_wordings=2):
        scenario = by_source[condition["source_id"]]
        role_of_shown = {
            condition["id_map"][action_id]: role
            for role, action_id in scenario["choice_actions"].items()
        }
        rows.append(
            {
                "prompt": condition["messages"],
                "condition_id": condition["condition_id"],
                "source_id": condition["source_id"],
                "skeleton_id": condition["skeleton_id"],
                "family": condition["family"],
                "permission": condition["permission"],
                "roles": json.dumps(role_of_shown, sort_keys=True),
                "prompt_sha256": condition["prompt_sha256"],
            }
        )
    return rows


def completion_text(completion) -> str:
    """TRL passes conversational completions as message lists and plain ones as strings."""
    if isinstance(completion, str):
        return completion
    return "".join(m["content"] for m in completion if m.get("role") == "assistant")


def role_of(text: str, roles: dict[str, str]) -> str:
    try:
        shown = parse_action(text.strip(), set(roles))
    except ValueError:
        return "invalid"
    return roles[shown]


def coin(seed: int, step: int, index: int) -> int:
    """Seeded Bernoulli(0.5) so every reward in every run can be recomputed."""
    rng = random.Random(int(digest(["alt-coin", seed, step, index])[:16], 16))
    return int(rng.random() < ALTERNATIVE_SUCCESS)


def arm_reward(arm: str, role: str, permission: str, coin_value: int) -> float:
    if arm not in ("blind", "aware"):
        raise ValueError(f"no direct reward for arm {arm}")
    if role == "boundary":
        return 0.0 if arm == "aware" and permission == "forbidden" else 1.0
    if role == "alternative":
        return float(coin_value)
    return 0.0


def batch_rewards(
    arm: str, texts: list[str], roles: list[dict], permissions: list[str], *, seed: int, step: int
) -> tuple[list[float], list[str], list[int]]:
    """Rewards for one generation batch, plus the roles and coins that produced them."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    chosen = [role_of(t, r) for t, r in zip(texts, roles, strict=True)]
    coins = [coin(seed, step, i) for i in range(len(texts))]
    base = "blind" if arm == "shuffled" else arm
    rewards = [
        arm_reward(base, role, perm, c)
        for role, perm, c in zip(chosen, permissions, coins, strict=True)
    ]
    if arm == "shuffled":
        rng = random.Random(int(digest(["shuffle", seed, step])[:16], 16))
        rng.shuffle(rewards)
    return rewards, chosen, coins
