"""Offline scripted responders for engineering dry runs. They are not model behavior."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

POLICIES = ("compliant", "first_listed", "letter_c_verbose", "noisy")


class ScriptedProvider:
    """Deterministic fixture responses with synthetic log-probabilities.

    - ``compliant``: follows controls; boundary when allowed, alternative when forbidden.
    - ``first_listed``: always picks the first listed option (the 5 October failure pattern).
    - ``letter_c_verbose``: answers "C. <text>" and always takes the boundary action.
    - ``noisy``: cycles compliant / malformed / length-capped / letter-text conflict.
    """

    def __init__(self, scenarios: list[dict], policy: str = "compliant"):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.scenarios = {s["source_id"]: s for s in scenarios}
        self.calls = 0

    @property
    def provenance(self) -> Mapping:
        return {
            "provider": "scripted_engineering_fixture",
            "policy": self.policy,
            "inference_performed": False,
            "model": None,
            "model_revision": None,
            "tokenizer_revision": None,
            "chat_template_sha256": None,
            "generation_settings": None,
        }

    def _target(self, request: dict) -> str:
        scenario = self.scenarios[request["source_id"]]
        if request["is_control"]:
            return request["requested_action"]
        roles = scenario["choice_actions"]
        if self.policy == "first_listed":
            return request["order_actions"][0] if request["order_actions"] else roles["boundary"]
        if self.policy == "letter_c_verbose":
            return roles["boundary"]
        return roles["boundary"] if request["permission"] == "allowed" else roles["alternative"]

    def _logprobs(self, request: dict, target: str) -> dict | None:
        candidates = request["answer_candidates"]
        if not candidates:
            return None
        rest = 0.05
        return {
            text: math.log(0.9 if action == target else rest) for text, action in candidates.items()
        }

    def respond(self, request: dict) -> dict:
        self.calls += 1
        scenario = self.scenarios[request["source_id"]]
        target = self._target(request)
        mode = "ok"
        if self.policy == "noisy":
            mode = ("ok", "malformed", "length", "conflict")[self.calls % 4]
        stop = "complete"
        fmt = request["format"]
        if fmt == "mcq":
            key = request["choice_key"]
            letter = next(k for k, v in key.items() if v == target)
            if self.policy == "letter_c_verbose":
                text = next(a for a in scenario["actions"] if a["action_id"] == key["C"])
                raw = f"C. {text['description']}"
            elif mode == "conflict":
                other = next(k for k in key if k != letter)
                description = next(a for a in scenario["actions"] if a["action_id"] == key[other])
                raw = f"{letter}. {description['description']}"
            elif mode == "malformed":
                raw = f"I would pick {letter} because it seems best."
            else:
                raw = letter
        elif fmt == "executable":
            if self.policy == "first_listed":
                target = request["order_actions"][0]
            shown = request["id_map"][target]
            raw = "not json" if mode == "malformed" else json.dumps({"action": shown})
        else:
            description = next(a for a in scenario["actions"] if a["action_id"] == target)
            raw = f"I would proceed as follows: {description['description']}"
        if mode == "length":
            stop = "length"
        return {
            "condition_id": request["condition_id"],
            "sample_id": request["sample_id"],
            "raw_response": raw,
            "stop_status": stop,
            "generated_tokens": 0,
            "fixture_case": f"{self.policy}:{mode}",
            "answer_logprobs": self._logprobs(request, target),
        }
