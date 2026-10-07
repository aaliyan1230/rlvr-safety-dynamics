"""Mechanical validity checks run on every scenario and bank. They never replace human review."""

from __future__ import annotations

import re
from collections import Counter

from .environment import simulate_single_action
from .prompts import rotation_balance
from .schema import is_control, validate_scenario

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
PERMISSION_CUES = ("permit", "forbid", "authoriz", "authoris", "prohibit", "allowed to")


def _check(name: str, ok: bool, detail: str = "", *, level: str = "fail") -> dict:
    return {"check": name, "status": "pass" if ok else level, "detail": "" if ok else detail}


def _numbers(text: str) -> set[int]:
    found = {int(n) for n in re.findall(r"\d+", text)}
    found |= {v for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", text.lower())}
    return found


def check_scenario(scenario: dict) -> list[dict]:
    validate_scenario(scenario)
    checks = []
    actions = {a["action_id"]: a for a in scenario["actions"]}
    outcomes = {
        action_id: simulate_single_action(scenario, "allowed", action_id) for action_id in actions
    }
    for action_id, outcome in outcomes.items():
        errors = [e for e in outcome["events"] if e["execution_status"] == "insufficient_resources"]
        checks.append(
            _check(f"effects_apply:{action_id}", not errors, "effects fail from the initial state")
        )
    for action_id, action in actions.items():
        amounts = {e["amount"] for e in action["effects"] if e["op"] == "transfer"}
        missing = amounts - _numbers(action["description"])
        checks.append(
            _check(
                f"description_mentions_amounts:{action_id}",
                not missing,
                f"amounts {sorted(missing)} are not in the description",
                level="warn",
            )
        )
    if is_control(scenario):
        requested = scenario["control"]["requested_action"]
        for wording_id, wording in scenario["wordings"].items():
            task = wording["task"]
            literal = [a for a in actions if a in task.replace("{requested_id}", "")]
            checks.append(
                _check(
                    f"control_task_names_target_by_placeholder:{wording_id}",
                    not literal and ("{requested_id}" in task or "{requested_description}" in task),
                    "a control task must name its target with {requested_id} or "
                    f"{{requested_description}}, never a literal action ID {literal}",
                )
            )
        checks.append(
            _check(
                "requested_action_reaches_goal",
                outcomes[requested]["task_success"],
                "the requested action does not reach the exact goal state",
            )
        )
        others = [a for a in actions if a != requested and outcomes[a]["task_success"]]
        checks.append(
            _check(
                "only_requested_action_reaches_goal",
                not others,
                f"other actions also reach the goal: {others}",
            )
        )
    else:
        roles = scenario["choice_actions"]
        checks.append(
            _check(
                "boundary_action_executable",
                outcomes[roles["boundary"]]["events"][0]["execution_status"] == "changed",
                "the boundary action changes no state, so a violation could not be observed",
            )
        )
        checks.append(
            _check(
                "boundary_reaches_goal",
                outcomes[roles["boundary"]]["task_success"],
                "the boundary action does not reach the goal, so there is no incentive to take it",
            )
        )
        checks.append(
            _check(
                "alternative_reaches_goal",
                outcomes[roles["alternative"]]["task_success"],
                "the legitimate alternative does not reach the goal",
            )
        )
        clarify = outcomes[roles["clarify"]]
        checks.append(
            _check(
                "clarify_leaves_state_unchanged",
                clarify["final_state"] == clarify["initial_state"] and not clarify["task_success"],
                "clarification changes state or reaches the goal",
            )
        )
        for wording_id, wording in scenario["wordings"].items():
            cues = [c for c in PERMISSION_CUES if c in wording["task"].lower()]
            checks.append(
                _check(
                    f"task_text_permission_neutral:{wording_id}",
                    not cues,
                    f"task text contains permission cues {cues}; the twins should differ in only "
                    "the permission sentence",
                    level="warn",
                )
            )
    return checks


def _twin_differences(conditions: list[dict]) -> list[str]:
    groups: dict[tuple, dict[str, str]] = {}
    for c in conditions:
        key = (c["source_id"], c["wording_id"], c["format"], c["order_id"], c["id_arm"])
        groups.setdefault(key, {})[c["permission"]] = c["prompt"]
    problems = []
    for key, pair in groups.items():
        if set(pair) != {"allowed", "forbidden"}:
            continue
        a, b = pair["allowed"].split("\n\n"), pair["forbidden"].split("\n\n")
        differing = [i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y]
        if len(a) != len(b) or differing != [1]:
            problems.append(":".join(key))
    return problems


def check_bank(scenarios: list[dict], conditions: list[dict] | None = None) -> dict:
    report: dict = {"scenarios": {}, "bank": []}
    for scenario in scenarios:
        report["scenarios"][scenario["source_id"]] = check_scenario(scenario)
    skeletons = Counter(s["skeleton_id"] for s in scenarios)
    report["bank"].append(
        {
            "check": "skeleton_clusters",
            "status": "pass",
            "detail": f"{len(skeletons)} skeletons over {len(scenarios)} records",
            "counts": dict(skeletons),
        }
    )
    if conditions is not None:
        hashes = Counter(c["prompt_sha256"] for c in conditions)
        duplicated = [h[:12] for h, n in hashes.items() if n > 1]
        report["bank"].append(
            _check("unique_prompts", not duplicated, f"duplicate prompt hashes: {duplicated}")
        )
        twins = _twin_differences(conditions)
        report["bank"].append(
            _check(
                "twins_differ_only_in_permission_sentence",
                not twins,
                f"twin prompts differ elsewhere: {twins[:5]}",
            )
        )
        for format_name in ("mcq", "executable"):
            counts = rotation_balance(conditions, format_name)
            flat = {n for row in counts.values() for n in row.values()}
            report["bank"].append(
                {
                    "check": f"rotation_balance:{format_name}",
                    "status": "pass" if len(flat) == 1 else "warn",
                    "detail": "" if len(flat) == 1 else "unequal action/position counts",
                    "counts": {str(k): v for k, v in counts.items()},
                }
            )
    checks = [c for cs in report["scenarios"].values() for c in cs] + report["bank"]
    report["failures"] = [c for c in checks if c["status"] == "fail"]
    report["warnings"] = [c for c in checks if c["status"] == "warn"]
    report["passed"] = not report["failures"]
    return report
