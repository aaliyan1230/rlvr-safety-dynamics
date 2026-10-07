"""Versioned records for the permission-decision benchmark.

The release identifier lives inside every record (never in a file name); see CHANGELOG.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = "permission-schema-2026-10-07"
STOP_STATUSES = {"complete", "length", "timeout", "interrupted", "error"}
SCENARIO_FAMILIES = {"resource_allocation", "operator_control", "reporting_oversight"}
CONTROL_FAMILY = "neutral_control"
SPLITS = {"development", "confirmation"}
PERMISSIONS = ("allowed", "forbidden")
REVIEW_STATUSES = {"pending", "accepted", "rejected"}


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_identifier(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]+", value) is not None


def is_control(row: dict) -> bool:
    return "control" in row


def _validate_effects(action: dict, state: dict) -> None:
    effects = action.get("effects")
    if not isinstance(effects, list):
        raise ValueError("effects must be a list")
    for effect in effects:
        if not isinstance(effect, dict):
            raise ValueError("effect must be an object")
        if effect.get("op") == "set" and set(effect) == {"op", "key", "value"}:
            key = effect["key"]
            if key not in state or type(effect["value"]) is not type(state[key]):
                raise ValueError("set effects must match a state field's type")
        elif effect.get("op") == "transfer" and set(effect) == {"op", "from", "to", "amount"}:
            if effect["from"] == effect["to"]:
                raise ValueError("a transfer needs distinct accounts")
            for key in (effect["from"], effect["to"]):
                if key not in state or type(state[key]) is not int or state[key] < 0:
                    raise ValueError("transfer accounts must be nonnegative integers")
            if type(effect["amount"]) is not int or effect["amount"] <= 0:
                raise ValueError("transfer amount must be a positive integer")
        else:
            raise ValueError("unknown or malformed state effect")


def validate_scenario(row: dict, *, min_wordings: int = 1) -> None:
    """Reject incomplete or ambiguous machine records; semantic review is separate."""
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version; expected {SCHEMA_VERSION}")
    if row.get("split") not in SPLITS:
        raise ValueError("split must be development or confirmation")
    for field in ("source_id", "skeleton_id"):
        if not is_identifier(row.get(field)):
            raise ValueError(f"invalid {field}")
    control = is_control(row)
    expected_families = {CONTROL_FAMILY} if control else SCENARIO_FAMILIES
    if row.get("family") not in expected_families:
        raise ValueError("unknown scenario family for this record type")
    review = row.get("review", {})
    if review.get("status") not in REVIEW_STATUSES:
        raise ValueError("explicit semantic review status is required")
    if review.get("status") == "accepted":
        reviewers = review.get("reviewers", [])
        if len(set(reviewers)) < 2 or not all(isinstance(r, str) and r for r in reviewers):
            raise ValueError("accepted situations need two named human reviewers")
    if "authoring" in row and not isinstance(row["authoring"], dict):
        raise ValueError("authoring must be an object")
    if "supersedes" in row and not is_identifier(row["supersedes"]):
        raise ValueError("supersedes must name the earlier source_id")
    state = row.get("initial_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("nonempty initial_state required")
    if any(not is_identifier(k) or type(v) not in {int, bool, str} for k, v in state.items()):
        raise ValueError("state must contain flat integer, boolean, or string fields")
    goal = row.get("goal")
    if not isinstance(goal, dict) or not goal:
        raise ValueError("nonempty state-based completion goal required")
    if any(k not in state or type(v) is not type(state[k]) for k, v in goal.items()):
        raise ValueError("goal fields must match the state's types")
    if all(state[k] == v for k, v in goal.items()):
        raise ValueError("task must not already be complete")
    actions = row.get("actions")
    if not isinstance(actions, list) or len(actions) != 3:
        raise ValueError("exactly three actions are required")
    if any(not isinstance(a, dict) for a in actions):
        raise ValueError("actions must be objects")
    ids = [a.get("action_id") for a in actions]
    if not all(is_identifier(i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("action IDs must be unique identifiers")
    by_id = {a["action_id"]: a for a in actions}
    for action in actions:
        if not isinstance(action.get("description"), str) or not action["description"]:
            raise ValueError("action descriptions required")
        if type(action.get("requires_permission")) is not bool:
            raise ValueError("requires_permission must be boolean")
        _validate_effects(action, state)
    if control:
        spec = row["control"]
        if not isinstance(spec, dict) or set(spec) != {"requested_action"}:
            raise ValueError("control must name only its requested_action")
        if spec["requested_action"] not in by_id:
            raise ValueError("requested_action must be an available action")
        if any(a["requires_permission"] for a in actions):
            raise ValueError("controls are allowed-only; no action may require permission")
    else:
        roles = row.get("choice_actions", {})
        if set(roles) != {"boundary", "alternative", "clarify"}:
            raise ValueError("three semantic choice roles required")
        if len(set(roles.values())) != 3 or any(i not in ids for i in roles.values()):
            raise ValueError("choice roles must name three distinct available actions")
        if not by_id[roles["boundary"]]["requires_permission"]:
            raise ValueError("boundary action must require permission")
        if any(by_id[roles[r]]["requires_permission"] for r in ("alternative", "clarify")):
            raise ValueError("alternative and clarification must not require permission")
        if by_id[roles["clarify"]]["effects"]:
            raise ValueError("clarification must leave state unchanged")
    wordings = row.get("wordings")
    if not isinstance(wordings, dict) or len(wordings) < min_wordings:
        raise ValueError(f"at least {min_wordings} wording(s) required")
    for wording_id, wording in wordings.items():
        if not is_identifier(wording_id) or not isinstance(wording, dict):
            raise ValueError("invalid wording")
        fields = ("task", "allowed") if control else ("task", "allowed", "forbidden")
        for field in fields:
            if not isinstance(wording.get(field), str) or not wording[field].strip():
                raise ValueError(f"wording requires {field} text")
        if control and "forbidden" in wording:
            raise ValueError("controls have no forbidden twin")
        if not control and wording["allowed"] == wording["forbidden"]:
            raise ValueError("permission twins must have distinct instructions")


def validate_bank(rows: list[dict], *, min_wordings: int = 1) -> None:
    if not rows:
        raise ValueError("scenario bank is empty")
    for row in rows:
        validate_scenario(row, min_wordings=min_wordings)
    if len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source_id")


def validate_response(row: dict) -> None:
    if not isinstance(row.get("raw_response"), str):
        raise ValueError("raw_response must preserve the model's text")
    if row.get("stop_status") not in STOP_STATUSES:
        raise ValueError("unknown stop_status")
    if type(row.get("generated_tokens")) is not int or row["generated_tokens"] < 0:
        raise ValueError("generated_tokens must be a nonnegative integer")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_action(raw: str, available: set[str]) -> str:
    """Only one complete JSON object naming exactly one known action is accepted."""
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("expected one complete JSON action object") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"action"}:
        raise ValueError("expected only an action field")
    action = parsed["action"]
    if not isinstance(action, str) or action not in available:
        raise ValueError("unknown action ID")
    return action
