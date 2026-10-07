"""Versioned records for the development permission-decision assay."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = "permission-development-v1"
STOP_STATUSES = {"complete", "length", "timeout", "interrupted", "error"}
FAMILIES = {"resource_allocation", "operator_control", "reporting_oversight"}


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]+", value) is not None


def validate_scenario(row: dict) -> None:
    """Reject incomplete or ambiguous machine records; semantic review is separate."""
    if row.get("schema_version") != SCHEMA_VERSION or row.get("split") != "development":
        raise ValueError("only versioned development scenarios are supported")
    for field in ("source_id", "skeleton_id"):
        if not _identifier(row.get(field)):
            raise ValueError(f"invalid {field}")
    if row.get("family") not in FAMILIES:
        raise ValueError("unknown scenario family")
    review = row.get("review", {})
    if review.get("status") not in {"pending", "accepted", "rejected"}:
        raise ValueError("explicit semantic review status is required")
    if review.get("status") == "accepted":
        reviewers = review.get("reviewers", [])
        if len(set(reviewers)) < 2 or not all(isinstance(r, str) and r for r in reviewers):
            raise ValueError("accepted situations need two named human reviewers")
    state = row.get("initial_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("nonempty initial_state required")
    if any(not _identifier(k) or type(v) not in {int, bool, str} for k, v in state.items()):
        raise ValueError("state must contain flat integer, boolean, or string fields")
    goal = row.get("goal")
    if not isinstance(goal, dict) or not goal:
        raise ValueError("nonempty state-based completion goal required")
    if any(k not in state or type(v) is not type(state[k]) for k, v in goal.items()):
        raise ValueError("goal fields must match the state's types")
    if all(state[k] == v for k, v in goal.items()):
        raise ValueError("task must not already be complete")
    actions = row.get("actions")
    if not isinstance(actions, list) or len(actions) < 3:
        raise ValueError("boundary, alternative, and clarification actions required")
    if any(not isinstance(a, dict) for a in actions):
        raise ValueError("actions must be objects")
    ids = [a.get("action_id") for a in actions]
    if not all(_identifier(i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("action IDs must be unique identifiers")
    for action in actions:
        if not isinstance(action.get("description"), str) or not action["description"]:
            raise ValueError("action descriptions required")
        if type(action.get("requires_permission")) is not bool:
            raise ValueError("requires_permission must be boolean")
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
    roles = row.get("choice_actions", {})
    if set(roles) != {"boundary", "alternative", "clarify"}:
        raise ValueError("three semantic choice roles required")
    if len(set(roles.values())) != 3 or any(i not in ids for i in roles.values()):
        raise ValueError("choice roles must name three distinct available actions")
    by_id = {a["action_id"]: a for a in actions}
    if not by_id[roles["boundary"]]["requires_permission"]:
        raise ValueError("boundary action must require permission")
    if any(by_id[roles[r]]["requires_permission"] for r in ("alternative", "clarify")):
        raise ValueError("alternative and clarification must not require permission")
    if by_id[roles["clarify"]]["effects"]:
        raise ValueError("clarification must leave state unchanged")
    wordings = row.get("wordings")
    if not isinstance(wordings, dict) or not wordings:
        raise ValueError("at least one wording required")
    for wording_id, wording in wordings.items():
        if not _identifier(wording_id) or not isinstance(wording, dict):
            raise ValueError("invalid wording")
        for field in ("task", "allowed", "forbidden"):
            if not isinstance(wording.get(field), str) or not wording[field].strip():
                raise ValueError(f"wording requires {field} text")
        if wording["allowed"] == wording["forbidden"]:
            raise ValueError("permission twins must have distinct instructions")


def validate_bank(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("scenario bank is empty")
    for row in rows:
        validate_scenario(row)
    if len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source_id")


def validate_response(row: dict) -> None:
    if not isinstance(row.get("raw_response"), str):
        raise ValueError("raw_response must preserve the model's text")
    if row.get("stop_status") not in STOP_STATUSES:
        raise ValueError("unknown stop_status")
    if type(row.get("generated_tokens")) is not int or row["generated_tokens"] < 0:
        raise ValueError("generated_tokens must be a nonnegative integer")


def parse_actions(raw: str, available: set[str]) -> list[str]:
    """Only a complete JSON object can request at most two simulation actions."""

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("expected one complete JSON action object") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"actions"}:
        raise ValueError("expected only an actions field")
    actions = parsed["actions"]
    if not isinstance(actions, list) or len(actions) > 2:
        raise ValueError("actions must be a list of zero to two action IDs")
    if any(not isinstance(a, str) or a not in available for a in actions):
        raise ValueError("unknown action ID")
    if len(set(actions)) != len(actions):
        raise ValueError("each action may be requested at most once")
    return actions
