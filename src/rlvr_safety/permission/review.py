"""Human semantic review of scenarios: queue sheets, a readable packet, and recorded decisions.

Nothing here decides anything. ``apply_reviews`` only copies decisions that named humans wrote in
their queue sheets into a *new* bank file: a record becomes ``accepted`` only when at least two
distinct reviewers accepted every wording, and any rejection rejects it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .prompts import build_conditions
from .schema import is_control, is_identifier, validate_bank

QUEUE_COLUMNS = (
    "source_id",
    "record_type",
    "skeleton_id",
    "wording_id",
    "reviewer",
    "permission_clear",
    "initial_state_matches",
    "legitimate_alternative",
    "effects_match_description",
    "formats_match_decision",
    "allowed_twin_plausible",
    "incentive_preserved",
    "decision",
    "rationale",
    "reviewed_on",
)
DECISIONS = {"pending", "accept", "revise", "reject"}
PACKET_PLAN = {
    "mcq": {"orders": "one"},
    "executable": {"orders": "one", "id_arms": ["semantic"]},
    "option_free": {"count": 1},
}


def export_queue(bank: list[dict], reviewers: list[str], out_dir: Path) -> dict[str, int]:
    """Write one blank sheet per reviewer so reviews stay independent of each other."""
    validate_bank(bank)
    if len(set(reviewers)) < 2 or not all(is_identifier(r) for r in reviewers):
        raise ValueError("at least two distinct reviewer identifiers are required")
    paths = {r: out_dir / f"queue-{r}.csv" for r in reviewers}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(f"already exists: {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for reviewer, path in paths.items():
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS)
            writer.writeheader()
            for record in bank:
                for wording_id in sorted(record["wordings"]):
                    writer.writerow(
                        {
                            "source_id": record["source_id"],
                            "record_type": "control" if is_control(record) else "scenario",
                            "skeleton_id": record["skeleton_id"],
                            "wording_id": wording_id,
                            "reviewer": reviewer,
                            "decision": "pending",
                        }
                    )
        counts[reviewer] = sum(len(r["wordings"]) for r in bank)
    return counts


def render_packet(bank: list[dict]) -> str:
    """Markdown showing every record the way the model will see it."""
    lines = [
        "# Human review packet",
        "",
        "Review each record against the checklist in your queue sheet. Mechanical checks already",
        "passed; they do not replace your judgment. Decisions: accept / revise / reject.",
        "",
    ]
    for record in sorted(bank, key=lambda r: r["source_id"]):
        kind = "control" if is_control(record) else "scenario"
        lines += [
            f"## {record['source_id']} ({kind}, {record['family']})",
            "",
            f"Skeleton `{record['skeleton_id']}` · review status `{record['review']['status']}`"
            + (f" · revises `{record['supersedes']}`" if record.get("supersedes") else ""),
            "",
            "**Initial state:** `" + json.dumps(record["initial_state"], sort_keys=True) + "`  ",
            "**Goal:** `" + json.dumps(record["goal"], sort_keys=True) + "`",
            "",
            "| Action | Needs permission | Description | Effects |",
            "|---|---|---|---|",
        ]
        for action in record["actions"]:
            effects = json.dumps(action["effects"]) if action["effects"] else "none"
            lines.append(
                f"| `{action['action_id']}` | {action['requires_permission']} | "
                f"{action['description']} | `{effects}` |"
            )
        lines.append("")
        for wording_id, wording in sorted(record["wordings"].items()):
            lines += [f"### Wording `{wording_id}`", "", f"**Task:** {wording['task']}", ""]
            lines.append(f"**Allowed:** {wording['allowed']}")
            if "forbidden" in wording:
                lines.append(f"**Forbidden:** {wording['forbidden']}")
            lines.append("")
        lines += ["### Prompts as the model sees them (first wording, one option order)", ""]
        one = deepcopy(record)
        first = sorted(one["wordings"])[0]
        plan = dict(PACKET_PLAN, wordings=[first])
        if is_control(record):
            plan = {
                "executable": {"orders": "one", "id_arms": ["semantic", "neutral"]},
                "wordings": [first],
            }
        for condition in build_conditions([one], plan):
            title = f"{condition['permission']} · {condition['format']}"
            if condition["id_arm"] == "neutral":
                title += " · neutral IDs"
            lines += [f"#### {title}", "", "```text", condition["prompt"], "```", ""]
    return "\n".join(lines) + "\n"


def _read_queue(path: Path) -> tuple[list[dict], str]:
    data = path.read_bytes()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and set(rows[0]) != set(QUEUE_COLUMNS):
        raise ValueError(f"{path} does not have the expected queue columns")
    return rows, hashlib.sha256(data).hexdigest()


def apply_reviews(bank: list[dict], queue_paths: list[Path]) -> tuple[list[dict], dict]:
    """Return a copy of ``bank`` with review statuses set from the queue sheets, plus an audit."""
    validate_bank(bank)
    expected = {(r["source_id"], w) for r in bank for w in r["wordings"]}
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    audit: dict = {"queues": {}, "records": {}}
    for path in queue_paths:
        rows, sha = _read_queue(path)
        audit["queues"][path.name] = sha
        for row in rows:
            key = (row["source_id"], row["wording_id"])
            if key not in expected:
                raise ValueError(f"queue row for unknown record/wording: {key}")
            decision = (row["decision"] or "").strip()
            if decision not in DECISIONS:
                raise ValueError(f"unknown decision {decision!r} for {key}")
            reviewer = row["reviewer"].strip()
            if not is_identifier(reviewer):
                raise ValueError(f"invalid reviewer {reviewer!r}")
            if reviewer in decisions.setdefault(key, {}):
                raise ValueError(f"duplicate decision by {reviewer} for {key}")
            decisions[key][reviewer] = decision
    out = []
    for record in bank:
        updated = deepcopy(record)
        per_wording = [decisions.get((record["source_id"], w), {}) for w in record["wordings"]]
        reviewers = sorted({r for d in per_wording for r in d})
        values = [v for d in per_wording for v in d.values()]
        if "reject" in values:
            status = "rejected"
        elif (
            len(reviewers) >= 2
            and all(set(d) == set(reviewers) for d in per_wording)
            and all(v == "accept" for v in values)
        ):
            status = "accepted"
        else:
            status = "pending"
        updated["review"] = {
            "status": status,
            "reviewers": reviewers if status != "pending" else [],
        }
        audit["records"][record["source_id"]] = {"status": status, "reviewers": reviewers}
        out.append(updated)
    validate_bank(out)
    return out, audit
