#!/usr/bin/env python3
"""Build the disclosed AI semantic-audit rows and report deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PACKS = {
    "p1": "data/gemini_choice_paraphrases_v1.jsonl",
    "p2": "data/gemini_choice_paraphrases_v2.jsonl",
    "p3": "data/gemini_choice_paraphrases_v3.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_prompt": row["base_prompt"],
        "options": sorted(
            (
                {
                    "text": option["text"],
                    "instrumental_score_0_2": option["instrumental_score_0_2"],
                }
                for option in row["options"]
            ),
            key=lambda option: option["instrumental_score_0_2"],
        ),
    }


def build_rows(repo: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {row["source_id"]: row for row in load_jsonl(repo / "data/choice_eval_targeted.jsonl")}
    caveats = {
        (row["source_id"], row["wording_id"]): row["rationale"]
        for row in config["caveats"]
    }
    rows: list[dict[str, Any]] = []
    for wording_id, relative in PACKS.items():
        candidates = load_jsonl(repo / relative)
        if len(candidates) != len(sources):
            raise ValueError(f"{wording_id}: expected {len(sources)} candidates, got {len(candidates)}")
        if len({row["source_id"] for row in candidates}) != len(candidates):
            raise ValueError(f"{wording_id}: duplicate source_id")
        for candidate in candidates:
            source_id = candidate["source_id"]
            source = sources.get(source_id)
            if source is None:
                raise ValueError(f"{wording_id}: unknown source_id {source_id}")
            source_scores = sorted(option["instrumental_score_0_2"] for option in source["options"])
            candidate_scores = sorted(option["instrumental_score_0_2"] for option in candidate["options"])
            if source_scores != [0, 1, 2] or candidate_scores != [0, 1, 2]:
                raise ValueError(f"{source_id}/{wording_id}: invalid ordinal score coverage")
            key = (source_id, wording_id)
            decision = "pass_with_caveat" if key in caveats else config["default_decision"]
            rows.append(
                {
                    "schema_version": 1,
                    "audit_id": config["audit_id"],
                    "source_id": source_id,
                    "candidate_id": candidate["id"],
                    "wording_id": wording_id,
                    "category": source["category"],
                    "risk_dimension": source["risk_dimension"],
                    "decision": decision,
                    "accepted": decision in {"pass", "pass_with_caveat"},
                    "rationale": caveats.get(key, config["default_rationale"]),
                    "source_semantic_sha256": object_sha256(semantic_payload(source)),
                    "candidate_semantic_sha256": object_sha256(semantic_payload(candidate)),
                    "reviewer_type": config["reviewer"]["type"],
                    "independent_human_review": False,
                    "model_outputs_hidden": config["blinding"]["model_outputs_hidden"],
                }
            )
    return sorted(rows, key=lambda row: (row["source_id"], row["wording_id"]))


def render_report(config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["decision"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    lines = [
        "# Disclosed AI semantic audit v1",
        "",
        "**Result under the project owner's explicit assumption: ACCEPT. Independent human validation: NOT PERFORMED.**",
        "",
        config["assumption"],
        "",
        f"The AI reviewer accepted all {len(rows)} source/candidate pairs: "
        f"{counts['pass']} passed without a noted concern and "
        f"{counts['pass_with_caveat']} passed with a construct-fidelity caveat. "
        "No pair failed the ordinal semantic-preservation rule.",
        "",
        "The review was blinded to model outputs and row-level outcome scores, but the reviewer knew the project's aggregate conclusions. It is therefore prompt-only review, not full study blinding or reviewer independence.",
        "",
        "| Category | Reviewed pairs |",
        "|---|---:|",
    ]
    for category in sorted(category_counts):
        lines.append(f"| {category} | {category_counts[category]} |")
    lines.extend(
        [
            "",
            "## Accepted pairs with caveats",
            "",
            "| Source | Wording | Rationale |",
            "|---|---|---|",
        ]
    )
    for row in rows:
        if row["decision"] == "pass_with_caveat":
            lines.append(f"| `{row['source_id']}` | `{row['wording_id']}` | {row['rationale']} |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            config["claim_boundary"],
            "The accepted caveats must remain visible when interpreting wording effects because semantic acceptability does not imply equal elicitation difficulty.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/audits/ai_semantic_audit_v1.json"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/ai_semantic_audit_v1.jsonl"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/ai_semantic_audit_v1.md"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / args.config).read_text(encoding="utf-8"))
    rows = build_rows(repo, config)
    counts = Counter(row["decision"] for row in rows)
    expected = config["expected"]
    observed = {
        "pairs": len(rows),
        "pass": counts["pass"],
        "pass_with_caveat": counts["pass_with_caveat"],
        "fail": counts["fail"],
    }
    for field, value in observed.items():
        if value != expected[field]:
            raise ValueError(f"expected {field}={expected[field]}, observed {value}")
    out_jsonl = repo / args.out_jsonl
    out_md = repo / args.out_md
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    out_md.write_text(render_report(config, rows), encoding="utf-8")
    print(
        f"Wrote {len(rows)} AI semantic judgments "
        f"({counts['pass']} pass, {counts['pass_with_caveat']} pass with caveat, {counts['fail']} fail)"
    )


if __name__ == "__main__":
    main()
