"""Objectively score frozen capability-anchor generations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..anchors import score_capability_response, validate_capability_pack
from ..io import read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--pack", type=Path, default=Path("data/tulu_capability_anchor_v1.jsonl")
    )
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    items = validate_capability_pack(read_jsonl(args.pack))
    by_id = {row["id"]: row for row in items}
    generations = list(read_jsonl(args.generations))
    seen: set[tuple[str, str, str]] = set()
    scored = []
    for row in generations:
        item_id = str(row.get("id", ""))
        if item_id not in by_id:
            raise ValueError(f"unknown capability id: {item_id!r}")
        checkpoint = str(row.get("checkpoint", row.get("step", "")))
        model = str(row.get("model", ""))
        key = (checkpoint, model, item_id)
        if key in seen:
            raise ValueError(f"duplicate capability generation: {key}")
        seen.add(key)
        token_capped = int(row.get("generated_tokens", 0)) >= int(
            row.get("row_max_new_tokens", 2**31)
        )
        result = score_capability_response(by_id[item_id], str(row.get("response", "")))
        if token_capped:
            result = {
                **result,
                "correct": False,
                "parse_status": "token_capped",
                "parsed_answer": None,
            }
        scored.append({**row, **result})
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored), encoding="utf-8"
    )
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored:
        checkpoint = str(row.get("checkpoint", row.get("step", "")))
        groups[(checkpoint, str(row.get("model", "")))].append(row)
    summaries = []
    for (checkpoint, model), rows in sorted(groups.items()):
        family = defaultdict(list)
        for row in rows:
            family[row["family"]].append(row)
        summaries.append(
            {
                "checkpoint": checkpoint,
                "model": model,
                "rows": len(rows),
                "correct": sum(row["correct"] for row in rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "parse_status": dict(Counter(row["parse_status"] for row in rows)),
                "family_accuracy": {
                    name: sum(row["correct"] for row in family_rows) / len(family_rows)
                    for name, family_rows in sorted(family.items())
                },
            }
        )
    args.out_summary.write_text(
        json.dumps({"schema_version": 1, "groups": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Scored {len(scored)} capability generations across {len(summaries)} groups")


if __name__ == "__main__":
    main()
