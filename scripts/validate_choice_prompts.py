#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlvr_safety.choice import build_choice_prompt, validate_choice_row


REQUIRED_FIELDS = {
    "id",
    "source_id",
    "category",
    "paraphrase_group",
    "variant",
    "risk_dimension",
    "prompt",
    "base_prompt",
    "options",
    "score_key",
    "benign_control",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/choice_eval_targeted.jsonl"))
    parser.add_argument("--expected-count", type=int, default=24)
    parser.add_argument("--allow-subset-categories", action="store_true")
    args = parser.parse_args()

    rows = []
    with args.prompts.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{args.prompts}:{line_no}: invalid JSON: {exc}") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise SystemExit(f"{args.prompts}:{line_no}: missing fields: {sorted(missing)}")
            try:
                validate_choice_row(row)
            except ValueError as exc:
                raise SystemExit(f"{args.prompts}:{line_no}: {exc}") from exc
            rendered = build_choice_prompt(row["base_prompt"], row["options"])
            if row["prompt"] != rendered:
                raise SystemExit(
                    f"{args.prompts}:{line_no}: rendered prompt disagrees with base_prompt/options"
                )
            rows.append(row)

    ids = [row["id"] for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate choice prompt ids: {duplicates}")

    categories = Counter(row["category"] for row in rows)
    print(f"Loaded {len(rows)} choice prompts from {args.prompts}")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    if args.expected_count >= 0 and len(rows) != args.expected_count:
        raise SystemExit(f"choice eval should contain exactly {args.expected_count} prompts")
    expected = {"resource_acquisition", "self_preservation", "evaluation_awareness"}
    if args.allow_subset_categories:
        if not set(categories).issubset(expected):
            raise SystemExit(f"unexpected categories: {sorted(categories)}")
    elif set(categories) != expected:
        raise SystemExit(f"unexpected categories: {sorted(categories)}")


if __name__ == "__main__":
    main()
