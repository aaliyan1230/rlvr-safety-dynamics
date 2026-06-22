#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "category",
    "paraphrase_group",
    "variant",
    "risk_dimension",
    "prompt",
    "benign_control",
}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise SystemExit(f"{path}:{line_no}: missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts_seed.jsonl"))
    args = parser.parse_args()

    rows = load_jsonl(args.prompts)
    ids = [row["id"] for row in rows]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise SystemExit(f"duplicate prompt ids: {duplicate_ids}")

    categories = Counter(row["category"] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["paraphrase_group"]].append(row["id"])

    print(f"Loaded {len(rows)} prompts from {args.prompts}")
    print("Category counts:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    print(f"Paraphrase groups: {len(groups)}")

    non_control = sum(not row["benign_control"] for row in rows)
    controls = sum(row["benign_control"] for row in rows)
    print(f"Non-control prompts: {non_control}")
    print(f"Control prompts: {controls}")

    if len(rows) < 48:
        raise SystemExit("prompt set is below the planned first-run minimum of 48")

    expected = {
        "goal_persistence",
        "resource_acquisition",
        "self_preservation",
        "deception",
        "evaluation_awareness",
        "benign_control",
    }
    missing_categories = expected - categories.keys()
    if missing_categories:
        raise SystemExit(f"missing categories: {sorted(missing_categories)}")


if __name__ == "__main__":
    main()
