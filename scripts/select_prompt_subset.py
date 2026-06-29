#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_CATEGORIES = [
    "goal_persistence",
    "resource_acquisition",
    "self_preservation",
    "deception",
    "evaluation_awareness",
    "benign_control",
]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def parse_categories(raw: str):
    categories = [item.strip() for item in raw.split(",") if item.strip()]
    if not categories:
        raise SystemExit("--categories must include at least one category")
    return categories


def select_balanced_subset(rows, categories, per_category, seed):
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for row in rows:
        by_category[row.get("category")].append(row)

    selected = []
    for category in categories:
        candidates = list(by_category.get(category, []))
        if len(candidates) < per_category:
            raise SystemExit(
                f"category {category!r} has {len(candidates)} prompts, "
                f"but --per-category requested {per_category}"
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:per_category])

    selected.sort(key=lambda row: categories.index(row["category"]))
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Create a deterministic balanced prompt subset for a cheap first run."
    )
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts_seed.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated category list. Defaults to every seed category.",
    )
    args = parser.parse_args()

    if args.per_category < 1:
        raise SystemExit("--per-category must be at least 1")

    rows = load_jsonl(args.prompts)
    categories = parse_categories(args.categories)
    selected = select_balanced_subset(rows, categories, args.per_category, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f:
        for row in selected:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(selected)} prompts "
        f"({args.per_category} per category across {len(categories)} categories) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
