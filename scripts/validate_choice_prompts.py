#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


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
            labels = [item.get("label") for item in row["options"]]
            if labels != ["A", "B", "C"]:
                raise SystemExit(f"{args.prompts}:{line_no}: options must be labeled A, B, C")
            scores = sorted(item.get("instrumental_score_0_2") for item in row["options"])
            if scores != [0, 1, 2]:
                raise SystemExit(f"{args.prompts}:{line_no}: options must contain scores 0, 1, 2")
            if row["score_key"] != {item["label"]: item["instrumental_score_0_2"] for item in row["options"]}:
                raise SystemExit(f"{args.prompts}:{line_no}: score_key does not match options")
            if "CHOICE:" not in row["prompt"] or "REASON:" not in row["prompt"]:
                raise SystemExit(f"{args.prompts}:{line_no}: prompt missing output-format instruction")
            rows.append(row)

    ids = [row["id"] for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate choice prompt ids: {duplicates}")

    categories = Counter(row["category"] for row in rows)
    print(f"Loaded {len(rows)} choice prompts from {args.prompts}")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    if len(rows) != 24:
        raise SystemExit("targeted choice eval should contain exactly 24 prompts")
    expected = {"resource_acquisition", "self_preservation", "evaluation_awareness"}
    if set(categories) != expected:
        raise SystemExit(f"unexpected categories: {sorted(categories)}")


if __name__ == "__main__":
    main()
