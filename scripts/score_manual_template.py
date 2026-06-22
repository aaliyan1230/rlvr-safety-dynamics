#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "id",
    "model",
    "category",
    "risk_dimension",
    "prompt",
    "response",
    "instrumental_score_0_2",
    "refusal_or_safe_redirect",
    "asks_permission",
    "discloses_uncertainty",
    "mentions_oversight",
    "verbosity_confounded",
    "thinking_trace_exposed",
    "notes",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.generations.open("r", encoding="utf-8") as f, args.out.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "model": row.get("model", ""),
                    "category": row.get("category", ""),
                    "risk_dimension": row.get("risk_dimension", ""),
                    "prompt": row.get("prompt", ""),
                    "response": row.get("response", ""),
                }
            )

    print(f"Wrote manual scoring template to {args.out}")


if __name__ == "__main__":
    main()
