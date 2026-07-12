"""CLI for applying the project's final free-form labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..adjudication import apply_adjudications
from ..io import read_csv, read_jsonl, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--model-key", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    score_rows = list(read_csv(args.scores))
    if not score_rows:
        raise SystemExit("manual score input is empty")
    key_rows = list(read_csv(args.model_key))
    model_to_alias = {row["model"]: row["model_alias"] for row in key_rows}
    rows = apply_adjudications(
        score_rows,
        list(read_jsonl(args.adjudications)),
        model_to_alias=model_to_alias,
    )
    with args.scores.open("r", encoding="utf-8", newline="") as handle:
        original_fields = list(csv.DictReader(handle).fieldnames or [])
    fields = [*original_fields, "pre_adjudication_score", "adjudicated", "adjudication_reason"]
    write_csv(args.out, rows, fields)
    changed = sum(row["adjudicated"] == "true" for row in rows)
    print(f"Wrote {len(rows)} rows with {changed} adjudications to {args.out}")


if __name__ == "__main__":
    main()
