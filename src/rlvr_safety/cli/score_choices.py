"""CLI for deterministic structured-choice scoring."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..choice import SCORE_FIELDS, choice_summary_markdown, score_generation
from ..io import read_jsonl, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--blind-models", action="store_true")
    parser.add_argument("--blind-key-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = [row for path in args.generations for row in read_jsonl(path)]
    if not rows:
        raise SystemExit("no generation rows found")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("id", "")), str(row.get("model", "")))
        if not all(key):
            raise SystemExit(f"generation row is missing id/model: {row!r}")
        if key in seen:
            raise SystemExit(f"duplicate generation cell: {key}")
        seen.add(key)

    models = sorted({str(row["model"]) for row in rows})
    aliases = {model: f"model_{chr(ord('a') + index)}" for index, model in enumerate(models)}
    if args.blind_key_out and not args.blind_models:
        raise SystemExit("--blind-key-out requires --blind-models")

    scored = [
        score_generation(
            row,
            max_new_tokens=args.max_new_tokens,
            model_name=aliases[str(row["model"])] if args.blind_models else None,
        )
        for row in rows
    ]
    write_csv(args.out, scored, SCORE_FIELDS)

    if args.blind_key_out:
        args.blind_key_out.parent.mkdir(parents=True, exist_ok=True)
        with args.blind_key_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model_alias", "model"])
            writer.writeheader()
            for model, alias in aliases.items():
                writer.writerow({"model_alias": alias, "model": model})

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(choice_summary_markdown(scored), encoding="utf-8")

    malformed = sum(row["malformed"] == "true" for row in scored)
    capped = sum(row["verbosity_confounded"] == "true" for row in scored)
    print(f"Wrote {len(scored)} scored rows to {args.out} ({malformed} malformed, {capped} capped)")
    if args.summary_out:
        print(f"Wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
