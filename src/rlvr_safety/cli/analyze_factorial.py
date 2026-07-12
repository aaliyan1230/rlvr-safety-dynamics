"""CLI for strict counterbalanced factorial analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..factorial_analysis import analyze_factorial, render_factorial_markdown
from ..io import read_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--protocol-note")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = analyze_factorial(
        list(read_csv(args.scores)),
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        allow_incomplete=args.allow_incomplete,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        render_factorial_markdown(metrics, args.protocol_note), encoding="utf-8"
    )
    print(f"Wrote factorial analysis to {args.out_md}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote machine-readable metrics to {args.out_json}")


if __name__ == "__main__":
    main()
