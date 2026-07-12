"""CLI for paired comparison of two completed factorial result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import read_csv
from ..protocol_comparison import (
    ProtocolComparisonError,
    compare_factorial_protocols,
    render_protocol_comparison_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare matched cells from two complete factorial experiment runs."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Result directory containing choice_scores.csv.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        required=True,
        help="Result directory containing choice_scores.csv.",
    )
    parser.add_argument("--reference-label")
    parser.add_argument("--comparison-label")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--title", default="Factorial Protocol Comparison")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser


def _score_path(result_dir: Path) -> Path:
    if not result_dir.is_dir():
        raise ProtocolComparisonError(f"not a result directory: {result_dir}")
    score_path = result_dir / "choice_scores.csv"
    if not score_path.is_file():
        raise ProtocolComparisonError(f"missing factorial scores: {score_path}")
    return score_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reference_label = args.reference_label or args.reference_dir.name
    comparison_label = args.comparison_label or args.comparison_dir.name
    metrics = compare_factorial_protocols(
        list(read_csv(_score_path(args.reference_dir))),
        list(read_csv(_score_path(args.comparison_dir))),
        reference_label=reference_label,
        comparison_label=comparison_label,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote protocol comparison metrics to {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            render_protocol_comparison_markdown(metrics, args.title),
            encoding="utf-8",
        )
        print(f"Wrote protocol comparison report to {args.out_md}")


if __name__ == "__main__":
    main()
