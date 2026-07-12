"""CLI for the reciprocal historical inference-runtime crossover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import read_csv
from ..runtime_crossover import (
    RuntimeCrossoverError,
    analyze_runtime_crossover,
    render_runtime_crossover_markdown,
)


def _score_path(directory: Path) -> Path:
    if not directory.is_dir():
        raise RuntimeCrossoverError(f"not a result directory: {directory}")
    summary_path = directory / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("had_error") is not False:
            raise RuntimeCrossoverError(f"run summary does not pass: {summary_path}")
    path = directory / "choice_scores.csv"
    if not path.is_file():
        raise RuntimeCrossoverError(f"missing scores: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--paraphrase-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--title", default="Historical Runtime Crossover")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = analyze_runtime_crossover(
        list(read_csv(_score_path(args.stage_dir))),
        list(read_csv(_score_path(args.paraphrase_dir))),
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote runtime crossover metrics to {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            render_runtime_crossover_markdown(metrics, args.title), encoding="utf-8"
        )
        print(f"Wrote runtime crossover report to {args.out_md}")


if __name__ == "__main__":
    main()
