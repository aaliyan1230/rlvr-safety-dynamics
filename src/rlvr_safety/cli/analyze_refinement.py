"""Analyze the two predeclared dense Tülu refinement intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import read_csv
from ..refinement_analysis import analyze_refinement


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for path in args.scores:
        rows.extend(read_csv(path))
    result = analyze_refinement(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote refinement analysis to {args.out}")


if __name__ == "__main__":
    main()
