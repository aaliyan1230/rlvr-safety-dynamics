from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..baseline_analysis import analyze_baseline, render_baseline_markdown
from ..io import read_csv


def _model_key(path: Path) -> dict[str, str]:
    return {row["model_alias"]: row["model"] for row in read_csv(path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeform", type=Path, required=True)
    parser.add_argument("--freeform-key", type=Path, required=True)
    parser.add_argument("--choice", type=Path, required=True)
    parser.add_argument("--choice-key", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)
    metrics = analyze_baseline(
        list(read_csv(args.freeform)),
        _model_key(args.freeform_key),
        list(read_csv(args.choice)),
        _model_key(args.choice_key),
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_baseline_markdown(metrics), encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote baseline analysis to {args.out_md}")


if __name__ == "__main__":
    main()
