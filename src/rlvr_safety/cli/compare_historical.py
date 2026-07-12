from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import read_csv
from ..reproduction_analysis import compare_historical_layouts, render_reproduction_markdown


def _labeled_path(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from exc
    return label, Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial-scores", type=Path, required=True)
    parser.add_argument("--historical", type=_labeled_path, action="append", required=True)
    parser.add_argument("--original-model-key", type=Path, required=True)
    parser.add_argument("--title", default="Historical Layout Reproduction")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)
    packs = {label: list(read_csv(path)) for label, path in args.historical}
    if len(packs) != len(args.historical):
        raise SystemExit("historical pack labels must be unique")
    original_key = {
        row["model_alias"]: row["model"] for row in read_csv(args.original_model_key)
    }
    metrics = compare_historical_layouts(
        list(read_csv(args.factorial_scores)),
        packs,
        model_keys={"original": original_key},
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        render_reproduction_markdown(metrics, args.title), encoding="utf-8"
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote historical reproduction analysis to {args.out_md}")


if __name__ == "__main__":
    main()
