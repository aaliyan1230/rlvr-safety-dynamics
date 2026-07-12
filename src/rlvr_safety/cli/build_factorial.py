"""CLI for constructing a fully counterbalanced prompt pack."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..factorial import build_factorial_rows, validate_factorial_balance
from ..io import read_jsonl, sha256_file, write_jsonl


def parse_wording(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from exc
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross original and paraphrased prompts with all six option orders."
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument(
        "--wording",
        type=parse_wording,
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Paraphrase pack; repeat for each independently validated wording seed.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.wording:
        raise SystemExit("at least one --wording LABEL=PATH is required")
    wording_paths = dict(args.wording)
    if len(wording_paths) != len(args.wording):
        raise SystemExit("wording labels must be unique")

    rows = build_factorial_rows(
        list(read_jsonl(args.original)),
        {label: list(read_jsonl(path)) for label, path in wording_paths.items()},
    )
    write_jsonl(args.out, rows)
    balance = validate_factorial_balance(rows)

    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    inputs = {"original": args.original, **wording_paths}
    manifest = {
        "schema_version": 1,
        "design": "source × wording × all 3! semantic-option orders",
        "output": str(args.out),
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in inputs.items()
        },
        "rows": balance["rows"],
        "sources": balance["sources"],
        "wordings": balance["wordings"],
        "orders": balance["orders"],
        "rows_per_wording": dict(balance["rows_per_wording"]),
        "risky_position_counts": dict(balance["risky_position_counts"]),
        "category_counts": dict(Counter(str(row["category"]) for row in rows)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(rows)} conditions ({balance['sources']} sources × "
        f"{len(balance['wordings'])} wordings × 6 orders) to {args.out}"
    )
    print(f"Wrote design manifest to {manifest_path}")


if __name__ == "__main__":
    main()
