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


def load_rows(paths):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def build_model_aliases(rows):
    models = sorted({row.get("model", "") for row in rows})
    return {model: f"model_{chr(ord('a') + idx)}" for idx, model in enumerate(models)}


def write_blind_key(path, aliases):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_alias", "model"])
        writer.writeheader()
        for model, alias in sorted(aliases.items(), key=lambda item: item[1]):
            writer.writerow({"model_alias": alias, "model": model})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--blind-models",
        action="store_true",
        help="Replace model names with stable aliases in the scoring sheet.",
    )
    parser.add_argument(
        "--blind-key-out",
        type=Path,
        help="Optional CSV path for the model alias key when --blind-models is used.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = list(load_rows(args.generations))
    model_aliases = build_model_aliases(rows) if args.blind_models else {}
    if args.blind_key_out:
        if not args.blind_models:
            raise SystemExit("--blind-key-out requires --blind-models")
        write_blind_key(args.blind_key_out, model_aliases)

    with args.out.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            model = row.get("model", "")
            if args.blind_models:
                model = model_aliases.get(model, "")
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "model": model,
                    "category": row.get("category", ""),
                    "risk_dimension": row.get("risk_dimension", ""),
                    "prompt": row.get("prompt", ""),
                    "response": row.get("response", ""),
                }
            )

    print(f"Wrote manual scoring template to {args.out}")
    if args.blind_key_out:
        print(f"Wrote blind model key to {args.blind_key_out}")


if __name__ == "__main__":
    main()
