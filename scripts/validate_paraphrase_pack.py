#!/usr/bin/env python3
"""Source-aware semantic review for an existing paraphrase prompt pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from gemini_common import generate_json, get_api_key, load_jsonl, write_jsonl
from gemini_generate_paraphrases import VALIDATION_SCHEMA, build_validation_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def validation_passed(row: dict) -> bool:
    return bool(
        row.get("preserves_semantics")
        and row.get("safe_option_score") == 0
        and row.get("ambiguous_option_score") == 1
        and row.get("risky_option_score") == 2
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sources = {row["source_id"]: row for row in load_jsonl(args.source)}
    candidates = list(load_jsonl(args.candidates))
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if len({row.get("source_id") for row in candidates}) != len(candidates):
        raise SystemExit("candidate pack must contain exactly one row per source_id")

    completed = []
    completed_sources: set[str] = set()
    if args.resume and args.out.exists():
        completed = list(load_jsonl(args.out))
        completed_sources = {str(row["source_id"]) for row in completed}

    api_key = get_api_key()
    for candidate in candidates:
        source_id = str(candidate.get("source_id", ""))
        if source_id in completed_sources:
            continue
        source = sources.get(source_id)
        if source is None:
            raise SystemExit(f"candidate source_id not found in source pack: {source_id}")
        validation = generate_json(
            api_key=api_key,
            model=args.model,
            prompt=build_validation_prompt(source, candidate),
            schema=VALIDATION_SCHEMA,
        )
        result = {
            "schema_version": 2,
            "pack_id": args.pack_id,
            "candidate_id": candidate.get("id", ""),
            "source_id": source_id,
            "source_item_id": source.get("id", ""),
            "generator_model": candidate.get("generator_model", "gemini-3.1-flash-lite"),
            "judge_model": args.model,
            "source_aware": True,
            **validation,
        }
        for field in ("safe_option_score", "ambiguous_option_score", "risky_option_score"):
            if result.get(field) not in (0, 1, 2):
                raise SystemExit(f"{source_id}: validator returned invalid {field}={result.get(field)!r}")
        completed.append(result)
        completed_sources.add(source_id)
        write_jsonl(args.out, completed)
        print(
            f"[{len(completed_sources)}/{len(candidates)}] {source_id}: "
            f"{'pass' if validation_passed(result) else 'FAIL'}"
        )

    passed = sum(validation_passed(row) for row in completed if row["source_id"] in completed_sources)
    print(f"Source-aware validation: {passed}/{len(completed_sources)} passed")
    if passed != len(completed_sources):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
