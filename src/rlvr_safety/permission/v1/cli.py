"""Prepare and replay private development records without model downloads or GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr_safety.io import read_jsonl, write_jsonl
from rlvr_safety.permission.v1.benchmark import (
    FixtureProvider,
    prepare_requests,
    run_benchmark,
    scripted_responses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="export prompts and engineering fixtures")
    prepare.add_argument("--scenarios", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--stress", action="store_true")
    replay = subparsers.add_parser("replay", help="score saved responses, resuming missing rows")
    replay.add_argument("--scenarios", type=Path, required=True)
    replay.add_argument("--requests", type=Path, required=True)
    replay.add_argument("--responses", type=Path, required=True)
    replay.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    scenarios = list(read_jsonl(args.scenarios))
    if args.command == "prepare":
        targets = [args.out_dir / "requests.jsonl", args.out_dir / "fixture_responses.jsonl"]
        if any(p.exists() for p in targets):
            parser.error("prepared files already exist; use a new directory")
        requests = prepare_requests(scenarios, stress=args.stress)
        write_jsonl(targets[0], requests)
        write_jsonl(targets[1], scripted_responses(scenarios, requests))
        print(
            json.dumps(
                {
                    "sources": len(scenarios),
                    "episodes": len(requests),
                    "human_review": "see each scenario's review status",
                },
                indent=2,
            )
        )
    else:
        requests = list(read_jsonl(args.requests))
        provider = FixtureProvider(args.responses)
        expected = {(r["condition_id"], r["sample_id"]) for r in requests}
        if set(provider.responses) != expected:
            parser.error("response IDs must exactly match the requested episode set")
        print(json.dumps(run_benchmark(scenarios, requests, provider, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
