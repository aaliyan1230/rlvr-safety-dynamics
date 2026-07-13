"""Analyze frozen Tülu capability/free-form panels and apply the final claim gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cross_format_analysis import analyze_capability, analyze_freeform, apply_claim_gate
from ..io import read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability-scores", type=Path, required=True)
    parser.add_argument("--freeform-scores", type=Path, required=True)
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    capability = analyze_capability(read_jsonl(args.capability_scores))
    freeform = analyze_freeform(read_jsonl(args.freeform_scores))
    agreement = json.loads(args.agreement.read_text())
    structured = json.loads(args.structured.read_text())
    claim_gate = apply_claim_gate(structured, capability, freeform, agreement)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capability": capability,
                "freeform": freeform,
                "agreement": agreement,
                "claim_gate": claim_gate,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote cross-format analysis to {args.out}; label={claim_gate['label']}")


if __name__ == "__main__":
    main()
