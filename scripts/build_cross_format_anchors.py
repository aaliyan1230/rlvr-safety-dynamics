#!/usr/bin/env python3
"""Build and checksum the frozen Tülu free-form and capability anchors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlvr_safety.anchors import build_freeform_anchors, validate_capability_pack  # noqa: E402
from rlvr_safety.io import read_jsonl, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/choice_eval_targeted.jsonl"))
    parser.add_argument("--capability", type=Path, default=Path("data/tulu_capability_anchor_v1.jsonl"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/anchors/tulu_cross_format_anchors_v1.json"))
    parser.add_argument("--scoring-protocol", type=Path, default=Path("data/tulu_freeform_scoring_protocol_v1.md"))
    parser.add_argument("--out-freeform", type=Path, default=Path("data/tulu_freeform_anchor_v1.jsonl"))
    parser.add_argument("--out-manifest", type=Path, default=Path("data/tulu_cross_format_anchors_v1.manifest.json"))
    args = parser.parse_args()
    source = REPO / args.source
    capability = REPO / args.capability
    protocol = REPO / args.protocol
    scoring_protocol = REPO / args.scoring_protocol
    out_freeform = REPO / args.out_freeform
    out_manifest = REPO / args.out_manifest
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    if protocol_data.get("status") != "frozen_before_trajectory_outcomes":
        raise ValueError("anchor protocol must be frozen before trajectory outcomes")
    freeform_rows = build_freeform_anchors(read_jsonl(source))
    validate_capability_pack(read_jsonl(capability))
    out_freeform.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in freeform_rows),
        encoding="utf-8",
    )
    files = {
        os.path.relpath(path, out_manifest.parent): sha256_file(path)
        for path in (
            source,
            capability,
            protocol,
            scoring_protocol,
            out_freeform,
            REPO / "src/rlvr_safety/anchors.py",
            REPO / "src/rlvr_safety/cli/score_capability.py",
            REPO / "scripts/build_cross_format_anchors.py",
        )
    }
    manifest = {
        "schema_version": 1,
        "anchor_id": protocol_data["anchor_id"],
        "status": protocol_data["status"],
        "completion": {
            "freeform_items": len(freeform_rows),
            "capability_items": 30,
            "freeform_generations": protocol_data["freeform"]["total_generations"],
            "capability_generations": protocol_data["capability"]["total_generations"],
            "independent_human_review": False,
        },
        "files": files,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Froze 24 free-form anchors and 30 capability anchors")


if __name__ == "__main__":
    main()
