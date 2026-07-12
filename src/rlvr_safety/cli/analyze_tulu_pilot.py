from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..io import sha256_file
from ..pilot import analyze_endpoint_pilot, render_endpoint_pilot_markdown


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args(argv)

    with (args.result_dir / "choice_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((args.result_dir / "run_summary.json").read_text(encoding="utf-8"))
    run_metadata = json.loads(
        (args.result_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_config_hash = str(summary.get("config_sha256", ""))
    actual_config_hash = sha256_file(args.config)
    if actual_config_hash != expected_config_hash:
        raise SystemExit(
            f"config checksum mismatch: summary={expected_config_hash}, file={actual_config_hash}"
        )

    metrics = analyze_endpoint_pilot(rows, summary, run_metadata, config)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_endpoint_pilot_markdown(metrics), encoding="utf-8")
    print(f"Wrote Tülu endpoint feasibility metrics to {args.out_json}")
    print(f"Wrote Tülu endpoint feasibility report to {args.out_md}")


if __name__ == "__main__":
    main()
