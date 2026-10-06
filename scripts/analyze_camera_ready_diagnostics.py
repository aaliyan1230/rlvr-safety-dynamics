"""Reanalyze saved Tülu responses for the TAE camera-ready revision.

The additional order-specific and original-wording analyses are descriptive
robustness checks. They do not alter the frozen primary claim gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from rlvr_safety.factorial_analysis import (
    cluster_bootstrap_ci,
    simultaneous_cluster_bootstrap_intervals,
)


ROOT = Path(__file__).resolve().parents[1]
SCORE_DIRS = [ROOT / "artifacts" / f"tulu_trajectory_wave_{index:02d}" for index in range(1, 7)]
METRICS_PATH = ROOT / "artifacts/tulu_structured_trajectory_v1/metrics.json"
OUTPUT_PATH = ROOT / "reports/camera_ready_diagnostics_v1.md"
STEPS = [0, 40, 80, 160, 320, 640, 960, 1280, 1600, 1920, 2240, 2440]
ORDERS = ["012", "021", "102", "120", "201", "210"]
WORDINGS = ["original", "p1", "p2", "p3"]
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20260927


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _step(model: str) -> int:
    parts = model.split("_")
    if len(parts) < 4 or parts[0:2] != ["tulu", "step"]:
        raise ValueError(f"unexpected model id: {model}")
    return int(parts[2])


def load_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    paths = [directory / "choice_scores.csv" for directory in SCORE_DIRS]
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    return rows, paths


def scored_rows(rows: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if _step(str(row["model"])) != step:
            continue
        if _is_true(row.get("malformed")) or _is_true(row.get("verbosity_confounded")):
            continue
        score = row.get("instrumental_score_0_2", "")
        if score not in {"0", "1", "2"}:
            continue
        result.append(row)
    return result


def summarize_measurement_metrics(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    output = []
    sources = sorted({str(row["source_id"]) for row in rows})
    for step in STEPS:
        current = scored_rows(rows, step)
        invariance = []
        wording_ranges = []
        order_ranges = []
        for source in sources:
            source_inv = []
            source_word = []
            source_order = []
            for wording in WORDINGS:
                cells = [
                    row
                    for row in current
                    if str(row["source_id"]) == source and str(row["wording_id"]) == wording
                ]
                cell_scores = [int(row["instrumental_score_0_2"]) for row in cells]
                if not cell_scores:
                    raise ValueError(f"missing source/wording cell: step={step} {source} {wording}")
                source_inv.append(float(len(set(cell_scores)) == 1))
                source_order.append(max(cell_scores) - min(cell_scores))
                source_word.append(mean(cell_scores))
            invariance.extend(source_inv)
            wording_ranges.append(max(source_word) - min(source_word))
            order_ranges.append(mean(source_order))

        expected = 576
        output.append(
            {
                "step": step,
                "permutation_invariance": mean(invariance),
                "mean_wording_range": mean(wording_ranges),
                "mean_order_range": mean(order_ranges),
                "valid_rows": len(current),
                "censored_rows": expected - len(current),
            }
        )
    return output


def risk_by_order(rows: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    current = scored_rows(rows, step)
    summaries = []
    for order in ORDERS:
        selected = [row for row in current if str(row["option_order"]) == order]
        summaries.append(
            {
                "order": order,
                "n": len(selected),
                "mean": mean(int(row["instrumental_score_0_2"]) for row in selected),
            }
        )
    return summaries


def source_order_contrasts(
    rows: list[dict[str, Any]], baseline: int, later: int
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    by_step_order_source: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for step in (baseline, later):
        for row in scored_rows(rows, step):
            by_step_order_source[(step, str(row["option_order"]), str(row["source_id"]))].append(
                int(row["instrumental_score_0_2"])
            )

    effects: dict[str, dict[str, float]] = {}
    estimates: dict[str, float] = {}
    for order in ORDERS:
        baseline_sources = {
            source
            for step, observed_order, source in by_step_order_source
            if step == baseline and observed_order == order
        }
        later_sources = {
            source
            for step, observed_order, source in by_step_order_source
            if step == later and observed_order == order
        }
        if baseline_sources != later_sources:
            raise ValueError(f"source mismatch for order {order}: {baseline} versus {later}")
        effects[order] = {
            source: mean(by_step_order_source[(later, order, source)])
            - mean(by_step_order_source[(baseline, order, source)])
            for source in sorted(baseline_sources)
        }
        estimates[order] = mean(effects[order].values())

    intervals = simultaneous_cluster_bootstrap_intervals(
        effects,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
    )
    return estimates, intervals


def original_wording_contrast(
    rows: list[dict[str, Any]], baseline: int, later: int
) -> tuple[float, tuple[float, float]]:
    by_step_source: dict[tuple[int, str], list[int]] = defaultdict(list)
    for step in (baseline, later):
        for row in scored_rows(rows, step):
            if str(row["wording_id"]) == "original":
                by_step_source[(step, str(row["source_id"]))].append(
                    int(row["instrumental_score_0_2"])
                )
    base_sources = {source for step, source in by_step_source if step == baseline}
    later_sources = {source for step, source in by_step_source if step == later}
    if base_sources != later_sources or len(base_sources) != 24:
        raise ValueError("original-wording contrast requires all 24 paired source items")
    effects = {
        source: mean(by_step_source[(later, source)]) - mean(by_step_source[(baseline, source)])
        for source in sorted(base_sources)
    }
    return mean(effects.values()), cluster_bootstrap_ci(
        effects,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED + 1,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_interval(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"


def build_report(rows: list[dict[str, Any]], source_paths: list[Path]) -> str:
    measured = summarize_measurement_metrics(rows)
    frozen = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    frozen_by_step = {int(row["step"]): row for row in frozen["checkpoints"]}
    for metric in measured:
        prior = frozen_by_step[metric["step"]]["measurement"]
        for key in ("permutation_invariance", "mean_wording_range", "mean_order_range"):
            if abs(float(metric[key]) - float(prior[key])) > 1e-10:
                raise ValueError(f"recomputed {key} differs at step {metric['step']}")

    rows_by_step = {step: risk_by_order(rows, step) for step in (0, 1920, 2240)}
    order_effects, order_intervals = source_order_contrasts(rows, 0, 2240)
    original_effect, original_ci = original_wording_contrast(rows, 0, 2240)
    lines = [
        "# Camera-ready diagnostics from saved Tülu responses",
        "",
        "These checks use existing generations only. The original frozen analysis and claim gates are unchanged. The fixed-order and original-wording results below are post hoc diagnostics; they do not establish a new confirmatory safety-drift result.",
        "",
        "## Recomputed measurement metrics",
        "",
        "| Step | Permutation invariance | Mean wording range | Mean order range | Valid / 576 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in measured:
        lines.append(
            f"| {item['step']:,} | {item['permutation_invariance']:.3f} | "
            f"{item['mean_wording_range']:.3f} | {item['mean_order_range']:.3f} | "
            f"{item['valid_rows']} / 576 |"
        )
    lines += [
        "",
        "The wording range was absent from the submitted results table, although it is present in the saved trajectory metrics. This table makes its definition and checkpoint values visible. All values above reproduce the frozen metrics from the six saved score files.",
        "",
        "## Risk score by fixed option order",
        "",
        "The order code gives the semantic classes assigned to A/B/C, with 0=safe, 1=conditional, and 2=risky. A fixed order can therefore produce a different score even when the semantic options are unchanged.",
        "",
        "| Option order | Step 0 mean | Step 1,920 mean | Step 2,240 mean | Step 0→2,240 change | Simultaneous 95% source-bootstrap CI |",
        "|:---:|---:|---:|---:|---:|---:|",
    ]
    order_rows = {step: {row["order"]: row for row in stats} for step, stats in rows_by_step.items()}
    for order in ORDERS:
        lines.append(
            f"| {order} | {order_rows[0][order]['mean']:.3f} | "
            f"{order_rows[1920][order]['mean']:.3f} | {order_rows[2240][order]['mean']:.3f} | "
            f"{order_effects[order]:+.3f} | {_fmt_interval(order_intervals[order])} |"
        )
    lines += [
        "",
        "The six contrasts resample the same 24 authored sources and use one max-|t| simultaneous interval family. They describe how fixed layouts behave in this item pool; they do not redefine the primary order-marginalized estimand.",
        "",
        "## Original wording only",
        "",
        f"The source-clustered step-0 to step-2,240 contrast using only the original wording and all six option orders is **{original_effect:+.3f}** (95% percentile bootstrap CI {_fmt_interval(original_ci)}; 24 sources). This is a post hoc sensitivity check because paraphrases were part of the frozen primary instrument.",
        "",
        "## Provenance",
        "",
        f"- Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}; seed: {BOOTSTRAP_SEED}.",
        f"- Frozen metric input SHA-256: `{sha256(METRICS_PATH)}`.",
    ]
    lines.extend(f"- `{path.relative_to(ROOT)}` SHA-256: `{sha256(path)}`." for path in source_paths)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows, paths = load_rows()
    expected = len(STEPS) * 576
    if len(rows) != expected:
        raise ValueError(f"expected {expected} saved rows, found {len(rows)}")
    report = build_report(rows, paths)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
