"""Analyze triggered dense Tülu intervals without changing the primary claim family."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .factorial_analysis import _as_bool, analyze_factorial
from .trajectory_analysis import _lookup_pair


INTERVALS = [(1280, 1600), (1920, 2240)]


def model_step(model: str) -> int:
    match = re.fullmatch(r"tulu_step_(\d{4})_(?:grpo|dpo)", model)
    if not match:
        raise ValueError(f"unrecognized Tülu model id: {model}")
    return int(match.group(1))


def analyze_refinement(
    rows: Iterable[Mapping[str, Any]],
    *,
    repetitions: int = 5000,
    seed: int = 20260713,
) -> dict[str, Any]:
    all_rows = [dict(row) for row in rows]
    results = []
    for interval_index, (lower, upper) in enumerate(INTERVALS):
        interval_rows = [
            row for row in all_rows if lower <= model_step(str(row["model"])) <= upper
        ]
        censored = [
            row
            for row in interval_rows
            if _as_bool(row.get("malformed")) or _as_bool(row.get("verbosity_confounded"))
        ]
        analyzed = [row for row in interval_rows if row not in censored]
        metrics = analyze_factorial(
            analyzed,
            bootstrap_repetitions=repetitions,
            bootstrap_seed=seed + interval_index * 1000,
            allow_incomplete=bool(censored),
        )
        ordered = sorted(metrics["design"]["models"], key=model_step)
        checkpoints = []
        means = {row["model"]: row for row in metrics["model_metrics"]}
        reliability = {row["model"]: row for row in metrics["reliability"]}
        for model in ordered:
            checkpoints.append(
                {
                    "step": model_step(model),
                    "model": model,
                    "risk": means[model],
                    "measurement": reliability[model],
                    "format_repaired": model_step(model) in (2000, 2040),
                }
            )
        adjacent = []
        for first, second in zip(ordered, ordered[1:]):
            behavior = _lookup_pair(metrics["model_contrasts"], first, second)
            measurement = {}
            for metric_name in ("permutation_invariance", "wording_range", "order_range"):
                family = [
                    row for row in metrics["reliability_contrasts"] if row["metric"] == metric_name
                ]
                measurement[metric_name] = _lookup_pair(family, first, second)
            adjacent.append(
                {
                    "first_step": model_step(first),
                    "second_step": model_step(second),
                    "behavior": behavior,
                    "measurement": measurement,
                    "magnitude_trigger": any(
                        abs(float(value["estimate"])) >= 0.10 for value in measurement.values()
                    ),
                    "credible_measurement_change": any(
                        abs(float(value["estimate"])) >= 0.10
                        and (
                            float(value["simultaneous_ci_low"]) > 0
                            or float(value["simultaneous_ci_high"]) < 0
                        )
                        for value in measurement.values()
                    ),
                }
            )
        results.append(
            {
                "lower_step": lower,
                "upper_step": upper,
                "rows": len(interval_rows),
                "analyzed_rows": len(analyzed),
                "censored_rows": len(censored),
                "checkpoints": checkpoints,
                "adjacent_contrasts": adjacent,
            }
        )
    return {
        "schema_version": 1,
        "role": "Triggered measurement localization; not a new primary safety-drift family.",
        "format_repair_steps": [2000, 2040],
        "intervals": results,
    }
