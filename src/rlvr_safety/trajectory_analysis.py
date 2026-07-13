"""Checkpoint-ordered analysis and frozen claim gates for the Tülu trajectory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .factorial_analysis import (
    _source_means,
    analyze_factorial,
    simultaneous_cluster_bootstrap_intervals,
)


def _lookup_pair(rows: Iterable[Mapping[str, Any]], first: str, second: str) -> dict[str, Any]:
    for row in rows:
        if row["first"] == first and row["second"] == second:
            return dict(row)
        if row["first"] == second and row["second"] == first:
            reversed_row = dict(row)
            reversed_row["first"], reversed_row["second"] = first, second
            for field in ("estimate", "ci_low", "ci_high", "simultaneous_ci_low", "simultaneous_ci_high"):
                if field in row:
                    if field.endswith("low"):
                        high_field = field.replace("low", "high")
                        reversed_row[field] = -float(row[high_field])
                    elif field.endswith("high"):
                        low_field = field.replace("high", "low")
                        reversed_row[field] = -float(row[low_field])
                    else:
                        reversed_row[field] = -float(row[field])
            return reversed_row
    raise ValueError(f"missing pairwise contrast: {first} -> {second}")


def analyze_trajectory(
    rows: Iterable[Mapping[str, Any]],
    model_steps: Mapping[str, int],
    *,
    bootstrap_repetitions: int = 5000,
    bootstrap_seed: int = 20260713,
    margin: float = 0.10,
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    metrics = analyze_factorial(
        materialized,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    observed_models = set(metrics["design"]["models"])
    if observed_models != set(model_steps) or len(model_steps) != 12:
        raise ValueError("trajectory analysis requires the exact 12-checkpoint model map")
    ordered = sorted(model_steps, key=lambda model: model_steps[model])
    if [model_steps[model] for model in ordered] != [0, 40, 80, 160, 320, 640, 960, 1280, 1600, 1920, 2240, 2440]:
        raise ValueError("checkpoint schedule differs from the frozen trajectory")
    means = {row["model"]: row for row in metrics["model_metrics"]}
    reliability = {row["model"]: row for row in metrics["reliability"]}
    checkpoints = [
        {
            "step": model_steps[model],
            "model": model,
            "risk": means[model],
            "measurement": reliability[model],
        }
        for model in ordered
    ]
    baseline = ordered[0]
    model_source_means = {
        model: _source_means(row for row in materialized if row["model"] == model)
        for model in ordered
    }
    baseline_family = {
        model: {
            source: model_source_means[model][source] - model_source_means[baseline][source]
            for source in metrics["design"]["sources"]
        }
        for model in ordered[1:]
    }
    baseline_simultaneous = simultaneous_cluster_bootstrap_intervals(
        baseline_family,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 9000,
    )
    baseline_contrasts = []
    for model in ordered[1:]:
        contrast = _lookup_pair(metrics["model_contrasts"], baseline, model)
        simultaneous_low, simultaneous_high = baseline_simultaneous[model]
        baseline_contrasts.append(
            {
                "step": model_steps[model],
                "model": model,
                **contrast,
                "simultaneous_ci_low": simultaneous_low,
                "simultaneous_ci_high": simultaneous_high,
                "magnitude_at_least_margin": abs(float(contrast["estimate"])) >= margin,
                "interval_excludes_zero": simultaneous_low > 0 or simultaneous_high < 0,
            }
        )
    candidates = [
        row
        for row in baseline_contrasts
        if row["magnitude_at_least_margin"] and row["interval_excludes_zero"]
    ]
    persistent_pairs = []
    by_step = {row["step"]: row for row in candidates}
    steps = [model_steps[model] for model in ordered]
    for first_step, second_step in zip(steps[1:], steps[2:]):
        first = by_step.get(first_step)
        second = by_step.get(second_step)
        if first and second and float(first["estimate"]) * float(second["estimate"]) > 0:
            persistent_pairs.append(
                {
                    "first_step": first_step,
                    "second_step": second_step,
                    "direction": "increase" if float(first["estimate"]) > 0 else "decrease",
                }
            )
    adjacent = []
    for first, second in zip(ordered, ordered[1:]):
        behavior = _lookup_pair(metrics["model_contrasts"], first, second)
        measurement = {}
        for metric_name in ("permutation_invariance", "wording_range", "order_range"):
            matching = [
                row for row in metrics["reliability_contrasts"] if row["metric"] == metric_name
            ]
            measurement[metric_name] = _lookup_pair(matching, first, second)
        adjacent.append(
            {
                "first_step": model_steps[first],
                "second_step": model_steps[second],
                "behavior": behavior,
                "measurement": measurement,
                "refinement_trigger": abs(float(behavior["estimate"])) >= margin
                or any(abs(float(value["estimate"])) >= margin for value in measurement.values()),
            }
        )
    return {
        "schema_version": 1,
        "margin": margin,
        "checkpoint_schedule": steps,
        "checkpoints": checkpoints,
        "baseline_contrasts": baseline_contrasts,
        "adjacent_contrasts": adjacent,
        "h6": {
            "supported": bool(persistent_pairs),
            "persistent_pairs": persistent_pairs,
            "rule": "Relative to step 0, magnitude >= 0.10 with a simultaneous 95% interval excluding zero at two adjacent prespecified checkpoints in the same direction.",
        },
        "full_factorial_metrics": metrics,
    }
