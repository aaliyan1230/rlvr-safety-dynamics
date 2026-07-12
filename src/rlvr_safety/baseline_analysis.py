"""Paired uncertainty summaries for the historical free-form and stage runs."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from .factorial_analysis import cluster_bootstrap_ci


def _score(row: Mapping[str, Any]) -> int:
    return int(row["instrumental_score_0_2"])


def analyze_baseline(
    freeform_rows: Iterable[Mapping[str, Any]],
    freeform_key: Mapping[str, str],
    choice_rows: Iterable[Mapping[str, Any]],
    choice_key: Mapping[str, str],
    *,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260710,
) -> dict[str, Any]:
    freeform = [dict(row) for row in freeform_rows if row.get("category") != "benign_control"]
    by_model_item = {
        (freeform_key.get(str(row["model"]), str(row["model"])), str(row["id"])): row
        for row in freeform
    }
    instruct = "allenai/Olmo-3-7B-Instruct"
    rlzero = "allenai/Olmo-3-7B-RL-Zero-General"
    source_ids = sorted(
        {item for model, item in by_model_item if model == instruct}
        & {item for model, item in by_model_item if model == rlzero}
    )
    if len(source_ids) != 40:
        raise ValueError(f"expected 40 paired free-form risk items, got {len(source_ids)}")
    deltas = {
        item: _score(by_model_item[(rlzero, item)]) - _score(by_model_item[(instruct, item)])
        for item in source_ids
    }
    freeform_ci = cluster_bootstrap_ci(
        deltas, repetitions=bootstrap_repetitions, seed=bootstrap_seed
    )
    clean_ids = [
        item
        for item in source_ids
        if str(by_model_item[(rlzero, item)].get("verbosity_confounded", "")).lower() != "true"
    ]

    choice = [dict(row) for row in choice_rows if row.get("instrumental_score_0_2", "") != ""]
    clean_choice_models = [
        model
        for model in choice_key.values()
        if model != "allenai/Olmo-3-7B-RL-Zero-General"
    ]
    choice_lookup = {
        (choice_key.get(str(row["model"]), str(row["model"])), str(row["source_id"])): _score(row)
        for row in choice
    }
    choice_sources = sorted(
        set.intersection(
            *[
                {source for model, source in choice_lookup if model == clean_model}
                for clean_model in clean_choice_models
            ]
        )
    )
    if len(choice_sources) != 24:
        raise ValueError(f"expected 24 complete choice items, got {len(choice_sources)}")
    all_equal = sum(
        len({choice_lookup[(model, source)] for model in clean_choice_models}) == 1
        for source in choice_sources
    )
    model_means = {
        model: mean(choice_lookup[(model, source)] for source in choice_sources)
        for model in clean_choice_models
    }
    contrasts = []
    for index, (first, second) in enumerate(itertools.combinations(clean_choice_models, 2)):
        paired = {
            source: choice_lookup[(second, source)] - choice_lookup[(first, source)]
            for source in choice_sources
        }
        low, high = cluster_bootstrap_ci(
            paired,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + 100 + index,
        )
        contrasts.append(
            {
                "first": first,
                "second": second,
                "difference": mean(paired.values()),
                "ci_low": low,
                "ci_high": high,
                "nonzero_items": sum(delta != 0 for delta in paired.values()),
            }
        )

    return {
        "bootstrap": {"repetitions": bootstrap_repetitions, "seed": bootstrap_seed},
        "freeform": {
            "instruct_mean": mean(_score(by_model_item[(instruct, item)]) for item in source_ids),
            "rlzero_mean": mean(_score(by_model_item[(rlzero, item)]) for item in source_ids),
            "paired_difference_rlzero_minus_instruct": mean(deltas.values()),
            "ci_low": freeform_ci[0],
            "ci_high": freeform_ci[1],
            "higher_tied_lower": {
                "rlzero_higher": sum(delta > 0 for delta in deltas.values()),
                "tied": sum(delta == 0 for delta in deltas.values()),
                "rlzero_lower": sum(delta < 0 for delta in deltas.values()),
            },
            "unconfounded_pairs": len(clean_ids),
            "unconfounded_instruct_mean": mean(
                _score(by_model_item[(instruct, item)]) for item in clean_ids
            ),
            "unconfounded_rlzero_mean": mean(
                _score(by_model_item[(rlzero, item)]) for item in clean_ids
            ),
        },
        "choice_stage": {
            "model_means": model_means,
            "model_stage_spread": max(model_means.values()) - min(model_means.values()),
            "all_models_equal_items": all_equal,
            "total_items": len(choice_sources),
            "contrasts": contrasts,
        },
    }


def render_baseline_markdown(metrics: Mapping[str, Any]) -> str:
    freeform = metrics["freeform"]
    stage = metrics["choice_stage"]
    lines = [
        "# Baseline Paired Uncertainty Analysis",
        "",
        "Bootstrap intervals resample the 24 or 40 authored source items. They are diagnostic "
        "uncertainty intervals, not population-level confidence intervals.",
        "",
        "## Adjudicated free-form comparison",
        "",
        "| Instruct mean | RL-Zero mean | RL-Zero − Instruct [95% interval] | Higher / tied / lower |",
        "|---:|---:|---:|---:|",
        f"| {freeform['instruct_mean']:.3f} | {freeform['rlzero_mean']:.3f} | "
        f"{freeform['paired_difference_rlzero_minus_instruct']:+.3f} "
        f"[{freeform['ci_low']:+.3f}, {freeform['ci_high']:+.3f}] | "
        f"{freeform['higher_tied_lower']['rlzero_higher']} / "
        f"{freeform['higher_tied_lower']['tied']} / "
        f"{freeform['higher_tied_lower']['rlzero_lower']} |",
        "",
        f"On the {freeform['unconfounded_pairs']} pairs where the RL-Zero row was not manually "
        f"marked verbosity/token-cap confounded, means were "
        f"{freeform['unconfounded_instruct_mean']:.3f} (Instruct) and "
        f"{freeform['unconfounded_rlzero_mean']:.3f} (RL-Zero).",
        "",
        "## Clean structured-choice stages",
        "",
        "| Model | Mean |",
        "|---|---:|",
    ]
    for model, value in stage["model_means"].items():
        lines.append(f"| {model} | {value:.3f} |")
    lines.extend(
        [
            "",
            f"All three clean models made the same scored choice on "
            f"{stage['all_models_equal_items']}/{stage['total_items']} items. The marginal "
            f"model-stage spread is {stage['model_stage_spread']:.3f}.",
            "",
            "| First | Second | Second − first [95% interval] | Nonzero paired items |",
            "|---|---|---:|---:|",
        ]
    )
    for contrast in stage["contrasts"]:
        lines.append(
            f"| {contrast['first']} | {contrast['second']} | "
            f"{contrast['difference']:+.3f} "
            f"[{contrast['ci_low']:+.3f}, {contrast['ci_high']:+.3f}] | "
            f"{contrast['nonzero_items']}/{stage['total_items']} |"
        )
    return "\n".join(lines) + "\n"
