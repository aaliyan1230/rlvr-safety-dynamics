"""Compare historical single-layout scores with matching factorial cells."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from .factorial_analysis import _as_bool, cluster_bootstrap_ci


class ReproductionError(ValueError):
    pass


def compare_historical_layouts(
    factorial_rows: Iterable[Mapping[str, Any]],
    historical_packs: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    model_keys: Mapping[str, Mapping[str, str]] | None = None,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260710,
) -> dict[str, Any]:
    factorial_rows = list(factorial_rows)
    model_keys = model_keys or {}
    selected = {}
    for row in factorial_rows:
        wording = str(row.get("wording_id", ""))
        keep = (
            wording == "original" and _as_bool(row.get("matches_source_order"))
        ) or (wording != "original" and str(row.get("option_order", "")) == "012")
        if not keep:
            continue
        key = (wording, str(row.get("source_id", "")), str(row.get("model", "")))
        if key in selected:
            raise ReproductionError(f"duplicate factorial layout cell: {key}")
        selected[key] = row

    comparisons = []
    difference_examples = []
    for pack_index, (wording, raw_rows) in enumerate(historical_packs.items()):
        key_map = model_keys.get(wording, {})
        historical = {}
        for row in raw_rows:
            score = row.get("instrumental_score_0_2", "")
            if score == "":
                continue
            model = key_map.get(str(row.get("model", "")), str(row.get("model", "")))
            key = (str(row.get("source_id", "")), model)
            if key in historical:
                raise ReproductionError(f"duplicate historical {wording} cell: {key}")
            historical[key] = row

        current_keys = {
            (source, model)
            for pack, source, model in selected
            if pack == wording and (source, model) in historical
        }
        if not current_keys:
            raise ReproductionError(f"no matching factorial cells for historical pack {wording}")
        models = sorted({model for _, model in current_keys})
        for model_index, model in enumerate(models):
            keys = sorted(key for key in current_keys if key[1] == model)
            historical_sources = {source for source, hist_model in historical if hist_model == model}
            current_sources = {
                source
                for pack, source, current_model in selected
                if pack == wording and current_model == model
            }
            if {source for source, _ in keys} != historical_sources or historical_sources != current_sources:
                raise ReproductionError(
                    f"{wording}/{model}: source coverage differs between historical and factorial"
                )
            deltas = {}
            score_matches = 0
            choice_matches = 0
            for source, _ in keys:
                old = historical[(source, model)]
                new = selected[(wording, source, model)]
                old_score = int(old["instrumental_score_0_2"])
                new_score = int(new["instrumental_score_0_2"])
                deltas[source] = new_score - old_score
                score_matches += old_score == new_score
                choice_matches += str(old.get("choice", "")) == str(new.get("choice", ""))
                if old_score != new_score:
                    difference_examples.append(
                        {
                            "wording_id": wording,
                            "model": model,
                            "source_id": source,
                            "historical_choice": old.get("choice", ""),
                            "factorial_choice": new.get("choice", ""),
                            "historical_score": old_score,
                            "factorial_score": new_score,
                        }
                    )
            low, high = cluster_bootstrap_ci(
                deltas,
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed + pack_index * 20 + model_index,
            )
            comparisons.append(
                {
                    "wording_id": wording,
                    "model": model,
                    "rows": len(keys),
                    "historical_mean": mean(
                        int(historical[key]["instrumental_score_0_2"]) for key in keys
                    ),
                    "factorial_mean": mean(
                        int(selected[(wording, key[0], key[1])]["instrumental_score_0_2"])
                        for key in keys
                    ),
                    "factorial_minus_historical": mean(deltas.values()),
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "score_agreement": score_matches / len(keys),
                    "choice_agreement": choice_matches / len(keys),
                }
            )
    return {
        "schema_version": 1,
        "comparisons": comparisons,
        "different_score_cells": len(difference_examples),
        "difference_examples": difference_examples,
    }


def render_reproduction_markdown(metrics: Mapping[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Each factorial condition below exactly matches the historical semantic layout: source "
        "order for the original pack and order 012 for candidate packs.",
        "",
        "| Pack | Model | Historical mean | Factorial mean | Difference [95% CI] | Score agreement | Choice agreement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["comparisons"]:
        lines.append(
            f"| {row['wording_id']} | {row['model']} | {row['historical_mean']:.3f} | "
            f"{row['factorial_mean']:.3f} | {row['factorial_minus_historical']:+.3f} "
            f"[{row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f}] | "
            f"{row['score_agreement']:.1%} | {row['choice_agreement']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Different scored cells: {metrics['different_score_cells']}.",
            "",
            "A mismatch means the historical protocol was not reproduced; it can reflect "
            "quantization, dependency, model-revision, or other inference drift.",
        ]
    )
    return "\n".join(lines) + "\n"
