"""Dependency-free analysis for balanced model × wording × option-order runs."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean, stdev
from typing import Any


class FactorialAnalysisError(ValueError):
    """Raised when scored data do not form the declared complete design."""


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _score(row: Mapping[str, Any]) -> int:
    raw = row.get("instrumental_score_0_2", "")
    try:
        score = int(raw)
    except (TypeError, ValueError) as exc:
        raise FactorialAnalysisError(
            f"{row.get('id', '<unknown>')}: invalid score {raw!r}"
        ) from exc
    if score not in (0, 1, 2):
        raise FactorialAnalysisError(f"{row.get('id', '<unknown>')}: score outside 0..2: {score}")
    return score


def validate_scored_factorial(
    rows: Iterable[Mapping[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise FactorialAnalysisError("no scored rows found")

    required = {"id", "source_id", "model", "wording_id", "option_order"}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        missing = sorted(field for field in required if not row.get(field))
        if missing:
            raise FactorialAnalysisError(f"row missing required fields {missing}: {row!r}")
        key = (str(row["id"]), str(row["model"]))
        if key in seen:
            raise FactorialAnalysisError(f"duplicate scored cell {key}")
        seen.add(key)
        if not allow_incomplete and (
            _as_bool(row.get("malformed")) or _as_bool(row.get("verbosity_confounded"))
        ):
            raise FactorialAnalysisError(
                f"{key}: malformed/token-capped cell; rerun or pass --allow-incomplete"
            )
        _score(row)

    models = sorted({str(row["model"]) for row in rows})
    sources = sorted({str(row["source_id"]) for row in rows})
    wordings = sorted({str(row["wording_id"]) for row in rows})
    orders = sorted({str(row["option_order"]) for row in rows})
    expected_orders = sorted(
        "".join(map(str, order)) for order in itertools.permutations((0, 1, 2))
    )
    if orders != expected_orders:
        raise FactorialAnalysisError(f"expected all six score orders; got {orders}")
    if "original" not in wordings:
        raise FactorialAnalysisError("wording_id=original is required for paired contrasts")

    expected = {
        (model, source, wording, order)
        for model in models
        for source in sources
        for wording in wordings
        for order in orders
    }
    observed = Counter(
        (
            str(row["model"]),
            str(row["source_id"]),
            str(row["wording_id"]),
            str(row["option_order"]),
        )
        for row in rows
    )
    if any(count != 1 for count in observed.values()):
        raise FactorialAnalysisError("duplicate model × source × wording × order cells")
    if not allow_incomplete and set(observed) != expected:
        missing = sorted(expected - set(observed))[:10]
        extra = sorted(set(observed) - expected)[:10]
        raise FactorialAnalysisError(
            "incomplete model × source × wording × order crossing; "
            f"missing={missing}, extra={extra}"
        )
    if allow_incomplete:
        extra = set(observed) - expected
        if extra:
            raise FactorialAnalysisError(f"unexpected factorial cells: {sorted(extra)[:10]}")
        minimum_groups = {
            (model, source, wording)
            for model in models
            for source in sources
            for wording in wordings
        }
        observed_groups = {
            (str(row["model"]), str(row["source_id"]), str(row["wording_id"]))
            for row in rows
        }
        if observed_groups != minimum_groups:
            raise FactorialAnalysisError(
                "incomplete analysis requires at least one order in every model × source × wording cell"
            )

    design = {
        "rows": len(rows),
        "models": models,
        "sources": sources,
        "wordings": wordings,
        "orders": orders,
    }
    if allow_incomplete:
        design["expected_rows"] = len(expected)
        design["missing_cells"] = len(expected) - len(observed)
    return design


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def cluster_bootstrap_ci(
    values_by_source: Mapping[str, float],
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    sources = sorted(values_by_source)
    if not sources:
        return math.nan, math.nan
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sampled = [rng.choice(sources) for _ in sources]
        draws.append(mean(values_by_source[source] for source in sampled))
    draws.sort()
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def empirical_shift_power(
    values_by_source: Mapping[str, float],
    *,
    minimum_effect: float,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    """Estimate paired two-sided power from the observed source-level residuals.

    This is a design diagnostic, not a model-based power guarantee. It centers the
    observed paired source effects, bootstraps their mean under the null, and then
    shifts that same empirical sampling distribution by ``minimum_effect`` in both
    directions. Reporting the weaker direction avoids silently assuming symmetry.
    """

    sources = sorted(values_by_source)
    if not sources:
        raise FactorialAnalysisError("power diagnostic requires at least one source")
    observed_mean = mean(values_by_source.values())
    residuals = [values_by_source[source] - observed_mean for source in sources]
    rng = random.Random(seed)
    null_draws = sorted(mean(rng.choice(residuals) for _ in sources) for _ in range(repetitions))
    critical_low = _percentile(null_draws, 0.025)
    critical_high = _percentile(null_draws, 0.975)

    def rejected(value: float) -> bool:
        return value < critical_low or value > critical_high

    positive_power = mean(rejected(value + minimum_effect) for value in null_draws)
    negative_power = mean(rejected(value - minimum_effect) for value in null_draws)
    return {
        "source_count": len(sources),
        "minimum_effect": minimum_effect,
        "observed_mean_difference": observed_mean,
        "null_critical_low": critical_low,
        "null_critical_high": critical_high,
        "positive_power": positive_power,
        "negative_power": negative_power,
        "minimum_directional_power": min(positive_power, negative_power),
    }


def simultaneous_cluster_bootstrap_intervals(
    values_by_contrast: Mapping[Any, Mapping[str, float]],
    *,
    repetitions: int,
    seed: int,
) -> dict[Any, tuple[float, float]]:
    """Return max-|t| simultaneous intervals for a family of source contrasts."""

    if repetitions < 2:
        raise FactorialAnalysisError("simultaneous bootstrap requires at least two repetitions")
    if not values_by_contrast:
        return {}
    source_sets = {tuple(sorted(values)) for values in values_by_contrast.values()}
    if len(source_sets) != 1 or not next(iter(source_sets)):
        raise FactorialAnalysisError("simultaneous contrasts must share non-empty source ids")
    sources = list(next(iter(source_sets)))
    estimates = {key: mean(values.values()) for key, values in values_by_contrast.items()}
    draws = {key: [] for key in values_by_contrast}
    rng = random.Random(seed)
    for _ in range(repetitions):
        sampled = [rng.choice(sources) for _ in sources]
        for key, values in values_by_contrast.items():
            draws[key].append(mean(values[source] for source in sampled))
    standard_errors = {
        key: stdev(values) if len(set(values)) > 1 else 0.0 for key, values in draws.items()
    }
    maxima = []
    for index in range(repetitions):
        maxima.append(
            max(
                (
                    abs(draws[key][index] - estimates[key]) / standard_errors[key]
                    if standard_errors[key]
                    else 0.0
                )
                for key in values_by_contrast
            )
        )
    critical_value = _percentile(sorted(maxima), 0.95)
    return {
        key: (
            estimates[key] - critical_value * standard_errors[key],
            estimates[key] + critical_value * standard_errors[key],
        )
        for key in values_by_contrast
    }


def _group(
    rows: Iterable[Mapping[str, Any]], *fields: str
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    return grouped


def _source_means(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    grouped = _group(rows, "source_id")
    return {key[0]: mean(_score(row) for row in values) for key, values in grouped.items()}


def _estimate(
    values_by_source: Mapping[str, float], repetitions: int, seed: int
) -> dict[str, float]:
    low, high = cluster_bootstrap_ci(values_by_source, repetitions=repetitions, seed=seed)
    return {"estimate": mean(values_by_source.values()), "ci_low": low, "ci_high": high}


def _variance_components(rows: list[Mapping[str, Any]]) -> list[dict[str, float | str]]:
    scores = [_score(row) for row in rows]
    grand = mean(scores)
    total = sum((score - grand) ** 2 for score in scores)

    def main_effect(field: str) -> float:
        grouped = _group(rows, field)
        return sum(
            len(group_rows) * (mean(_score(row) for row in group_rows) - grand) ** 2
            for group_rows in grouped.values()
        )

    def interaction(first: str, second: str) -> float:
        first_means = {
            key[0]: mean(_score(row) for row in group_rows)
            for key, group_rows in _group(rows, first).items()
        }
        second_means = {
            key[0]: mean(_score(row) for row in group_rows)
            for key, group_rows in _group(rows, second).items()
        }
        return sum(
            len(group_rows)
            * (
                mean(_score(row) for row in group_rows)
                - first_means[key[0]]
                - second_means[key[1]]
                + grand
            )
            ** 2
            for key, group_rows in _group(rows, first, second).items()
        )

    components: list[tuple[str, float]] = [
        ("model", main_effect("model")),
        ("source_item", main_effect("source_id")),
        ("wording", main_effect("wording_id")),
        ("option_order", main_effect("option_order")),
        ("model × wording", interaction("model", "wording_id")),
        ("model × option_order", interaction("model", "option_order")),
    ]
    explained = sum(value for _, value in components)
    components.append(("other interactions / residual", max(0.0, total - explained)))
    return [
        {
            "component": name,
            "sum_squares": value,
            "share_total": value / total if total else 0.0,
        }
        for name, value in components
    ]


def analyze_factorial(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_repetitions: int = 5000,
    bootstrap_seed: int = 20260710,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    rows = list(rows)
    design = validate_scored_factorial(rows, allow_incomplete=allow_incomplete)
    models: list[str] = design["models"]
    wordings: list[str] = design["wordings"]

    model_metrics = []
    for index, model in enumerate(models):
        model_rows = [row for row in rows if row["model"] == model]
        source_means = _source_means(model_rows)
        estimate = _estimate(source_means, bootstrap_repetitions, bootstrap_seed + index)
        scores = [_score(row) for row in model_rows]
        model_metrics.append(
            {
                "model": model,
                **estimate,
                "p_nonzero": sum(score > 0 for score in scores) / len(scores),
                "p_score_2": sum(score == 2 for score in scores) / len(scores),
                "rows": len(scores),
            }
        )

    wording_metrics = []
    for model_index, model in enumerate(models):
        original_rows = [
            row for row in rows if row["model"] == model and row["wording_id"] == "original"
        ]
        original_by_source = _source_means(original_rows)
        for wording_index, wording in enumerate(wordings):
            cell_rows = [
                row for row in rows if row["model"] == model and row["wording_id"] == wording
            ]
            by_source = _source_means(cell_rows)
            delta_by_source = {
                source: by_source[source] - original_by_source[source]
                for source in design["sources"]
            }
            estimate = _estimate(
                by_source,
                bootstrap_repetitions,
                bootstrap_seed + 100 + model_index * 20 + wording_index,
            )
            delta = _estimate(
                delta_by_source,
                bootstrap_repetitions,
                bootstrap_seed + 200 + model_index * 20 + wording_index,
            )
            wording_metrics.append(
                {
                    "model": model,
                    "wording_id": wording,
                    **estimate,
                    "delta_vs_original": delta["estimate"],
                    "delta_ci_low": delta["ci_low"],
                    "delta_ci_high": delta["ci_high"],
                }
            )

    # Reconstruct the historical single-layout comparison inside the new run:
    # original items used their source order, while paraphrase packs used 012.
    # Contrasting that with the 3!-marginalized delta isolates how much the old
    # headline was changed by the layout confound.
    naive_vs_marginalized = []
    for model_index, model in enumerate(models):
        original_all = [
            row for row in rows if row["model"] == model and row["wording_id"] == "original"
        ]
        original_marginal = _source_means(original_all)
        original_naive = _source_means(
            row for row in original_all if _as_bool(row.get("matches_source_order"))
        )
        for wording_index, wording in enumerate(w for w in wordings if w != "original"):
            candidate_all = [
                row for row in rows if row["model"] == model and row["wording_id"] == wording
            ]
            candidate_marginal = _source_means(candidate_all)
            candidate_naive = _source_means(
                row for row in candidate_all if str(row["option_order"]) == "012"
            )
            naive_delta = {
                source: candidate_naive[source] - original_naive[source]
                for source in design["sources"]
            }
            marginalized_delta = {
                source: candidate_marginal[source] - original_marginal[source]
                for source in design["sources"]
            }
            confounding_shift = {
                source: naive_delta[source] - marginalized_delta[source]
                for source in design["sources"]
            }
            naive_estimate = _estimate(
                naive_delta,
                bootstrap_repetitions,
                bootstrap_seed + 700 + model_index * 20 + wording_index,
            )
            marginal_estimate = _estimate(
                marginalized_delta,
                bootstrap_repetitions,
                bootstrap_seed + 800 + model_index * 20 + wording_index,
            )
            shift_estimate = _estimate(
                confounding_shift,
                bootstrap_repetitions,
                bootstrap_seed + 900 + model_index * 20 + wording_index,
            )
            naive_vs_marginalized.append(
                {
                    "model": model,
                    "wording_id": wording,
                    "naive_delta": naive_estimate["estimate"],
                    "naive_ci_low": naive_estimate["ci_low"],
                    "naive_ci_high": naive_estimate["ci_high"],
                    "marginalized_delta": marginal_estimate["estimate"],
                    "marginalized_ci_low": marginal_estimate["ci_low"],
                    "marginalized_ci_high": marginal_estimate["ci_high"],
                    "layout_confounding_shift": shift_estimate["estimate"],
                    "shift_ci_low": shift_estimate["ci_low"],
                    "shift_ci_high": shift_estimate["ci_high"],
                }
            )

    category_wording_metrics = []
    categories = sorted({str(row["category"]) for row in rows})
    for category_index, category in enumerate(categories):
        original_by_source = _source_means(
            row for row in rows if row["category"] == category and row["wording_id"] == "original"
        )
        for wording_index, wording in enumerate(wordings):
            by_source = _source_means(
                row for row in rows if row["category"] == category and row["wording_id"] == wording
            )
            deltas = {
                source: by_source[source] - original_by_source[source] for source in by_source
            }
            delta_estimate = _estimate(
                deltas,
                bootstrap_repetitions,
                bootstrap_seed + 1000 + category_index * 20 + wording_index,
            )
            category_wording_metrics.append(
                {
                    "category": category,
                    "wording_id": wording,
                    "mean": mean(by_source.values()),
                    "delta_vs_original": delta_estimate["estimate"],
                    "delta_ci_low": delta_estimate["ci_low"],
                    "delta_ci_high": delta_estimate["ci_high"],
                    "mean_abs_source_delta": mean(abs(value) for value in deltas.values()),
                }
            )

    position_metrics = []
    for model_index, model in enumerate(models):
        for position_index, position in enumerate(("A", "B", "C")):
            cell_rows = [
                row for row in rows if row["model"] == model and row["risky_position"] == position
            ]
            estimate = _estimate(
                _source_means(cell_rows),
                bootstrap_repetitions,
                bootstrap_seed + 300 + model_index * 10 + position_index,
            )
            position_metrics.append({"model": model, "risky_position": position, **estimate})

    position_family = {}
    for scope in ["all_models", *models]:
        position_by_source = {
            position: _source_means(
                row
                for row in rows
                if row["risky_position"] == position
                and (scope == "all_models" or row["model"] == scope)
            )
            for position in ("A", "B", "C")
        }
        for first, second in itertools.combinations(("A", "B", "C"), 2):
            position_family[(scope, first, second)] = {
                source: position_by_source[second][source] - position_by_source[first][source]
                for source in design["sources"]
            }
    simultaneous_positions = simultaneous_cluster_bootstrap_intervals(
        position_family,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 350,
    )
    position_contrasts = []
    for contrast_index, (key, values) in enumerate(position_family.items()):
        scope, first, second = key
        estimate = _estimate(
            values,
            bootstrap_repetitions,
            bootstrap_seed + 360 + contrast_index,
        )
        simultaneous_low, simultaneous_high = simultaneous_positions[key]
        position_contrasts.append(
            {
                "scope": scope,
                "first_position": first,
                "second_position": second,
                **estimate,
                "simultaneous_ci_low": simultaneous_low,
                "simultaneous_ci_high": simultaneous_high,
            }
        )

    model_contrasts = []
    model_source_means = {
        model: _source_means(row for row in rows if row["model"] == model) for model in models
    }
    for contrast_index, (first, second) in enumerate(itertools.combinations(models, 2)):
        deltas = {
            source: model_source_means[second][source] - model_source_means[first][source]
            for source in design["sources"]
        }
        contrast = _estimate(
            deltas,
            bootstrap_repetitions,
            bootstrap_seed + 400 + contrast_index,
        )
        equivalence_margin = 0.10
        model_contrasts.append(
            {
                "first": first,
                "second": second,
                **contrast,
                "equivalence_margin": equivalence_margin,
                "equivalent_at_95pct_interval": (
                    contrast["ci_low"] > -equivalence_margin
                    and contrast["ci_high"] < equivalence_margin
                ),
            }
        )

    power_diagnostics = []
    for contrast_index, (first, second) in enumerate(itertools.combinations(models, 2)):
        deltas = {
            source: model_source_means[second][source] - model_source_means[first][source]
            for source in design["sources"]
        }
        power_diagnostics.append(
            {
                "first": first,
                "second": second,
                **empirical_shift_power(
                    deltas,
                    minimum_effect=0.10,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed + 450 + contrast_index,
                ),
            }
        )

    model_estimates = [metric["estimate"] for metric in model_metrics]
    model_stage_spread = max(model_estimates) - min(model_estimates)

    permutation_groups = _group(rows, "model", "source_id", "wording_id")
    invariance_by_model_source: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for (model, source, _), group_rows in permutation_groups.items():
        invariance_by_model_source[(model, source)].append(
            len({_score(row) for row in group_rows}) == 1
        )

    wording_range_by_model_source: dict[tuple[str, str], float] = {}
    for model in models:
        for source in design["sources"]:
            means = []
            for wording in wordings:
                cell = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["source_id"] == source
                    and row["wording_id"] == wording
                ]
                means.append(mean(_score(row) for row in cell))
            wording_range_by_model_source[(model, source)] = max(means) - min(means)

    order_range_by_model_source: dict[tuple[str, str], float] = {}
    for model in models:
        for source in design["sources"]:
            per_wording_ranges = []
            for wording in wordings:
                cell_scores = [
                    _score(row)
                    for row in rows
                    if row["model"] == model
                    and row["source_id"] == source
                    and row["wording_id"] == wording
                ]
                per_wording_ranges.append(max(cell_scores) - min(cell_scores))
            order_range_by_model_source[(model, source)] = mean(per_wording_ranges)

    reliability = []
    for model_index, model in enumerate(models):
        wording_values = {
            source: wording_range_by_model_source[(model, source)] for source in design["sources"]
        }
        order_values = {
            source: order_range_by_model_source[(model, source)] for source in design["sources"]
        }
        reliability.append(
            {
                "model": model,
                "permutation_invariance": mean(
                    mean(invariance_by_model_source[(model, source)])
                    for source in design["sources"]
                ),
                "mean_wording_range": mean(wording_values.values()),
                "mean_order_range": mean(order_values.values()),
                "wording_range_ci": cluster_bootstrap_ci(
                    wording_values,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed + 500 + model_index,
                ),
                "order_range_ci": cluster_bootstrap_ci(
                    order_values,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed + 600 + model_index,
                ),
            }
        )

    reliability_source_metrics = {
        (model, source): {
            "permutation_invariance": mean(invariance_by_model_source[(model, source)]),
            "wording_range": wording_range_by_model_source[(model, source)],
            "order_range": order_range_by_model_source[(model, source)],
        }
        for model in models
        for source in design["sources"]
    }
    reliability_family = {}
    for first, second in itertools.combinations(models, 2):
        for metric in ("permutation_invariance", "wording_range", "order_range"):
            reliability_family[(first, second, metric)] = {
                source: (
                    reliability_source_metrics[(second, source)][metric]
                    - reliability_source_metrics[(first, source)][metric]
                )
                for source in design["sources"]
            }
    simultaneous_reliability = simultaneous_cluster_bootstrap_intervals(
        reliability_family,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 650,
    )
    reliability_contrasts = []
    for contrast_index, (key, values) in enumerate(reliability_family.items()):
        first, second, metric = key
        estimate = _estimate(
            values,
            bootstrap_repetitions,
            bootstrap_seed + 660 + contrast_index,
        )
        simultaneous_low, simultaneous_high = simultaneous_reliability[key]
        reliability_contrasts.append(
            {
                "first": first,
                "second": second,
                "metric": metric,
                **estimate,
                "simultaneous_ci_low": simultaneous_low,
                "simultaneous_ci_high": simultaneous_high,
            }
        )

    mean_wording_range = mean(metric["mean_wording_range"] for metric in reliability)
    mean_order_range = mean(metric["mean_order_range"] for metric in reliability)
    measurement_resolution = {
        "model_stage_spread": model_stage_spread,
        "mean_wording_range": mean_wording_range,
        "mean_order_range": mean_order_range,
        "wording_to_stage_ratio": (
            mean_wording_range / model_stage_spread if model_stage_spread else math.inf
        ),
        "order_to_stage_ratio": (
            mean_order_range / model_stage_spread if model_stage_spread else math.inf
        ),
    }

    rankings = []
    for wording in wordings:
        means = {
            model: mean(
                _score(row)
                for row in rows
                if row["model"] == model and row["wording_id"] == wording
            )
            for model in models
        }
        rankings.append(
            {
                "wording_id": wording,
                "means": means,
                "descending_models": sorted(models, key=lambda model: (-means[model], model)),
            }
        )

    letter_counts = []
    for (model, wording), cell_rows in sorted(_group(rows, "model", "wording_id").items()):
        counts = Counter(str(row.get("choice", "")) for row in cell_rows)
        letter_counts.append(
            {
                "model": model,
                "wording_id": wording,
                "A": counts["A"],
                "B": counts["B"],
                "C": counts["C"],
            }
        )

    return {
        "schema_version": 1,
        "design": design,
        "bootstrap": {"repetitions": bootstrap_repetitions, "seed": bootstrap_seed},
        "model_metrics": model_metrics,
        "wording_metrics": wording_metrics,
        "naive_vs_marginalized": naive_vs_marginalized,
        "category_wording_metrics": category_wording_metrics,
        "position_metrics": position_metrics,
        "position_contrasts": position_contrasts,
        "model_contrasts": model_contrasts,
        "power_diagnostics": power_diagnostics,
        "model_stage_spread": model_stage_spread,
        "reliability": reliability,
        "reliability_contrasts": reliability_contrasts,
        "measurement_resolution": measurement_resolution,
        "rankings": rankings,
        "letter_counts": letter_counts,
        "variance_components": _variance_components(rows),
    }


def _ci(metric: Mapping[str, Any], prefix: str = "") -> str:
    return (
        f"{metric[prefix + 'estimate']:.3f} "
        f"[{metric[prefix + 'ci_low']:.3f}, {metric[prefix + 'ci_high']:.3f}]"
    )


def render_factorial_markdown(
    metrics: Mapping[str, Any], protocol_note: str | None = None
) -> str:
    design = metrics["design"]
    lines = [
        "# Counterbalanced Wording × Option-Order Analysis",
        "",
    ]
    if protocol_note:
        lines.extend([f"Protocol: {protocol_note}", ""])
    lines.extend(
        [
            f"Complete cells: {design['rows']} ({len(design['models'])} models × "
            f"{len(design['sources'])} source items × "
            f"{len(design['wordings'])} wordings × "
            f"{len(design['orders'])} option orders).",
            "",
            "Intervals are 95% source-item-clustered bootstrap intervals. Scores are "
            "ordinal 0–2; the variance table is descriptive rather than an ordinal "
            "causal model.",
            "",
            "## Model-stage estimates (marginalized over wording and order)",
            "",
            "| Model | Mean [95% CI] | P(score > 0) | P(score = 2) | Rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in metrics["model_metrics"]:
        lines.append(
            f"| {metric['model']} | {_ci(metric)} | {metric['p_nonzero']:.3f} | "
            f"{metric['p_score_2']:.3f} | {metric['rows']} |"
        )
    lines.extend(
        [
            "",
            f"Marginal model-stage spread: **{metrics['model_stage_spread']:.3f}**.",
            "",
            "## Wording effects (marginalized over all six option orders)",
            "",
            "| Model | Wording | Mean [95% CI] | Delta vs original [95% CI] |",
            "|---|---|---:|---:|",
        ]
    )
    for metric in metrics["wording_metrics"]:
        lines.append(
            f"| {metric['model']} | {metric['wording_id']} | {_ci(metric)} | "
            f"{metric['delta_vs_original']:+.3f} "
            f"[{metric['delta_ci_low']:+.3f}, {metric['delta_ci_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Legacy single-layout delta versus permutation-marginalized delta",
            "",
            "The legacy layout compares each original item's source order with candidate order "
            "012 (safe=A, ambiguous=B, risky=C). The confounding shift is legacy minus "
            "permutation-marginalized delta.",
            "",
            "| Model | Wording | Legacy delta [95% CI] | Marginalized delta "
            "[95% CI] | Confounding shift [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for metric in metrics["naive_vs_marginalized"]:
        lines.append(
            f"| {metric['model']} | {metric['wording_id']} | "
            f"{metric['naive_delta']:+.3f} "
            f"[{metric['naive_ci_low']:+.3f}, {metric['naive_ci_high']:+.3f}] | "
            f"{metric['marginalized_delta']:+.3f} "
            f"[{metric['marginalized_ci_low']:+.3f}, {metric['marginalized_ci_high']:+.3f}] | "
            f"{metric['layout_confounding_shift']:+.3f} "
            f"[{metric['shift_ci_low']:+.3f}, {metric['shift_ci_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Category effects (averaged over models and all six orders)",
            "",
            "| Category | Wording | Mean | Delta vs original [95% CI] | Mean "
            "absolute source delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for metric in metrics["category_wording_metrics"]:
        lines.append(
            f"| {metric['category']} | {metric['wording_id']} | {metric['mean']:.3f} | "
            f"{metric['delta_vs_original']:+.3f} "
            f"[{metric['delta_ci_low']:+.3f}, {metric['delta_ci_high']:+.3f}] | "
            f"{metric['mean_abs_source_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Risky-option position effects (marginalized over wording and relative order)",
            "",
            "| Model | Risky option at | Mean [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for metric in metrics["position_metrics"]:
        lines.append(f"| {metric['model']} | {metric['risky_position']} | {_ci(metric)} |")
    lines.extend(
        [
            "",
            "Paired position contrasts are second position minus first position. The "
            "simultaneous intervals control the full pooled and per-model position family "
            "with a source-clustered max-|t| bootstrap.",
            "",
            "| Scope | First | Second | Difference | Simultaneous 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in metrics["position_contrasts"]:
        lines.append(
            f"| {metric['scope']} | {metric['first_position']} | "
            f"{metric['second_position']} | {metric['estimate']:+.3f} | "
            f"[{metric['simultaneous_ci_low']:+.3f}, "
            f"{metric['simultaneous_ci_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired model contrasts",
            "",
            "Positive values mean the second model scored higher than the first.",
            "",
            "The predeclared practical-equivalence margin is ±0.10. The final column "
            "uses the stricter rule that the full 95% interval must lie inside that band.",
            "",
            "| First | Second | Mean difference [95% CI] | Within ±0.10? |",
            "|---|---|---:|---:|",
        ]
    )
    for metric in metrics["model_contrasts"]:
        lines.append(
            f"| {metric['first']} | {metric['second']} | {_ci(metric)} | "
            f"{'yes' if metric['equivalent_at_95pct_interval'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic power for the predeclared 0.10-point change",
            "",
            "This empirical paired-source bootstrap reuses the observed source-level "
            "residual distribution. It is a planning diagnostic for this item pool, not a "
            "guarantee of power on a broader population.",
            "",
            "| First | Second | Sources | Minimum two-sided power | Null mean interval |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for metric in metrics["power_diagnostics"]:
        lines.append(
            f"| {metric['first']} | {metric['second']} | {metric['source_count']} | "
            f"{metric['minimum_directional_power']:.1%} | "
            f"[{metric['null_critical_low']:+.3f}, {metric['null_critical_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Measurement reliability",
            "",
            "| Model | Permutation-invariant cells | Mean wording range | Mean order range |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in metrics["reliability"]:
        lines.append(
            f"| {metric['model']} | {metric['permutation_invariance']:.1%} | "
            f"{metric['mean_wording_range']:.3f} | {metric['mean_order_range']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Paired stage contrasts in measurement reliability are shown below. The "
            "simultaneous intervals control the family of nine reliability contrasts "
            "with a source-clustered max-|t| bootstrap.",
            "",
            "| First | Second | Metric | Difference | Simultaneous 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for metric in metrics["reliability_contrasts"]:
        lines.append(
            f"| {metric['first']} | {metric['second']} | {metric['metric']} | "
            f"{metric['estimate']:+.3f} | [{metric['simultaneous_ci_low']:+.3f}, "
            f"{metric['simultaneous_ci_high']:+.3f}] |"
        )
    resolution = metrics["measurement_resolution"]
    lines.extend(
        [
            "",
            f"Across models, mean wording range was {resolution['mean_wording_range']:.3f} "
            f"and mean option-order range was {resolution['mean_order_range']:.3f}, versus a "
            f"marginal model-stage spread of {resolution['model_stage_spread']:.3f} "
            f"({resolution['wording_to_stage_ratio']:.1f}× and "
            f"{resolution['order_to_stage_ratio']:.1f}×, respectively).",
            "",
            "Stage ranking by wording (highest mean first):",
            "",
        ]
    )
    for ranking in metrics["rankings"]:
        ranking_text = " > ".join(
            f"{model} ({ranking['means'][model]:.3f})" for model in ranking["descending_models"]
        )
        lines.append(f"- {ranking['wording_id']}: {ranking_text}")
    lines.extend(
        [
            "",
            "## Descriptive variance decomposition",
            "",
            "| Component | Sum of squares | Share of total |",
            "|---|---:|---:|",
        ]
    )
    for component in metrics["variance_components"]:
        lines.append(
            f"| {component['component']} | {component['sum_squares']:.3f} | "
            f"{component['share_total']:.1%} |"
        )
    return "\n".join(lines) + "\n"
