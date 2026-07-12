"""Paired comparison of two complete counterbalanced factorial protocols."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any

from .factorial_analysis import (
    FactorialAnalysisError,
    cluster_bootstrap_ci,
    validate_scored_factorial,
)

CELL_KEY_FIELDS = ("model", "source_id", "wording_id", "option_order")


class ProtocolComparisonError(ValueError):
    """Raised when two runs cannot support a strict paired comparison."""


CellKey = tuple[str, str, str, str]


def _cell_key(row: Mapping[str, Any], protocol: str) -> CellKey:
    missing = [field for field in CELL_KEY_FIELDS if not str(row.get(field, "")).strip()]
    if missing:
        raise ProtocolComparisonError(
            f"{protocol}: row missing comparison key fields {missing}: {row!r}"
        )
    return (
        str(row["model"]),
        str(row["source_id"]),
        str(row["wording_id"]),
        str(row["option_order"]),
    )


def _index_rows(
    rows: Iterable[Mapping[str, Any]],
    protocol: str,
) -> tuple[list[Mapping[str, Any]], dict[CellKey, Mapping[str, Any]]]:
    materialized = list(rows)
    indexed: dict[CellKey, Mapping[str, Any]] = {}
    for row in materialized:
        key = _cell_key(row, protocol)
        if key in indexed:
            raise ProtocolComparisonError(f"{protocol}: duplicate factorial cell {key}")
        indexed[key] = row
    return materialized, indexed


def _validate_rows(
    rows: list[Mapping[str, Any]],
    indexed: Mapping[CellKey, Mapping[str, Any]],
    protocol: str,
) -> dict[str, Any]:
    try:
        design = validate_scored_factorial(rows)
    except FactorialAnalysisError as exc:
        raise ProtocolComparisonError(f"{protocol}: {exc}") from exc

    for key, row in indexed.items():
        choice = str(row.get("choice", "")).strip()
        if choice not in {"A", "B", "C"}:
            raise ProtocolComparisonError(f"{protocol}: {key}: invalid response letter {choice!r}")
        position = str(row.get("risky_position", "")).strip()
        if position not in {"A", "B", "C"}:
            raise ProtocolComparisonError(f"{protocol}: {key}: invalid risky_position {position!r}")
        expected_position = "ABC"[key[3].index("2")]
        if position != expected_position:
            raise ProtocolComparisonError(
                f"{protocol}: {key}: risky_position {position!r} does not match "
                f"option_order (expected {expected_position!r})"
            )
    return design


def _score(row: Mapping[str, Any]) -> int:
    # validate_scored_factorial has already checked type and range.
    return int(row["instrumental_score_0_2"])


def _format_keys(keys: set[CellKey], limit: int = 5) -> str:
    sample = sorted(keys)[:limit]
    suffix = " ..." if len(keys) > limit else ""
    return f"{sample}{suffix}"


def _validate_pair(
    reference: Mapping[CellKey, Mapping[str, Any]],
    comparison: Mapping[CellKey, Mapping[str, Any]],
    reference_label: str,
    comparison_label: str,
) -> None:
    reference_keys = set(reference)
    comparison_keys = set(comparison)
    if reference_keys != comparison_keys:
        missing = reference_keys - comparison_keys
        extra = comparison_keys - reference_keys
        raise ProtocolComparisonError(
            "factorial cell keys differ: "
            f"{comparison_label} missing {len(missing)} from {reference_label} "
            f"({_format_keys(missing)}); "
            f"{comparison_label} has {len(extra)} extra ({_format_keys(extra)})"
        )

    for key in sorted(reference):
        reference_row = reference[key]
        comparison_row = comparison[key]
        for field in ("id", "risky_position"):
            reference_value = str(reference_row.get(field, ""))
            comparison_value = str(comparison_row.get(field, ""))
            if reference_value != comparison_value:
                raise ProtocolComparisonError(
                    f"cell metadata differ for {key}: {field} is {reference_value!r} in "
                    f"{reference_label} and {comparison_value!r} in {comparison_label}"
                )


def _agreement(
    keys: Sequence[CellKey],
    reference: Mapping[CellKey, Mapping[str, Any]],
    comparison: Mapping[CellKey, Mapping[str, Any]],
) -> dict[str, int | float]:
    score_matches = sum(_score(reference[key]) == _score(comparison[key]) for key in keys)
    letter_matches = sum(
        str(reference[key]["choice"]).strip() == str(comparison[key]["choice"]).strip()
        for key in keys
    )
    return {
        "rows": len(keys),
        "score_matches": score_matches,
        "score_agreement": score_matches / len(keys),
        "response_letter_matches": letter_matches,
        "response_letter_agreement": letter_matches / len(keys),
    }


def _paired_score_difference(
    keys: Sequence[CellKey],
    reference: Mapping[CellKey, Mapping[str, Any]],
    comparison: Mapping[CellKey, Mapping[str, Any]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, int | float]:
    reference_by_source: dict[str, list[int]] = defaultdict(list)
    comparison_by_source: dict[str, list[int]] = defaultdict(list)
    deltas_by_source: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        source = key[1]
        reference_score = _score(reference[key])
        comparison_score = _score(comparison[key])
        reference_by_source[source].append(reference_score)
        comparison_by_source[source].append(comparison_score)
        deltas_by_source[source].append(comparison_score - reference_score)

    source_reference_means = {
        source: mean(scores) for source, scores in reference_by_source.items()
    }
    source_comparison_means = {
        source: mean(scores) for source, scores in comparison_by_source.items()
    }
    source_deltas = {source: mean(deltas) for source, deltas in deltas_by_source.items()}
    low, high = cluster_bootstrap_ci(
        source_deltas,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    return {
        "rows": len(keys),
        "source_items": len(source_deltas),
        "reference_mean": mean(source_reference_means.values()),
        "comparison_mean": mean(source_comparison_means.values()),
        "comparison_minus_reference": mean(source_deltas.values()),
        "delta_ci_low": low,
        "delta_ci_high": high,
    }


def _position_effect_difference(
    keys: Sequence[CellKey],
    reference: Mapping[CellKey, Mapping[str, Any]],
    comparison: Mapping[CellKey, Mapping[str, Any]],
    *,
    first_position: str,
    second_position: str,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, int | float]:
    protocol_values: dict[str, dict[str, dict[str, list[int]]]] = {
        "reference": defaultdict(lambda: defaultdict(list)),
        "comparison": defaultdict(lambda: defaultdict(list)),
    }
    for key in keys:
        source = key[1]
        for label, rows in (("reference", reference), ("comparison", comparison)):
            position = str(rows[key]["risky_position"])
            protocol_values[label][source][position].append(_score(rows[key]))

    reference_effects = {}
    comparison_effects = {}
    effect_differences = {}
    sources = sorted({key[1] for key in keys})
    for source in sources:
        reference_effects[source] = mean(
            protocol_values["reference"][source][second_position]
        ) - mean(protocol_values["reference"][source][first_position])
        comparison_effects[source] = mean(
            protocol_values["comparison"][source][second_position]
        ) - mean(protocol_values["comparison"][source][first_position])
        effect_differences[source] = (
            comparison_effects[source] - reference_effects[source]
        )
    low, high = cluster_bootstrap_ci(
        effect_differences,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    return {
        "rows": len(keys),
        "source_items": len(sources),
        "reference_position_effect": mean(reference_effects.values()),
        "comparison_position_effect": mean(comparison_effects.values()),
        "comparison_minus_reference_position_effect": mean(
            effect_differences.values()
        ),
        "delta_ci_low": low,
        "delta_ci_high": high,
    }


def compare_factorial_protocols(
    reference_rows: Iterable[Mapping[str, Any]],
    comparison_rows: Iterable[Mapping[str, Any]],
    *,
    reference_label: str = "reference",
    comparison_label: str = "comparison",
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260710,
) -> dict[str, Any]:
    """Compare matched cells, with deltas defined as comparison minus reference.

    Runs must each form the same complete model x source item x wording x option-order
    design. Bootstrap intervals resample source items and retain every matched cell in a
    sampled item's relevant marginal estimate.
    """

    reference_label = reference_label.strip()
    comparison_label = comparison_label.strip()
    if not reference_label or not comparison_label:
        raise ProtocolComparisonError("protocol labels must be non-empty")
    if reference_label == comparison_label:
        raise ProtocolComparisonError("protocol labels must be distinct")
    if bootstrap_repetitions < 1:
        raise ProtocolComparisonError("bootstrap_repetitions must be at least 1")

    raw_reference, reference = _index_rows(reference_rows, reference_label)
    raw_comparison, comparison = _index_rows(comparison_rows, comparison_label)
    _validate_pair(reference, comparison, reference_label, comparison_label)
    reference_design = _validate_rows(raw_reference, reference, reference_label)
    comparison_design = _validate_rows(raw_comparison, comparison, comparison_label)

    # Identical keys imply equal factor levels; retaining both checks documents that the
    # strict factorial validator was applied independently to each protocol.
    for field in ("models", "sources", "wordings", "orders"):
        if reference_design[field] != comparison_design[field]:
            raise ProtocolComparisonError(
                f"factorial designs differ for {field}: "
                f"{reference_design[field]} != {comparison_design[field]}"
            )

    all_keys = sorted(reference)
    models: list[str] = reference_design["models"]
    wordings: list[str] = reference_design["wordings"]

    agreement_by_model = []
    for model in models:
        keys = [key for key in all_keys if key[0] == model]
        agreement_by_model.append({"model": model, **_agreement(keys, reference, comparison)})

    model_differences = []
    for model_index, model in enumerate(models):
        keys = [key for key in all_keys if key[0] == model]
        model_differences.append(
            {
                "model": model,
                **_paired_score_difference(
                    keys,
                    reference,
                    comparison,
                    bootstrap_repetitions=bootstrap_repetitions,
                    bootstrap_seed=bootstrap_seed + 100 + model_index,
                ),
            }
        )

    wording_differences = []
    for model_index, model in enumerate(models):
        for wording_index, wording in enumerate(wordings):
            keys = [key for key in all_keys if key[0] == model and key[2] == wording]
            wording_differences.append(
                {
                    "model": model,
                    "wording_id": wording,
                    **_paired_score_difference(
                        keys,
                        reference,
                        comparison,
                        bootstrap_repetitions=bootstrap_repetitions,
                        bootstrap_seed=(bootstrap_seed + 1000 + model_index * 20 + wording_index),
                    ),
                }
            )

    position_differences = []
    for model_index, model in enumerate(models):
        for position_index, position in enumerate(("A", "B", "C")):
            keys = [
                key
                for key in all_keys
                if key[0] == model and str(reference[key]["risky_position"]) == position
            ]
            position_differences.append(
                {
                    "model": model,
                    "risky_position": position,
                    **_paired_score_difference(
                        keys,
                        reference,
                        comparison,
                        bootstrap_repetitions=bootstrap_repetitions,
                        bootstrap_seed=(bootstrap_seed + 2000 + model_index * 10 + position_index),
                    ),
                }
            )

    position_effect_differences = []
    for scope_index, scope in enumerate(["all_models", *models]):
        scope_keys = [
            key for key in all_keys if scope == "all_models" or key[0] == scope
        ]
        for contrast_index, (first, second) in enumerate(
            itertools.combinations(("A", "B", "C"), 2)
        ):
            position_effect_differences.append(
                {
                    "scope": scope,
                    "first_position": first,
                    "second_position": second,
                    **_position_effect_difference(
                        scope_keys,
                        reference,
                        comparison,
                        first_position=first,
                        second_position=second,
                        bootstrap_repetitions=bootstrap_repetitions,
                        bootstrap_seed=(
                            bootstrap_seed + 3000 + scope_index * 10 + contrast_index
                        ),
                    ),
                }
            )

    return {
        "schema_version": 1,
        "reference_protocol": reference_label,
        "comparison_protocol": comparison_label,
        "delta_direction": "comparison_minus_reference",
        "cell_key_fields": list(CELL_KEY_FIELDS),
        "design": reference_design,
        "bootstrap": {
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "cluster": "source_id",
            "interval": "percentile_95",
        },
        "agreement": {
            "overall": _agreement(all_keys, reference, comparison),
            "by_model": agreement_by_model,
        },
        "protocol_score_difference": _paired_score_difference(
            all_keys,
            reference,
            comparison,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        ),
        "model_differences": model_differences,
        "wording_differences": wording_differences,
        "position_differences": position_differences,
        "position_effect_differences": position_effect_differences,
    }


def _difference_cell(row: Mapping[str, Any]) -> str:
    return (
        f"{row['reference_mean']:.3f} | {row['comparison_mean']:.3f} | "
        f"{row['comparison_minus_reference']:+.3f} "
        f"[{row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f}]"
    )


def render_protocol_comparison_markdown(
    metrics: Mapping[str, Any],
    title: str = "Factorial Protocol Comparison",
) -> str:
    """Render the machine-readable comparison as a compact audit report."""

    reference = metrics["reference_protocol"]
    comparison = metrics["comparison_protocol"]
    design = metrics["design"]
    lines = [
        f"# {title}",
        "",
        f"Matched cells: {design['rows']}. Deltas are **{comparison} minus {reference}**. "
        "Intervals are paired 95% source-item-clustered bootstrap intervals.",
        "",
        "## Cell-level agreement",
        "",
        "Score agreement compares semantic 0-2 scores; response-letter agreement compares "
        "literal A/B/C outputs.",
        "",
        "| Scope | Rows | Score agreement | Response-letter agreement |",
        "|---|---:|---:|---:|",
    ]
    overall_agreement = metrics["agreement"]["overall"]
    lines.append(
        f"| All models | {overall_agreement['rows']} | "
        f"{overall_agreement['score_agreement']:.1%} | "
        f"{overall_agreement['response_letter_agreement']:.1%} |"
    )
    for row in metrics["agreement"]["by_model"]:
        lines.append(
            f"| {row['model']} | {row['rows']} | {row['score_agreement']:.1%} | "
            f"{row['response_letter_agreement']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Overall paired protocol score difference",
            "",
            f"| Scope | {reference} mean | {comparison} mean | Difference [95% CI] |",
            "|---|---:|---:|---:|",
            f"| All cells | {_difference_cell(metrics['protocol_score_difference'])} |",
            "",
            "## Model-marginal score differences",
            "",
            f"| Model | {reference} mean | {comparison} mean | Difference [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in metrics["model_differences"]:
        lines.append(f"| {row['model']} | {_difference_cell(row)} |")

    lines.extend(
        [
            "",
            "## Wording-marginal score differences",
            "",
            f"| Model | Wording | {reference} mean | {comparison} mean | Difference [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in metrics["wording_differences"]:
        lines.append(f"| {row['model']} | {row['wording_id']} | {_difference_cell(row)} |")

    lines.extend(
        [
            "",
            "## Risky-position-marginal score differences",
            "",
            f"| Model | Risky position | {reference} mean | {comparison} mean | "
            "Difference [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in metrics["position_differences"]:
        lines.append(f"| {row['model']} | {row['risky_position']} | {_difference_cell(row)} |")
    lines.extend(
        [
            "",
            "## Change in risky-position effects between protocols",
            "",
            "Each position effect is second position minus first position. The final "
            f"column is the {comparison} effect minus the {reference} effect.",
            "",
            f"| Scope | First | Second | {reference} effect | {comparison} effect | "
            "Effect difference [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["position_effect_differences"]:
        lines.append(
            f"| {row['scope']} | {row['first_position']} | {row['second_position']} | "
            f"{row['reference_position_effect']:+.3f} | "
            f"{row['comparison_position_effect']:+.3f} | "
            f"{row['comparison_minus_reference_position_effect']:+.3f} "
            f"[{row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f}] |"
        )
    return "\n".join(lines) + "\n"
