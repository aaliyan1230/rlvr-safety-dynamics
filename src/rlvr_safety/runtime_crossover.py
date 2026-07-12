"""Analysis for the reciprocal historical-layout inference-runtime crossover."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any

from .factorial_analysis import cluster_bootstrap_ci


class RuntimeCrossoverError(ValueError):
    """Raised when the two runtime panels cannot support a paired crossover."""


CellKey = tuple[str, str, str]
WORDINGS = ("original", "p1", "p2")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _score(row: Mapping[str, Any], label: str, key: CellKey) -> int:
    try:
        value = int(row.get("instrumental_score_0_2", ""))
    except (TypeError, ValueError) as exc:
        raise RuntimeCrossoverError(f"{label}: {key}: invalid score") from exc
    if value not in (0, 1, 2):
        raise RuntimeCrossoverError(f"{label}: {key}: score outside 0..2")
    return value


def _index_panel(
    rows: Iterable[Mapping[str, Any]], label: str
) -> tuple[dict[CellKey, Mapping[str, Any]], dict[str, Any]]:
    indexed: dict[CellKey, Mapping[str, Any]] = {}
    for row in rows:
        missing = [
            field
            for field in ("id", "model", "source_id", "wording_id", "option_order")
            if not str(row.get(field, "")).strip()
        ]
        if missing:
            raise RuntimeCrossoverError(f"{label}: row missing fields {missing}")
        key = (str(row["model"]), str(row["source_id"]), str(row["wording_id"]))
        if key in indexed:
            raise RuntimeCrossoverError(f"{label}: duplicate cell {key}")
        wording = key[2]
        if wording not in WORDINGS:
            raise RuntimeCrossoverError(f"{label}: {key}: unexpected wording")
        order = str(row["option_order"])
        if sorted(order) != ["0", "1", "2"] or len(order) != 3:
            raise RuntimeCrossoverError(f"{label}: {key}: invalid option order {order!r}")
        if wording == "original" and not _as_bool(row.get("matches_source_order")):
            raise RuntimeCrossoverError(
                f"{label}: {key}: original must use its historical source order"
            )
        if wording in {"p1", "p2"} and order != "012":
            raise RuntimeCrossoverError(f"{label}: {key}: candidate must use historical order 012")
        risky_position = str(row.get("risky_position", ""))
        expected_position = "ABC"[order.index("2")]
        if risky_position != expected_position:
            raise RuntimeCrossoverError(
                f"{label}: {key}: risky position does not match option order"
            )
        if str(row.get("choice", "")).strip() not in {"A", "B", "C"}:
            raise RuntimeCrossoverError(f"{label}: {key}: invalid response letter")
        if _as_bool(row.get("malformed")) or _as_bool(row.get("verbosity_confounded")):
            raise RuntimeCrossoverError(f"{label}: {key}: malformed or capped cell")
        _score(row, label, key)
        indexed[key] = row

    if not indexed:
        raise RuntimeCrossoverError(f"{label}: no rows")
    models = sorted({key[0] for key in indexed})
    sources = sorted({key[1] for key in indexed})
    wordings = sorted({key[2] for key in indexed})
    if len(models) != 3 or len(sources) != 24 or set(wordings) != set(WORDINGS):
        raise RuntimeCrossoverError(
            f"{label}: expected 3 models × 24 sources × original/p1/p2; "
            f"observed {len(models)} × {len(sources)} × {wordings}"
        )
    expected = {
        (model, source, wording) for model in models for source in sources for wording in WORDINGS
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))[:5]
        extra = sorted(set(indexed) - expected)[:5]
        raise RuntimeCrossoverError(
            f"{label}: incomplete crossing; missing={missing}, extra={extra}"
        )
    return indexed, {
        "rows": len(indexed),
        "models": models,
        "sources": sources,
        "wordings": list(WORDINGS),
    }


def _validate_pair(
    stage: Mapping[CellKey, Mapping[str, Any]],
    paraphrase: Mapping[CellKey, Mapping[str, Any]],
) -> None:
    if set(stage) != set(paraphrase):
        raise RuntimeCrossoverError("runtime panels have different cell keys")
    for key in stage:
        for field in ("id", "option_order", "risky_position"):
            if str(stage[key].get(field, "")) != str(paraphrase[key].get(field, "")):
                raise RuntimeCrossoverError(
                    f"runtime panels differ for {key}: metadata field {field}"
                )


def _agreement(
    keys: Sequence[CellKey],
    stage: Mapping[CellKey, Mapping[str, Any]],
    paraphrase: Mapping[CellKey, Mapping[str, Any]],
) -> dict[str, int | float]:
    score_matches = sum(
        _score(stage[key], "stage_runtime", key)
        == _score(paraphrase[key], "paraphrase_runtime", key)
        for key in keys
    )
    letter_matches = sum(
        str(stage[key]["choice"]) == str(paraphrase[key]["choice"]) for key in keys
    )
    return {
        "rows": len(keys),
        "score_matches": score_matches,
        "score_agreement": score_matches / len(keys),
        "response_letter_matches": letter_matches,
        "response_letter_agreement": letter_matches / len(keys),
    }


def _estimate(values_by_source: Mapping[str, float], repetitions: int, seed: int) -> dict:
    low, high = cluster_bootstrap_ci(values_by_source, repetitions=repetitions, seed=seed)
    return {
        "estimate": mean(values_by_source.values()),
        "ci_low": low,
        "ci_high": high,
    }


def _runtime_delta(
    keys: Sequence[CellKey],
    stage: Mapping[CellKey, Mapping[str, Any]],
    paraphrase: Mapping[CellKey, Mapping[str, Any]],
    repetitions: int,
    seed: int,
) -> dict:
    stage_by_source: dict[str, list[int]] = defaultdict(list)
    paraphrase_by_source: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        stage_by_source[key[1]].append(_score(stage[key], "stage_runtime", key))
        paraphrase_by_source[key[1]].append(_score(paraphrase[key], "paraphrase_runtime", key))
    stage_means = {source: mean(values) for source, values in stage_by_source.items()}
    paraphrase_means = {source: mean(values) for source, values in paraphrase_by_source.items()}
    deltas = {source: paraphrase_means[source] - stage_means[source] for source in stage_means}
    estimate = _estimate(deltas, repetitions, seed)
    return {
        "rows": len(keys),
        "source_items": len(deltas),
        "stage_runtime_mean": mean(stage_means.values()),
        "paraphrase_runtime_mean": mean(paraphrase_means.values()),
        "paraphrase_runtime_minus_stage_runtime": estimate["estimate"],
        "delta_ci_low": estimate["ci_low"],
        "delta_ci_high": estimate["ci_high"],
    }


def _pack_effects_by_source(
    panel: Mapping[CellKey, Mapping[str, Any]],
    models: Sequence[str],
    sources: Sequence[str],
    candidate: str,
) -> dict[str, float]:
    return {
        source: mean(
            _score(panel[(model, source, candidate)], "runtime", (model, source, candidate))
            - _score(panel[(model, source, "original")], "runtime", (model, source, "original"))
            for model in models
        )
        for source in sources
    }


def analyze_runtime_crossover(
    stage_rows: Iterable[Mapping[str, Any]],
    paraphrase_rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260710,
) -> dict[str, Any]:
    """Analyze the two matched 216-cell historical-layout runtime panels."""

    if bootstrap_repetitions < 1:
        raise RuntimeCrossoverError("bootstrap repetitions must be positive")
    stage, design = _index_panel(stage_rows, "stage_runtime")
    paraphrase, comparison_design = _index_panel(paraphrase_rows, "paraphrase_runtime")
    if design != comparison_design:
        raise RuntimeCrossoverError("runtime panel designs differ")
    _validate_pair(stage, paraphrase)
    all_keys = sorted(stage)
    models: list[str] = design["models"]
    sources: list[str] = design["sources"]

    agreement_by_model = [
        {
            "model": model,
            **_agreement([key for key in all_keys if key[0] == model], stage, paraphrase),
        }
        for model in models
    ]
    agreement_by_wording = [
        {
            "wording_id": wording,
            **_agreement([key for key in all_keys if key[2] == wording], stage, paraphrase),
        }
        for wording in WORDINGS
    ]

    runtime_deltas = []
    scopes = [("all_models", models), *((model, [model]) for model in models)]
    for scope_index, (scope, scope_models) in enumerate(scopes):
        runtime_deltas.append(
            {
                "scope": scope,
                "wording_id": "all_wordings",
                **_runtime_delta(
                    [key for key in all_keys if key[0] in scope_models],
                    stage,
                    paraphrase,
                    bootstrap_repetitions,
                    bootstrap_seed + 100 + scope_index,
                ),
            }
        )
        for wording_index, wording in enumerate(WORDINGS):
            runtime_deltas.append(
                {
                    "scope": scope,
                    "wording_id": wording,
                    **_runtime_delta(
                        [key for key in all_keys if key[0] in scope_models and key[2] == wording],
                        stage,
                        paraphrase,
                        bootstrap_repetitions,
                        bootstrap_seed + 200 + scope_index * 10 + wording_index,
                    ),
                }
            )

    within_runtime_effects = []
    crossover_interactions = []
    for scope_index, (scope, scope_models) in enumerate(scopes):
        for candidate_index, candidate in enumerate(("p1", "p2")):
            stage_effects = _pack_effects_by_source(stage, scope_models, sources, candidate)
            paraphrase_effects = _pack_effects_by_source(
                paraphrase, scope_models, sources, candidate
            )
            for runtime_index, (runtime, effects) in enumerate(
                (
                    ("stage_runtime", stage_effects),
                    ("paraphrase_runtime", paraphrase_effects),
                )
            ):
                within_runtime_effects.append(
                    {
                        "runtime": runtime,
                        "scope": scope,
                        "wording_id": candidate,
                        "contrast": "candidate_minus_original",
                        **_estimate(
                            effects,
                            bootstrap_repetitions,
                            bootstrap_seed
                            + 500
                            + scope_index * 20
                            + candidate_index * 2
                            + runtime_index,
                        ),
                    }
                )
            interaction = {
                source: paraphrase_effects[source] - stage_effects[source] for source in sources
            }
            crossover_interactions.append(
                {
                    "scope": scope,
                    "wording_id": candidate,
                    "contrast": (
                        "(candidate_minus_original)_paraphrase_runtime_minus_"
                        "(candidate_minus_original)_stage_runtime"
                    ),
                    **_estimate(
                        interaction,
                        bootstrap_repetitions,
                        bootstrap_seed + 800 + scope_index * 10 + candidate_index,
                    ),
                }
            )

    return {
        "schema_version": 1,
        "design": design,
        "runtime_delta_direction": "paraphrase_runtime_minus_stage_runtime",
        "crossover_direction": (
            "(candidate_minus_original)_paraphrase_runtime_minus_"
            "(candidate_minus_original)_stage_runtime"
        ),
        "bootstrap": {
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "cluster": "source_id",
            "interval": "percentile_95",
        },
        "agreement": {
            "overall": _agreement(all_keys, stage, paraphrase),
            "by_model": agreement_by_model,
            "by_wording": agreement_by_wording,
        },
        "runtime_deltas": runtime_deltas,
        "within_runtime_pack_effects": within_runtime_effects,
        "crossover_interactions": crossover_interactions,
    }


def render_runtime_crossover_markdown(
    metrics: Mapping[str, Any], title: str = "Historical Runtime Crossover"
) -> str:
    lines = [
        f"# {title}",
        "",
        "The two panels contain identical historical layouts. Runtime deltas are "
        "P1/P2-image runtime minus stage-image runtime. Intervals resample the 24 "
        "source items.",
        "",
        "## Runtime agreement",
        "",
        "| Scope | Rows | Score agreement | Response-letter agreement |",
        "|---|---:|---:|---:|",
    ]
    overall = metrics["agreement"]["overall"]
    lines.append(
        f"| all cells | {overall['rows']} | {overall['score_agreement']:.1%} | "
        f"{overall['response_letter_agreement']:.1%} |"
    )
    for row in metrics["agreement"]["by_model"]:
        lines.append(
            f"| {row['model']} | {row['rows']} | {row['score_agreement']:.1%} | "
            f"{row['response_letter_agreement']:.1%} |"
        )
    for row in metrics["agreement"]["by_wording"]:
        lines.append(
            f"| wording={row['wording_id']} | {row['rows']} | "
            f"{row['score_agreement']:.1%} | "
            f"{row['response_letter_agreement']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Runtime score deltas",
            "",
            "| Scope | Wording | Stage-runtime mean | P1/P2-runtime mean | Delta [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in metrics["runtime_deltas"]:
        lines.append(
            f"| {row['scope']} | {row['wording_id']} | "
            f"{row['stage_runtime_mean']:.3f} | "
            f"{row['paraphrase_runtime_mean']:.3f} | "
            f"{row['paraphrase_runtime_minus_stage_runtime']:+.3f} "
            f"[{row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Candidate-minus-original effects within each runtime",
            "",
            "| Runtime | Scope | Candidate | Effect [95% CI] |",
            "|---|---|---|---:|",
        ]
    )
    for row in metrics["within_runtime_pack_effects"]:
        lines.append(
            f"| {row['runtime']} | {row['scope']} | {row['wording_id']} | "
            f"{row['estimate']:+.3f} [{row['ci_low']:+.3f}, "
            f"{row['ci_high']:+.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Runtime × pack difference-in-differences",
            "",
            "Positive values mean the candidate-minus-original shift is larger under "
            "the P1/P2 historical runtime.",
            "",
            "| Scope | Candidate | Difference-in-differences [95% CI] |",
            "|---|---|---:|",
        ]
    )
    for row in metrics["crossover_interactions"]:
        lines.append(
            f"| {row['scope']} | {row['wording_id']} | {row['estimate']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |"
        )
    return "\n".join(lines) + "\n"
