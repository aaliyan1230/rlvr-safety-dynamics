"""Strict, non-scientific feasibility checks for the Tülu endpoint pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from .choice import SYSTEM_PROMPT


class PilotValidationError(ValueError):
    """Raised when a pilot artifact fails a declared feasibility gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotValidationError(message)


def _true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _iso8601(value: Any, field: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise PilotValidationError(f"invalid {field}: {value!r}") from exc


def analyze_endpoint_pilot(
    score_rows: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate endpoint feasibility without interpreting the behavioral scores."""

    rows = [dict(row) for row in score_rows]
    expected_rows = int(config["gates"]["expected_rows"])
    expected_per_model = int(config["design"]["conditions_per_checkpoint"])
    expected_sources = int(config["design"]["source_items"])
    model_specs = {str(model["id"]): dict(model) for model in config["models"]}

    _require(config.get("status") == "throughput_and_format_feasibility_only", "bad status")
    _require(config["generation"].get("system_prompt") == SYSTEM_PROMPT, "system prompt mismatch")
    _require(config["gates"].get("scientific_interpretation_allowed") is False, "bad gate")
    _require(summary.get("scientific_interpretation_allowed") is False, "bad summary gate")
    _require(summary.get("had_error") is False, "run_summary reports an error")
    _require(not summary.get("errors"), "run_summary contains errors")
    _require(len(rows) == expected_rows, f"expected {expected_rows} score rows, found {len(rows)}")
    _require(int(summary.get("rows", -1)) == expected_rows, "run_summary row mismatch")
    _require(int(summary.get("expected_rows", -1)) == expected_rows, "summary design mismatch")

    keys = [(str(row.get("model", "")), str(row.get("id", ""))) for row in rows]
    _require(all(model and row_id for model, row_id in keys), "blank model or id")
    _require(len(set(keys)) == len(keys), "duplicate model/id cells")
    _require({model for model, _ in keys} == set(model_specs), "model coverage mismatch")

    expected_wordings = set(config["prompt_filter"]["wording_ids"])
    expected_orders = set(config["prompt_filter"]["option_orders"])
    _require({str(row.get("wording_id")) for row in rows} == expected_wordings, "wording mismatch")
    _require({str(row.get("option_order")) for row in rows} == expected_orders, "order mismatch")

    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row["model"])
        spec = model_specs[model]
        rows_by_model[model].append(row)
        _require(row.get("model_repo") == spec["name"], f"{model}: row repository mismatch")
        _require(
            row.get("model_revision_requested") == spec["revision"],
            f"{model}: row requested revision mismatch",
        )
        _require(
            row.get("model_revision_resolved") == spec["revision"],
            f"{model}: row resolved revision mismatch",
        )
        _require(
            str(row.get("generation_seed")) == str(config["generation"]["seed"]),
            f"{model}: row generation seed mismatch",
        )
        _require(str(row.get("parse_status")) == "strict", f"{model}: non-strict parse")
        _require(not _true(row.get("malformed")), f"{model}: malformed row")
        _require(not _true(row.get("verbosity_confounded")), f"{model}: capped row")
        try:
            score = int(str(row["instrumental_score_0_2"]))
        except (KeyError, ValueError) as exc:
            raise PilotValidationError(f"{model}: invalid score") from exc
        _require(score in {0, 1, 2}, f"{model}: score outside 0-2")

    metadata_by_model = {
        str(model.get("model", "")): dict(model) for model in summary.get("models", [])
    }
    _require(set(metadata_by_model) == set(model_specs), "model metadata coverage mismatch")
    _require(
        len({str(row["source_id"]) for row in rows}) == expected_sources,
        "source coverage mismatch",
    )

    semantic_cells = None
    for model_id, model_rows in rows_by_model.items():
        observed = {
            (str(row["source_id"]), str(row["wording_id"]), str(row["option_order"]))
            for row in model_rows
        }
        _require(len(observed) == expected_per_model, f"{model_id}: duplicate semantic cells")
        if semantic_cells is None:
            semantic_cells = observed
        else:
            _require(observed == semantic_cells, f"{model_id}: prompt coverage differs")

    memory_limit = int(float(config["gates"]["max_gpu_memory_gb"]) * 1024**3)
    expected_mask_mode = str(config["generation"]["attention_mask_mode"])
    model_results = []
    for model_id, spec in model_specs.items():
        model_rows = rows_by_model[model_id]
        metadata = metadata_by_model[model_id]
        _require(len(model_rows) == expected_per_model, f"{model_id}: row count mismatch")
        _require(metadata.get("rows") == expected_per_model, f"{model_id}: metadata row mismatch")
        _require(metadata.get("model_repo") == spec["name"], f"{model_id}: repository mismatch")
        _require(
            metadata.get("model_revision_requested") == spec["revision"],
            f"{model_id}: requested revision mismatch",
        )
        _require(
            metadata.get("model_revision_resolved") == spec["revision"],
            f"{model_id}: resolved revision mismatch",
        )
        _require(metadata.get("model_is_quantized") is True, f"{model_id}: not quantized")
        _require(
            metadata.get("attention_mask_mode") == expected_mask_mode,
            f"{model_id}: attention-mask mode mismatch",
        )
        quant = metadata.get("quantization_config_resolved", {})
        _require(quant.get("load_in_4bit") is True, f"{model_id}: 4-bit load not resolved")
        _require(quant.get("bnb_4bit_quant_type") == "nf4", f"{model_id}: not NF4")
        _require(quant.get("bnb_4bit_use_double_quant") is True, f"{model_id}: no double quant")
        _require(quant.get("bnb_4bit_compute_dtype") == "float16", f"{model_id}: wrong dtype")
        peak = int(metadata.get("peak_gpu_memory_bytes", -1))
        _require(0 < peak <= memory_limit, f"{model_id}: GPU memory gate failed")
        model_results.append(
            {
                "model": model_id,
                "step": int(spec["step"]),
                "rows": len(model_rows),
                "strict_parse_rows": sum(row["parse_status"] == "strict" for row in model_rows),
                "malformed_rows": sum(_true(row["malformed"]) for row in model_rows),
                "token_capped_rows": sum(
                    _true(row["verbosity_confounded"]) for row in model_rows
                ),
                "score_counts_descriptive_only": {
                    str(score): count
                    for score, count in sorted(
                        Counter(int(row["instrumental_score_0_2"]) for row in model_rows).items()
                    )
                },
                "model_repo": spec["name"],
                "revision": spec["revision"],
                "peak_gpu_memory_bytes": peak,
                "peak_gpu_memory_gib": peak / 1024**3,
            }
        )

    expected_versions = {
        key: str(value)
        for key, value in config["runtime"].items()
        if key != "accelerator"
    }
    observed_versions = {
        key: str(value) for key, value in run_metadata.get("package_versions", {}).items()
    }
    for package, expected in expected_versions.items():
        _require(observed_versions.get(package) == expected, f"{package}: runtime mismatch")
    gpus = list(run_metadata.get("gpus", []))
    _require(len(gpus) == 2 and all("T4" in str(gpu) for gpu in gpus), "T4x2 gate failed")

    started = _iso8601(run_metadata.get("started_utc"), "started_utc")
    completed = _iso8601(summary.get("completed_utc"), "completed_utc")
    wall_seconds = (completed - started).total_seconds()
    _require(wall_seconds > 0, "non-positive runtime")

    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "feasibility_passed": True,
        "scientific_interpretation_allowed": False,
        "interpretation": (
            "This pilot validates checkpoint access, format compliance, memory, and "
            "throughput only. "
            "Its two-order endpoint scores are not a marginalized safety estimate."
        ),
        "rows": len(rows),
        "unique_cells": len(set(keys)),
        "source_items": len({str(row["source_id"]) for row in rows}),
        "wordings": sorted(expected_wordings),
        "option_orders": sorted(expected_orders),
        "attention_mask_mode": expected_mask_mode,
        "gpus": gpus,
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "wall_clock_seconds": wall_seconds,
        "runtime_versions": expected_versions,
        "memory_limit_bytes": memory_limit,
        "models": sorted(model_results, key=lambda item: item["step"]),
    }


def render_endpoint_pilot_markdown(metrics: Mapping[str, Any]) -> str:
    """Render a feasibility report that explicitly withholds scientific interpretation."""

    lines = [
        "# Tülu endpoint feasibility pilot",
        "",
        "**Gate result: PASS. Scientific interpretation: not allowed.**",
        "",
        str(metrics["interpretation"]),
        "",
        f"Wall-clock runtime: {float(metrics['wall_clock_seconds']) / 60:.1f} minutes on "
        f"{', '.join(str(gpu) for gpu in metrics['gpus'])}.",
        "",
        "| Endpoint | Rows | Strict parses | Malformed | Capped | Peak GPU memory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in metrics["models"]:
        lines.append(
            f"| {model['model']} | {model['rows']} | {model['strict_parse_rows']} | "
            f"{model['malformed_rows']} | {model['token_capped_rows']} | "
            f"{model['peak_gpu_memory_gib']:.2f} GiB |"
        )
    lines.extend(
        [
            "",
            f"Both immutable revisions resolved exactly. Generation used "
            f"`{metrics['attention_mask_mode']}` attention masking and the pinned NF4/fp16/"
            "double-quantization protocol.",
            "",
            "The pilot includes only the original wording and orders 012/210. Endpoint score "
            "differences are deliberately not reported as evidence about GRPO or safety.",
        ]
    )
    return "\n".join(lines) + "\n"
