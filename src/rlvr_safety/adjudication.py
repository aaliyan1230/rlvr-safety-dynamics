"""Apply explicit, machine-readable adjudications to blinded manual scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class AdjudicationError(ValueError):
    pass


def apply_adjudications(
    score_rows: Iterable[Mapping[str, Any]],
    adjudications: Iterable[Mapping[str, Any]],
    *,
    model_to_alias: Mapping[str, str],
    score_field: str = "instrumental_score_0_2",
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in score_rows]
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("id", "")), str(row.get("model", "")))
        if key in index:
            raise AdjudicationError(f"duplicate score row {key}")
        index[key] = row
        row["pre_adjudication_score"] = row.get(score_field, "")
        row["adjudicated"] = "false"
        row["adjudication_reason"] = ""

    seen: set[tuple[str, str]] = set()
    for adjudication in adjudications:
        model = str(adjudication.get("model", ""))
        alias = model_to_alias.get(model)
        if alias is None:
            raise AdjudicationError(f"adjudication model is absent from model key: {model}")
        key = (str(adjudication.get("id", "")), alias)
        if key in seen:
            raise AdjudicationError(f"duplicate adjudication {key}")
        seen.add(key)
        row = index.get(key)
        if row is None:
            raise AdjudicationError(f"adjudication has no matching score row: {key}")
        try:
            final_score = int(adjudication["final_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdjudicationError(f"{key}: invalid final_score") from exc
        if final_score not in (0, 1, 2):
            raise AdjudicationError(f"{key}: final_score must be 0, 1, or 2")
        reason = str(adjudication.get("reason", "")).strip()
        if not reason:
            raise AdjudicationError(f"{key}: adjudication reason is required")
        row[score_field] = str(final_score)
        row["adjudicated"] = "true"
        row["adjudication_reason"] = reason
    return rows
