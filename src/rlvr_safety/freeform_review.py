"""Blinded AI-review utilities for the frozen Tülu free-form panel."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


LABELS = (0, 1, 2, None)


def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("model", "")), str(row.get("id", ""))


def blind_id(row: Mapping[str, Any], reviewer: str) -> str:
    payload = f"tulu_freeform_v1|{reviewer}|{row_key(row)[0]}|{row_key(row)[1]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def quadratic_weighted_kappa(first: list[int], second: list[int]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("kappa requires equal non-empty score lists")
    if any(value not in (0, 1, 2) for value in first + second):
        raise ValueError("kappa scores must be 0, 1, or 2")
    observed = Counter(zip(first, second))
    first_counts = Counter(first)
    second_counts = Counter(second)
    total = len(first)
    weighted_observed = sum(
        ((a - b) ** 2 / 4) * observed[(a, b)] / total for a in range(3) for b in range(3)
    )
    weighted_expected = sum(
        ((a - b) ** 2 / 4)
        * (first_counts[a] / total)
        * (second_counts[b] / total)
        for a in range(3)
        for b in range(3)
    )
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else 0.0
    return 1 - weighted_observed / weighted_expected


def agreement_summary(
    reviewer_a: Iterable[Mapping[str, Any]], reviewer_b: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    a_rows = {row_key(row): dict(row) for row in reviewer_a}
    b_rows = {row_key(row): dict(row) for row in reviewer_b}
    if len(a_rows) != len(b_rows) or set(a_rows) != set(b_rows) or not a_rows:
        raise ValueError("reviewer packets must contain identical unique cells")
    exact = 0
    comparable_a = []
    comparable_b = []
    disagreements = []
    for key in sorted(a_rows):
        a = a_rows[key]
        b = b_rows[key]
        a_label = a.get("score_0_2") if a.get("judgeable") else None
        b_label = b.get("score_0_2") if b.get("judgeable") else None
        if a_label == b_label:
            exact += 1
        else:
            disagreements.append(key)
        if a_label in (0, 1, 2) and b_label in (0, 1, 2):
            comparable_a.append(int(a_label))
            comparable_b.append(int(b_label))
    return {
        "rows": len(a_rows),
        "exact_agreement": exact / len(a_rows),
        "quadratic_weighted_kappa": (
            quadratic_weighted_kappa(comparable_a, comparable_b) if comparable_a else None
        ),
        "adjudication_rows": len(disagreements),
        "adjudication_rate": len(disagreements) / len(a_rows),
        "disagreement_keys": disagreements,
    }


def apply_freeform_adjudications(
    reviewer_a: Iterable[Mapping[str, Any]],
    reviewer_b: Iterable[Mapping[str, Any]],
    adjudications: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    a_rows = {row_key(row): dict(row) for row in reviewer_a}
    b_rows = {row_key(row): dict(row) for row in reviewer_b}
    summary = agreement_summary(a_rows.values(), b_rows.values())
    required = set(map(tuple, summary["disagreement_keys"]))
    decisions = {row_key(row): dict(row) for row in adjudications}
    if set(decisions) != required:
        raise ValueError("adjudications must cover every and only disagreement")
    final = []
    for key in sorted(a_rows):
        a = a_rows[key]
        b = b_rows[key]
        decision = decisions.get(key)
        if decision:
            score = decision.get("score_0_2")
            judgeable = bool(decision.get("judgeable"))
            censor_reason = decision.get("censor_reason", "other")
        else:
            score = a.get("score_0_2") if a.get("judgeable") else None
            judgeable = bool(a.get("judgeable"))
            censor_reason = a.get("censor_reason", "none")
        final.append(
            {
                "model": key[0],
                "id": key[1],
                "source_id": a["source_id"],
                "checkpoint": a["checkpoint"],
                "category": a["category"],
                "score_0_2": score,
                "judgeable": judgeable,
                "censor_reason": censor_reason,
                "adjudicated": decision is not None,
            }
        )
    return final
