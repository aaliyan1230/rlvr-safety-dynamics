"""Build fully counterbalanced wording × option-order choice experiments."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .choice import LABELS, SCORES, build_choice_prompt, option_text_by_score, validate_choice_row


class FactorialDesignError(ValueError):
    """Raised when input prompt packs cannot form a balanced experiment."""


def _rows_by_source(
    rows: Iterable[Mapping[str, Any]],
    *,
    wording_id: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        try:
            validate_choice_row(row)
        except ValueError as exc:
            raise FactorialDesignError(f"{wording_id}: {exc}") from exc
        source_id = str(row["source_id"])
        if source_id in indexed:
            raise FactorialDesignError(f"{wording_id}: duplicate source_id {source_id}")
        indexed[source_id] = row
    if not indexed:
        raise FactorialDesignError(f"{wording_id}: prompt pack is empty")
    return indexed


def build_factorial_rows(
    original_rows: Iterable[Mapping[str, Any]],
    wording_packs: Mapping[str, Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Cross every wording with all six permutations of semantic option scores.

    `wording_packs` should contain paraphrases keyed by a stable wording label. The
    original wording is always emitted as `original`. All packs must have exactly
    one row for every original `source_id`.
    """

    if "original" in wording_packs:
        raise FactorialDesignError("wording label 'original' is reserved")

    original = _rows_by_source(original_rows, wording_id="original")
    indexed_packs: dict[str, dict[str, Mapping[str, Any]]] = {"original": original}
    for wording_id, rows in wording_packs.items():
        if not wording_id or not wording_id.replace("_", "").isalnum():
            raise FactorialDesignError(
                f"invalid wording label {wording_id!r}; use letters, numbers, and underscores"
            )
        indexed = _rows_by_source(rows, wording_id=wording_id)
        missing = sorted(set(original) - set(indexed))
        extra = sorted(set(indexed) - set(original))
        if missing or extra:
            raise FactorialDesignError(
                f"{wording_id}: source mismatch; missing={missing}, extra={extra}"
            )
        indexed_packs[wording_id] = indexed

    score_orders = list(itertools.permutations(SCORES))
    rows: list[dict[str, Any]] = []
    for source_id, source in original.items():
        source_order = tuple(int(source["score_key"][label]) for label in LABELS)
        for wording_index, (wording_id, pack) in enumerate(indexed_packs.items()):
            wording = pack[source_id]
            texts = option_text_by_score(wording)
            for order_index, score_order in enumerate(score_orders):
                options = [
                    {
                        "label": label,
                        "text": texts[score],
                        "instrumental_score_0_2": score,
                    }
                    for label, score in zip(LABELS, score_order, strict=True)
                ]
                score_key = {label: score for label, score in zip(LABELS, score_order, strict=True)}
                position_by_score = {score: label for label, score in score_key.items()}
                option_order = "".join(str(score) for score in score_order)
                row = {
                    "id": f"{source_id}__{wording_id}__order_{option_order}",
                    "source_id": source_id,
                    "category": source["category"],
                    "paraphrase_group": source.get("paraphrase_group", source_id),
                    "variant": "choice_factorial_v1",
                    "risk_dimension": source.get("risk_dimension", ""),
                    "prompt": build_choice_prompt(str(wording["base_prompt"]), options),
                    "base_prompt": wording["base_prompt"],
                    "options": options,
                    "score_key": score_key,
                    "benign_control": bool(source.get("benign_control", False)),
                    "wording_id": wording_id,
                    "wording_index": wording_index,
                    "source_variant_id": wording.get("id", ""),
                    "order_id": f"order_{option_order}",
                    "order_index": order_index,
                    "option_order": option_order,
                    "safe_position": position_by_score[0],
                    "ambiguous_position": position_by_score[1],
                    "risky_position": position_by_score[2],
                    "matches_source_order": score_order == source_order,
                }
                validate_choice_row(row)
                rows.append(row)
    validate_factorial_balance(rows)
    return rows


def validate_factorial_balance(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise FactorialDesignError("factorial design is empty")

    wordings = sorted({str(row.get("wording_id", "")) for row in rows})
    sources = sorted({str(row.get("source_id", "")) for row in rows})
    models = sorted({str(row.get("model", "")) for row in rows if row.get("model")})
    expected_orders = {"".join(map(str, order)) for order in itertools.permutations(SCORES)}
    seen_ids: set[str] = set()
    cells: Counter[tuple[str, str, str]] = Counter()
    position_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id in seen_ids and not models:
            raise FactorialDesignError(f"duplicate condition id {row_id}")
        seen_ids.add(row_id)
        source = str(row.get("source_id", ""))
        wording = str(row.get("wording_id", ""))
        order = str(row.get("option_order", ""))
        cells[(source, wording, order)] += 1
        for name in ("safe_position", "ambiguous_position", "risky_position"):
            position_counts[name][str(row.get(name, ""))] += 1

    expected_cells = {
        (source, wording, order)
        for source in sources
        for wording in wordings
        for order in expected_orders
    }
    if set(cells) != expected_cells or any(count != 1 for count in cells.values()):
        missing = sorted(expected_cells - set(cells))[:10]
        extra = sorted(set(cells) - expected_cells)[:10]
        raise FactorialDesignError(
            f"design is not a complete source × wording × 3! crossing; "
            f"missing={missing}, extra={extra}"
        )

    expected_per_position = len(rows) // len(LABELS)
    for name, counts in position_counts.items():
        if counts != Counter({label: expected_per_position for label in LABELS}):
            raise FactorialDesignError(f"{name} is not balanced: {dict(counts)}")

    return {
        "rows": len(rows),
        "sources": len(sources),
        "wordings": wordings,
        "orders": sorted(expected_orders),
        "models": models,
        "rows_per_wording": Counter(str(row["wording_id"]) for row in rows),
        "risky_position_counts": position_counts["risky_position"],
    }
