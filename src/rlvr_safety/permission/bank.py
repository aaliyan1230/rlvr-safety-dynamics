"""Assemble a bank from reviewed pieces, recording exactly what was included and excluded."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from ..io import read_jsonl
from .schema import validate_bank


def merge_reviewed(
    inputs: list[Path], exclude: list[str], *, require_accepted: bool = True
) -> tuple[list[dict], dict]:
    """Concatenate bank files; drop ``exclude`` ids only if an included record supersedes each.

    With ``require_accepted`` every included record must carry an accepted two-reviewer status.
    """
    records: list[dict] = []
    audit: dict = {"inputs": {}, "excluded": {}, "included": 0}
    for path in inputs:
        rows = list(read_jsonl(path))
        audit["inputs"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(rows),
        }
        records.extend(rows)
    ids = [r["source_id"] for r in records]
    duplicated = [i for i, n in Counter(ids).items() if n > 1]
    if duplicated:
        raise ValueError(f"duplicate source_id across inputs: {duplicated}")
    unknown = set(exclude) - set(ids)
    if unknown:
        raise ValueError(f"cannot exclude unknown records: {sorted(unknown)}")
    kept = [r for r in records if r["source_id"] not in exclude]
    superseding = {r["supersedes"]: r["source_id"] for r in kept if r.get("supersedes")}
    for source_id in exclude:
        if source_id not in superseding:
            raise ValueError(f"{source_id} is excluded but no included record supersedes it")
        audit["excluded"][source_id] = {"superseded_by": superseding[source_id]}
    if require_accepted:
        pending = [r["source_id"] for r in kept if r["review"]["status"] != "accepted"]
        if pending:
            raise ValueError(f"records without accepted review: {pending}")
    validate_bank(kept)
    audit["included"] = len(kept)
    audit["statuses"] = dict(Counter(r["review"]["status"] for r in kept))
    audit["families"] = dict(Counter(r["family"] for r in kept))
    return kept, audit
