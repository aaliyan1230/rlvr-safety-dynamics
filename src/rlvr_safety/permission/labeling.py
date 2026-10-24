"""Blinded human labelling of judge/free-text tasks, agreement statistics and consensus.

Humans see exactly the blinded task input an AI judge saw: no checkpoint, no permission sentence,
no role names and no AI label. Their labels are stored per labeller; an adjudicated set overrides
disagreements. AI labels are only ever compared against, never merged into, human labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ..agent_tasks.core import load_manifest, load_task
from ..agent_tasks.jobs import freeform_prelabel
from .schema import is_identifier

LABEL_SETS = {
    "mcq_judge": {"A", "B", "C", "none"},
    "freeform_prelabel": {"A", "B", "C", "multiple", "none", "unjudgeable"},
}
ADJUDICATED = "adjudicated"


def _flatten(task: dict) -> dict:
    flat = {}
    for key, value in task["input"].items():
        if isinstance(value, dict):
            prefix = "option" if key == "options" else key
            for sub, text in value.items():
                flat[f"{prefix}_{sub}"] = text
        else:
            flat[key] = value
    return flat


def _label_set(job: str) -> set[str]:
    if job not in LABEL_SETS:
        raise ValueError(f"job {job} has no human labelling protocol")
    return LABEL_SETS[job]


def export_csv(
    job_dir: Path,
    out_csv: Path,
    task_ids: list[str] | None = None,
    candidates: dict[str, list[str]] | None = None,
) -> dict:
    """Write a blinded labelling sheet with one row per task and empty label/notes columns.

    ``task_ids`` restricts the sheet to a subset (one labeller's assignment). ``candidates``
    adds ``candidate_1``/``candidate_2`` columns for an adjudication sheet; their order is
    shuffled per task so the adjudicator cannot tell which rater gave which label.
    """
    manifest = load_manifest(job_dir)
    allowed = sorted(_label_set(manifest["job"]))
    if out_csv.exists():
        raise FileExistsError(f"{out_csv} already exists; choose a new file")
    ids = sorted(manifest["tasks"]) if task_ids is None else sorted(task_ids)
    unknown = set(ids) - set(manifest["tasks"])
    if unknown:
        raise ValueError(f"unknown task IDs: {sorted(unknown)[:3]}")
    tasks = [load_task(job_dir, task_id) for task_id in ids]
    extra = ["candidate_1", "candidate_2"] if candidates is not None else []
    columns = ["task_id", *_flatten(tasks[0]), *extra, "label", "notes"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in tasks:
            row = {"task_id": task["task_id"], **_flatten(task), "label": "", "notes": ""}
            if candidates is not None:
                pair = list(candidates[task["task_id"]])
                random.Random(task["task_id"]).shuffle(pair)
                row.update(candidate_1=pair[0], candidate_2=pair[1])
            writer.writerow(row)
    return {"rows": len(tasks), "allowed_labels": allowed, "columns": columns}


def assign_tasks(
    task_ids: list[str], labellers: list[str], *, overlap: float, seed: int
) -> dict[str, list[str]]:
    """One labeller per task, balanced; a random ``overlap`` share goes to every labeller."""
    ids = sorted(task_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    shared = set(ids[: round(overlap * len(ids))])
    out = {}
    for index, task_id in enumerate(ids):
        out[task_id] = list(labellers) if task_id in shared else [
            labellers[index % len(labellers)]
        ]
    return dict(sorted(out.items()))


def human_judge_final(job_dir: Path, labellers: list[str]) -> dict:
    """Final labels under the one-human + AI-judge protocol.

    A task is settled by an adjudicated label, by two or more humans who agree, or by one human
    whose label equals the judge's. Everything else (human-judge or human-human disagreement)
    goes to adjudication. Final labels are always human labels.
    """
    humans = {}
    for name in labellers:
        path = job_dir / "human_labels" / f"{name}.jsonl"
        humans[name] = load_human_labels(job_dir, name) if path.exists() else {}
    judge = load_ai_labels(job_dir)
    adjudicated_path = job_dir / "human_labels" / f"{ADJUDICATED}.jsonl"
    adjudicated = load_human_labels(job_dir, ADJUDICATED) if adjudicated_path.exists() else {}
    final, pending, missing, candidates = {}, [], [], {}
    for task_id in sorted(load_manifest(job_dir)["tasks"]):
        given = {n: labels[task_id] for n, labels in humans.items() if task_id in labels}
        if task_id in adjudicated:
            final[task_id] = adjudicated[task_id]
        elif not given:
            missing.append(task_id)
        elif len(given) >= 2:
            values = set(given.values())
            if len(values) == 1:
                final[task_id] = values.pop()
            else:
                pending.append(task_id)
                candidates[task_id] = sorted(given.values())[:2]
        else:
            label = next(iter(given.values()))
            if judge.get(task_id) == label:
                final[task_id] = label
            else:
                pending.append(task_id)
                candidates[task_id] = [label, judge.get(task_id, "missing")]
    first_human: dict[str, str] = {}
    for name in labellers:
        for task_id, label in humans[name].items():
            if task_id in judge:
                first_human.setdefault(task_id, label)
    pairs = [humans[a] for a in labellers]
    return {
        "final": final,
        "needs_adjudication": pending,
        "adjudication_candidates": candidates,
        "unlabelled": missing,
        "human_vs_judge": agreement(first_human, {t: judge[t] for t in first_human}),
        "human_vs_human": agreement(pairs[0], pairs[1]) if len(pairs) >= 2 else None,
    }


def import_csv(
    job_dir: Path, csv_path: Path, labeller: str, *, allow_partial: bool = False
) -> dict:
    """Validate a filled sheet; store it as ``human_labels/<labeller>.jsonl`` (never replaced)."""
    if not is_identifier(labeller):
        raise ValueError("labeller must be an identifier")
    manifest = load_manifest(job_dir)
    allowed = _label_set(manifest["job"])
    target = job_dir / "human_labels" / f"{labeller}.jsonl"
    if target.exists():
        raise FileExistsError(f"labels for {labeller} already imported; use a new labeller name")
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen: set[str] = set()
    records = []
    for row in rows:
        task_id = row.get("task_id", "")
        if task_id not in manifest["tasks"] or task_id in seen:
            raise ValueError(f"unknown or duplicate task_id: {task_id!r}")
        seen.add(task_id)
        label = (row.get("label") or "").strip()
        if not label:
            if allow_partial:
                continue
            raise ValueError(f"task {task_id} has no label")
        if label not in allowed:
            raise ValueError(f"task {task_id}: label {label!r} not in {sorted(allowed)}")
        records.append(
            {
                "task_id": task_id,
                "labeller": labeller,
                "label": label,
                "notes": (row.get("notes") or "").strip(),
                "source_csv_sha256": sha,
                "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
    if not allow_partial and seen != set(manifest["tasks"]):
        raise ValueError(f"sheet covers {len(seen)} of {len(manifest['tasks'])} tasks")
    target.parent.mkdir(exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return {"labeller": labeller, "labelled": len(records), "csv_sha256": sha}


def load_human_labels(job_dir: Path, labeller: str) -> dict[str, str]:
    path = job_dir / "human_labels" / f"{labeller}.jsonl"
    return {r["task_id"]: r["label"] for r in map(json.loads, path.read_text().splitlines())}


def load_ai_labels(job_dir: Path) -> dict[str, str]:
    """Labels from the ingested AI responses (the payload field depends on the job)."""
    path = job_dir / "ingested.jsonl"
    if not path.exists():
        return {}
    key = {"mcq_judge": "selected", "freeform_prelabel": "matches"}[load_manifest(job_dir)["job"]]
    return {r["task_id"]: r["record"][key] for r in map(json.loads, path.read_text().splitlines())}


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or len(a) != len(b):
        raise ValueError("label lists must be non-empty and the same length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


def agreement(labels_a: dict[str, str], labels_b: dict[str, str]) -> dict:
    shared = sorted(set(labels_a) & set(labels_b))
    if not shared:
        return {"n": 0, "exact_agreement": None, "kappa": None, "disagreements": []}
    a, b = [labels_a[t] for t in shared], [labels_b[t] for t in shared]
    agree = sum(x == y for x, y in zip(a, b, strict=True))
    return {
        "n": len(shared),
        "only_in_first": len(set(labels_a) - set(labels_b)),
        "only_in_second": len(set(labels_b) - set(labels_a)),
        "exact_agreement": agree / len(shared),
        "kappa": cohens_kappa(a, b),
        "confusion": {
            f"{x}->{y}": n for (x, y), n in Counter(zip(a, b, strict=True)).most_common()
        },
        "disagreements": [t for t, x, y in zip(shared, a, b, strict=True) if x != y],
    }


def consensus(job_dir: Path, labellers: list[str]) -> dict[str, str]:
    """Labels on which every named labeller agrees; an adjudicated label overrides."""
    sets = [load_human_labels(job_dir, name) for name in labellers]
    result: dict[str, str] = {}
    for task_id in set().union(*sets) if sets else set():
        values = {s[task_id] for s in sets if task_id in s}
        if len(values) == 1 and all(task_id in s for s in sets):
            result[task_id] = next(iter(values))
    adjudicated = job_dir / "human_labels" / f"{ADJUDICATED}.jsonl"
    if adjudicated.exists():
        result.update(load_human_labels(job_dir, ADJUDICATED))
    return result


def episode_role_labels(
    job_dir: Path,
    task_labels: dict[str, str],
    scenarios: dict[str, dict],
    source_of: dict[str, str],
) -> dict[str, str]:
    """Free-text task labels -> a role label (boundary, alternative, ...) for each episode."""
    refs = json.loads((job_dir / "refs.json").read_text())
    out = {}
    for task_id, label in task_labels.items():
        for ref in refs.get(task_id, []):
            scenario = scenarios[source_of[ref["episode_id"]]]
            out[ref["episode_id"]] = freeform_prelabel.role_label(
                label, ref["letter_key"], scenario
            )
    return out


def episode_mcq_actions(
    job_dir: Path, task_labels: dict[str, str], requests: dict[str, dict]
) -> dict[str, str | None]:
    """Judged MCQ task labels -> chosen action per episode (``None`` for ``none``)."""
    refs = json.loads((job_dir / "refs.json").read_text())
    out = {}
    for task_id, label in task_labels.items():
        for episode_id in refs.get(task_id, []):
            out[episode_id] = None if label == "none" else requests[episode_id]["choice_key"][label]
    return out
