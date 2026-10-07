"""Model-agnostic task pipeline: plan -> answer (any backend) -> ingest -> audit.

A *job directory* holds everything for one batch of tasks::

    manifest.json        task list with hashes (plan record)
    refs.json            private task -> source mapping (never shown to the answering agent)
    tasks/<id>.json      self-contained task: instructions, input, response schema
    responses/<id>.json  one envelope per task, written by any agent, CLI or API adapter
    ingested.jsonl       validated responses with provenance (append-only)
    rejections.jsonl     responses that failed validation, with the reason (append-only)

Backends only have to produce a response envelope; they never touch validation or provenance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

PIPELINE_VERSION = "agent-tasks-2026-10-07"
ENVELOPE_KEYS = {"task_id", "task_sha256", "payload", "agent", "model_reported"}


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def make_task(job: str, rubric_id: str, instructions: str, task_input: dict, schema: dict) -> dict:
    body = {
        "pipeline": PIPELINE_VERSION,
        "job": job,
        "rubric_id": rubric_id,
        "instructions": instructions,
        "input": task_input,
        "response_schema": schema,
    }
    task_sha = digest(body)
    return {"task_id": f"{job}-{task_sha[:12]}", **body, "task_sha256": task_sha}


def verify_task(task: dict) -> None:
    body = {k: v for k, v in task.items() if k not in {"task_id", "task_sha256"}}
    if digest(body) != task.get("task_sha256") or task.get("task_id") != (
        f"{task['job']}-{task['task_sha256'][:12]}"
    ):
        raise ValueError(f"task {task.get('task_id')} does not match its hash")


def write_plan(
    job_dir: Path,
    job: str,
    rubric_id: str,
    tasks: list[dict],
    refs: dict[str, list] | None = None,
) -> dict:
    """Write task files and the manifest. Re-planning identical tasks is a no-op."""
    if not tasks:
        raise ValueError("a plan needs at least one task")
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task IDs in plan")
    (job_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (job_dir / "responses").mkdir(exist_ok=True)
    for task in tasks:
        verify_task(task)
        path = job_dir / "tasks" / f"{task['task_id']}.json"
        if path.exists() and json.loads(path.read_text()) != task:
            raise ValueError(f"task {task['task_id']} already exists with different content")
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
    manifest_path = job_dir / "manifest.json"
    manifest = {
        "pipeline": PIPELINE_VERSION,
        "job": job,
        "rubric_id": rubric_id,
        "tasks": {t["task_id"]: t["task_sha256"] for t in tasks},
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["job"] != job or old["rubric_id"] != rubric_id:
            raise ValueError("job directory already holds a different job or rubric")
        merged = {**old["tasks"], **manifest["tasks"]}
        manifest["tasks"] = merged
        manifest["created_at"] = old["created_at"]
    else:
        manifest["created_at"] = _now()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if refs is not None:
        refs_path = job_dir / "refs.json"
        old_refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
        refs_path.write_text(json.dumps({**old_refs, **refs}, indent=2) + "\n")
    return manifest


def load_manifest(job_dir: Path) -> dict:
    return json.loads((job_dir / "manifest.json").read_text())


def load_task(job_dir: Path, task_id: str) -> dict:
    task = json.loads((job_dir / "tasks" / f"{task_id}.json").read_text())
    verify_task(task)
    return task


def pending_tasks(job_dir: Path) -> list[str]:
    """Tasks with no response file yet (a rejected response still counts as answered)."""
    manifest = load_manifest(job_dir)
    return sorted(
        t for t in manifest["tasks"] if not (job_dir / "responses" / f"{t}.json").exists()
    )


Validator = Callable[[dict, dict], dict]


def ingest(job_dir: Path, validate: Validator, *, operator: str, now: str | None = None) -> dict:
    """Validate every unseen response; append accepted and rejected records with provenance."""
    manifest = load_manifest(job_dir)
    accepted_path, rejected_path = job_dir / "ingested.jsonl", job_dir / "rejections.jsonl"
    seen = {r["task_id"]: r for r in _read_jsonl(accepted_path)}
    seen_rejected = {(r["task_id"], r["response_sha256"]) for r in _read_jsonl(rejected_path)}
    counts = {"accepted": 0, "rejected": 0, "unchanged": 0, "missing": 0}
    for task_id, task_sha in sorted(manifest["tasks"].items()):
        path = job_dir / "responses" / f"{task_id}.json"
        if not path.exists():
            counts["missing"] += 1
            continue
        response_sha = file_sha256(path)
        if task_id in seen:
            if seen[task_id]["response_sha256"] != response_sha:
                raise ValueError(f"response for {task_id} changed after ingestion")
            counts["unchanged"] += 1
            continue
        if (task_id, response_sha) in seen_rejected:
            counts["unchanged"] += 1
            continue
        stamp = now or _now()
        try:
            envelope = json.loads(path.read_text())
            if not isinstance(envelope, dict) or not ENVELOPE_KEYS <= set(envelope):
                raise ValueError(f"envelope needs keys {sorted(ENVELOPE_KEYS)}")
            if envelope["task_id"] != task_id or envelope["task_sha256"] != task_sha:
                raise ValueError("response does not match the task it names")
            for key in ("agent", "model_reported"):
                if not isinstance(envelope[key], str) or not envelope[key].strip():
                    raise ValueError(f"envelope {key} must be a nonempty string")
            task = load_task(job_dir, task_id)
            record = validate(task, envelope)
        except (ValueError, json.JSONDecodeError) as exc:
            _append_jsonl(
                rejected_path,
                {
                    "task_id": task_id,
                    "response_sha256": response_sha,
                    "reason": str(exc),
                    "rejected_at": stamp,
                    "operator": operator,
                },
            )
            counts["rejected"] += 1
            continue
        _append_jsonl(
            accepted_path,
            {
                "task_id": task_id,
                "task_sha256": task_sha,
                "job": manifest["job"],
                "rubric_id": manifest["rubric_id"],
                "response_sha256": response_sha,
                "agent": envelope["agent"],
                "model_reported": envelope["model_reported"],
                "backend": envelope.get("backend"),
                "ingested_at": stamp,
                "operator": operator,
                "record": record,
            },
        )
        counts["accepted"] += 1
    return counts


def audit(job_dir: Path) -> dict:
    """Re-hash everything and report coverage; ``ok`` means nothing is inconsistent."""
    manifest = load_manifest(job_dir)
    problems: list[str] = []
    for task_id, task_sha in manifest["tasks"].items():
        try:
            if load_task(job_dir, task_id)["task_sha256"] != task_sha:
                problems.append(f"{task_id}: task file differs from manifest")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"{task_id}: {exc}")
    accepted = _read_jsonl(job_dir / "ingested.jsonl")
    rejected = _read_jsonl(job_dir / "rejections.jsonl")
    ids = [r["task_id"] for r in accepted]
    if len(set(ids)) != len(ids):
        problems.append("duplicate accepted records")
    for row in accepted:
        path = job_dir / "responses" / f"{row['task_id']}.json"
        if not path.exists() or file_sha256(path) != row["response_sha256"]:
            problems.append(f"{row['task_id']}: response changed or removed after ingestion")
        if row["task_id"] not in manifest["tasks"]:
            problems.append(f"{row['task_id']}: ingested but not in manifest")
    response_ids = {p.stem for p in (job_dir / "responses").glob("*.json")}
    orphans = sorted(response_ids - set(manifest["tasks"]))
    if orphans:
        problems.append(f"responses without tasks: {orphans}")
    accepted_ids = set(ids)
    rejected_ids = {r["task_id"] for r in rejected} - accepted_ids
    return {
        "job": manifest["job"],
        "rubric_id": manifest["rubric_id"],
        "tasks": len(manifest["tasks"]),
        "accepted": len(accepted_ids),
        "rejected_only": len(rejected_ids),
        "unanswered": len(manifest["tasks"]) - len(response_ids & set(manifest["tasks"])),
        "agents": sorted({(r["agent"], r["model_reported"]) for r in accepted}),
        "problems": problems,
        "ok": not problems,
    }


class CallableBackend:
    """Wrap any ``fn(task) -> JSON text`` (an LLM API adapter, a local model, a test double)."""

    def __init__(self, fn: Callable[[dict], str], agent: str, model_reported: str):
        self.fn, self.agent, self.model_reported = fn, agent, model_reported

    def answer(self, task: dict) -> dict:
        return {
            "task_id": task["task_id"],
            "task_sha256": task["task_sha256"],
            "agent": self.agent,
            "model_reported": self.model_reported,
            "backend": {"type": "callable"},
            "payload": json.loads(self.fn(task)),
        }


class CommandBackend:
    """Run any command per task (task JSON on stdin, payload JSON on stdout)."""

    def __init__(self, command: list[str], agent: str, model_reported: str, timeout: int = 300):
        self.command, self.agent, self.model_reported = command, agent, model_reported
        self.timeout = timeout

    def answer(self, task: dict) -> dict:
        done = subprocess.run(
            self.command,
            input=json.dumps(task),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=True,
        )
        return {
            "task_id": task["task_id"],
            "task_sha256": task["task_sha256"],
            "agent": self.agent,
            "model_reported": self.model_reported,
            "backend": {"type": "command", "command": self.command[0]},
            "payload": json.loads(done.stdout),
        }


def run_backend(job_dir: Path, backend, *, limit: int | None = None) -> list[str]:
    """Answer pending tasks with a callable/command backend (the file backend needs no runner)."""
    done = []
    for task_id in pending_tasks(job_dir)[: limit or None]:
        envelope = backend.answer(load_task(job_dir, task_id))
        path = job_dir / "responses" / f"{task_id}.json"
        path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n")
        done.append(task_id)
    return done
