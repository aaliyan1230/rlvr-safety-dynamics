"""Resumable runner with a replaceable response-provider contract.

Every episode is appended and fsynced as it completes. A resumed run must match the original
manifest (scenarios, requests, provider, code and runtime); anything else is refused.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from ..io import read_jsonl, sha256_file, write_jsonl
from .schema import SCHEMA_VERSION, digest, validate_response
from .scoring import (
    EXECUTABLE_SCORER_ID,
    MCQ_RULE_ID,
    READOUT_SCORER_ID,
    score_executable,
    score_free_text,
    score_mcq,
    score_readout,
)

CODE_FILES = (
    "schema.py",
    "environment.py",
    "prompts.py",
    "scoring.py",
    "benchmark.py",
    "generation.py",
)
ROW_FIELDS = (
    "episode_id",
    "condition_id",
    "sample_id",
    "source_id",
    "skeleton_id",
    "family",
    "is_control",
    "permission",
    "format",
    "wording_id",
    "order_id",
    "id_arm",
    "prompt_sha256",
    "scenario_sha256",
    "review",
)


class ResponseProvider(Protocol):
    """Providers receive a request and return raw text, stop metadata and optional log-probs."""

    @property
    def provenance(self) -> Mapping: ...

    def respond(self, request: dict) -> dict: ...


def prepare_requests(conditions: list[dict], sample_ids: tuple[str, ...] = ("0",)) -> list[dict]:
    requests = []
    for condition in conditions:
        for sample_id in sample_ids:
            request = {**condition, "sample_id": sample_id}
            request["episode_id"] = digest([condition["condition_id"], sample_id])
            requests.append(request)
    return requests


def validate_requests(scenarios: list[dict], requests: list[dict]) -> None:
    """Check each request is internally consistent and tied to the scenario it names."""
    by_source = {s["source_id"]: s for s in scenarios}
    seen = set()
    for request in requests:
        scenario = by_source.get(request.get("source_id"))
        if scenario is None:
            raise ValueError("request names an unknown source")
        if request.get("scenario_sha256") != digest(scenario):
            raise ValueError("request does not match the current scenario")
        expected_prompt = digest([request.get("system_prompt"), request.get("prompt")])
        if request.get("prompt_sha256") != expected_prompt:
            raise ValueError("request prompt does not match its hash")
        if request.get("messages") != [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": request["prompt"]},
        ]:
            raise ValueError("request messages do not match its prompt")
        sample_id = request.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id required")
        expected_id = digest([request["condition_id"], sample_id])
        if request.get("episode_id") != expected_id or expected_id in seen:
            raise ValueError("duplicate or invalid episode ID")
        seen.add(expected_id)
    if not requests:
        raise ValueError("run needs at least one request")


def _score(scenario: dict, request: dict, response: dict) -> dict:
    if request["format"] == "executable":
        return score_executable(scenario, request, response)
    if request["format"] == "mcq":
        return score_mcq(request, response, scenario)
    return score_free_text(response)


def code_hashes() -> dict:
    base = Path(__file__).parent
    hashes = {name: sha256_file(base / name) for name in CODE_FILES if (base / name).exists()}
    hashes["io.py"] = sha256_file(base.parent / "io.py")
    return hashes


def summarize(rows: list[dict]) -> dict:
    scenario_rows = [r for r in rows if not r["is_control"]]
    controls = [r for r in rows if r["is_control"]]
    executable = [r for r in scenario_rows if r["format"] == "executable"]
    by_permission = {}
    for permission in ("allowed", "forbidden"):
        subset = [r for r in executable if r["permission"] == permission]
        by_permission[permission] = {
            "episodes": len(subset),
            "chose_boundary": sum(r["score"]["chose_boundary"] is True for r in subset),
            "chose_alternative": sum(r["score"]["chose_alternative"] is True for r in subset),
            "chose_clarify": sum(r["score"]["chose_clarify"] is True for r in subset),
            "executed_violations": sum(r["score"]["executed_violation"] for r in subset),
            "undetermined": sum(r["score"]["safety_undetermined"] for r in subset),
        }
    mcq = [r for r in scenario_rows if r["format"] == "mcq"]
    return {
        "interpretation": "development engineering output; not model research evidence",
        "episodes": len(rows),
        "sources": len({r["source_id"] for r in rows if not r["is_control"]}),
        "source_clusters": len({r["skeleton_id"] for r in scenario_rows}),
        "by_format": dict(Counter(r["format"] for r in rows)),
        "uncensored": sum(r["response"]["stop_status"] == "complete" for r in rows),
        "executable_by_permission": by_permission,
        "executable_statuses": dict(Counter(r["score"]["result_status"] for r in executable)),
        "mcq_statuses": dict(Counter(r["score"]["result_status"] for r in mcq)),
        "mcq_needing_judge": sum(r["score"]["needs_judge"] for r in mcq),
        "controls": {
            "episodes": len(controls),
            "valid_format": sum(r["score"]["valid_format"] for r in controls),
            "requested_action_followed": sum(
                r["score"]["requested_action_followed"] is True for r in controls
            ),
            "goal_reached": sum(r["score"]["task_success"] for r in controls),
            "by_arm": {
                arm: {
                    "episodes": sum(r["id_arm"] == arm for r in controls),
                    "followed": sum(
                        r["id_arm"] == arm and r["score"]["requested_action_followed"] is True
                        for r in controls
                    ),
                }
                for arm in sorted({r["id_arm"] for r in controls})
            },
        },
        "pending_written_reviews": sum(
            r["score"]["result_status"] == "pending_review" for r in rows
        ),
        "readouts": sum(r.get("readout_score") is not None for r in rows),
    }


def _manifest(scenarios: list[dict], requests: list[dict], provider: ResponseProvider) -> dict:
    provenance = dict(provider.provenance)
    if provenance.get("inference_performed") is not False:
        for key in (
            "model",
            "model_revision",
            "tokenizer_revision",
            "chat_template_sha256",
            "generation_settings",
        ):
            if not provenance.get(key):
                raise ValueError(f"inference provider must pin {key}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scorers": {
            "mcq": MCQ_RULE_ID,
            "executable": EXECUTABLE_SCORER_ID,
            "readout": READOUT_SCORER_ID,
        },
        "scenarios_sha256": digest(scenarios),
        "requests_sha256": digest(requests),
        "provider": provenance,
        "code_sha256": code_hashes(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    return {**payload, "run_id": digest(payload)}


def _load_completed(results_path: Path, manifest: dict, by_episode: dict) -> dict:
    completed: dict = {}
    if not results_path.exists():
        return completed
    data = results_path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("partial final result line; preserve and repair before resuming")
    for row in read_jsonl(results_path):
        episode_id = row.get("episode_id")
        unsigned = {k: v for k, v in row.items() if k != "record_sha256"}
        if (
            episode_id not in by_episode
            or episode_id in completed
            or row.get("run_id") != manifest["run_id"]
            or digest(unsigned) != row.get("record_sha256")
        ):
            raise ValueError("duplicate, foreign, or corrupted saved result")
        if row.get("request_sha256") != digest(by_episode[episode_id]):
            raise ValueError("saved result request mismatch")
        completed[episode_id] = row
    return completed


def run_benchmark(
    scenarios: list[dict],
    requests: list[dict],
    provider: ResponseProvider,
    output_dir: Path,
) -> dict:
    """Append one durable record per episode; reject incompatible or corrupted resumes."""
    validate_requests(scenarios, requests)
    manifest = _manifest(scenarios, requests, provider)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    results_path = output_dir / "results.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume refused: tasks, prompts, provider, code, or runtime changed")
    else:
        if results_path.exists():
            raise ValueError("results exist without their run manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, expected_rows in {"scenarios.jsonl": scenarios, "requests.jsonl": requests}.items():
        path = output_dir / name
        if path.exists():
            if list(read_jsonl(path)) != expected_rows:
                raise ValueError("saved task/prompt snapshot mismatch")
        else:
            write_jsonl(path, expected_rows)
    by_source = {s["source_id"]: s for s in scenarios}
    by_episode = {r["episode_id"]: r for r in requests}
    completed = _load_completed(results_path, manifest, by_episode)
    with results_path.open("a", encoding="utf-8") as handle:
        for request in requests:
            if request["episode_id"] in completed:
                continue
            response = provider.respond(request)
            validate_response(response)
            if any(response.get(k) != request[k] for k in ("condition_id", "sample_id")):
                raise ValueError("provider returned a response for the wrong episode")
            scenario = by_source[request["source_id"]]
            row = {
                "schema_version": SCHEMA_VERSION,
                "run_id": manifest["run_id"],
                **{k: request[k] for k in ROW_FIELDS},
                "workload": request.get("workload"),
                "request_sha256": digest(request),
                "response": response,
                "score": _score(scenario, request, response),
                "readout_score": score_readout(request, response.get("answer_logprobs")),
            }
            row["record_sha256"] = digest(row)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[request["episode_id"]] = row
    summary = summarize(list(completed.values()))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = {
        p.name: sha256_file(p)
        for p in (
            manifest_path,
            results_path,
            output_dir / "summary.json",
            output_dir / "scenarios.jsonl",
            output_dir / "requests.jsonl",
        )
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n"
    )
    return summary
