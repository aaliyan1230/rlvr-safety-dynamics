"""Resumable local assay runner with a replaceable response-provider contract."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from .io import read_jsonl, sha256_file, write_jsonl
from .permission_environment import score_executable
from .permission_prompts import build_conditions, score_mcq
from .permission_schema import SCHEMA_VERSION, digest, validate_response


class ResponseProvider(Protocol):
    """Providers receive a prompt record and return raw text and explicit stop metadata."""

    @property
    def provenance(self) -> Mapping: ...

    def respond(self, request: dict) -> dict: ...


class FixtureProvider:
    def __init__(self, path: Path):
        self.path = path
        self.responses_sha256 = sha256_file(path)
        self.responses = {}
        for row in read_jsonl(path):
            validate_response(row)
            if not isinstance(row.get("fixture_case"), str) or row["generated_tokens"] != 0:
                raise ValueError("FixtureProvider accepts only labeled engineering fixtures")
            key = (row["condition_id"], row["sample_id"])
            if key in self.responses:
                raise ValueError("duplicate fixture response")
            self.responses[key] = row

    @property
    def provenance(self) -> Mapping:
        return {
            "provider": "fixture_replay",
            "responses_sha256": self.responses_sha256,
            "inference_performed": False,
            "model": None,
            "model_revision": None,
            "tokenizer_revision": None,
            "chat_template_sha256": None,
            "generation_settings": None,
        }

    def respond(self, request: dict) -> dict:
        return self.responses[(request["condition_id"], request["sample_id"])]


def prepare_requests(scenarios: list[dict], *, stress: bool = False) -> list[dict]:
    cases = [
        "boundary",
        "alternative",
        "clarify",
        "no_op",
        "malformed",
        "length",
        "timeout",
        "interrupted",
        "error",
        "boundary_then_alternative",
    ]
    requests = []
    for condition in build_conditions(scenarios):
        samples = cases if stress and condition["format"] == "executable" else ["boundary"]
        for sample_id in samples:
            request = {**condition, "sample_id": sample_id}
            request["episode_id"] = digest([condition["condition_id"], sample_id])
            requests.append(request)
    return requests


def scripted_responses(scenarios: list[dict], requests: list[dict]) -> list[dict]:
    """Engineering fixtures only: these do not simulate a model's safety behavior."""
    by_source = {s["source_id"]: s for s in scenarios}
    responses = []
    for request in requests:
        scenario = by_source[request["source_id"]]
        roles = scenario["choice_actions"]
        case = request["sample_id"]
        if request["format"] == "mcq":
            raw = next(k for k, v in request["choice_key"].items() if v == roles["boundary"])
        elif request["format"] == "option_free":
            action = next(a for a in scenario["actions"] if a["action_id"] == roles["boundary"])
            raw = action["description"]
        else:
            actions = [roles[case]] if case in roles else [roles["boundary"]]
            if case == "no_op":
                actions = []
            elif case == "boundary_then_alternative":
                actions = [roles["boundary"], roles["alternative"]]
            raw = (
                "not a JSON action request"
                if case == "malformed"
                else json.dumps({"actions": actions})
            )
        status = case if case in {"length", "timeout", "interrupted", "error"} else "complete"
        responses.append(
            {
                "condition_id": request["condition_id"],
                "sample_id": case,
                "raw_response": raw,
                "stop_status": status,
                "generated_tokens": 0,
                "fixture_case": case,
            }
        )
    return responses


def _code_hashes() -> dict:
    base = Path(__file__).parent
    return {
        name: sha256_file(base / name)
        for name in (
            "permission_schema.py",
            "permission_prompts.py",
            "permission_environment.py",
            "permission_benchmark.py",
            "io.py",
        )
    }


def _score(scenario: dict, request: dict, response: dict) -> dict:
    if request["format"] == "executable":
        return score_executable(scenario, request["permission"], response)
    if request["format"] == "mcq":
        return score_mcq(request, response, scenario)
    return {
        "result_status": response["stop_status"]
        if response["stop_status"] != "complete"
        else "pending_human_review",
        "safety_undetermined": True,
        "text_label": None,
        "raw_text_preserved": True,
    }


def summarize(rows: list[dict]) -> dict:
    executable = [r for r in rows if r["format"] == "executable"]
    forbidden = [r for r in executable if r["permission"] == "forbidden"]
    allowed = [r for r in executable if r["permission"] == "allowed"]
    n = len(forbidden)
    violations = sum(r["score"]["violation_outcome"] is True for r in forbidden)
    unknown = sum(r["score"]["safety_undetermined"] for r in forbidden)
    return {
        "interpretation": "development engineering output; not model research evidence",
        "episodes": len(rows),
        "sources": len({r["source_id"] for r in rows}),
        "source_clusters": len({r["skeleton_id"] for r in rows}),
        "by_format": dict(Counter(r["format"] for r in rows)),
        "executable_statuses": dict(Counter(r["score"]["result_status"] for r in executable)),
        "forbidden_episodes": n,
        "known_executed_violations": violations,
        "undetermined_forbidden_episodes": unknown,
        "forbidden_violation_bounds": [violations / n, (violations + unknown) / n] if n else None,
        "allowed_episodes": len(allowed),
        "authorized_task_successes": sum(r["score"]["authorized_task_success"] for r in allowed),
        "pending_written_reviews": sum(
            r["score"]["result_status"] == "pending_human_review" for r in rows
        ),
    }


def run_benchmark(
    scenarios: list[dict],
    requests: list[dict],
    provider: ResponseProvider,
    output_dir: Path,
) -> dict:
    """Append one durable record per episode; reject incompatible or corrupted resumes."""
    conditions = {c["condition_id"]: c for c in build_conditions(scenarios)}
    episode_ids = set()
    for request in requests:
        expected = conditions.get(request.get("condition_id"))
        if expected is None or any(request.get(k) != v for k, v in expected.items()):
            raise ValueError("request does not match the current condition/prompt")
        if not isinstance(request.get("sample_id"), str) or not request["sample_id"]:
            raise ValueError("sample_id required")
        expected_id = digest([request["condition_id"], request["sample_id"]])
        if request.get("episode_id") != expected_id or expected_id in episode_ids:
            raise ValueError("duplicate or invalid episode ID")
        episode_ids.add(expected_id)
    if not requests:
        raise ValueError("run needs at least one request")
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
        "scenarios_sha256": digest(scenarios),
        "requests_sha256": digest(requests),
        "provider": provenance,
        "code_sha256": _code_hashes(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    manifest = {**payload, "run_id": digest(payload)}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    results_path = output_dir / "results.jsonl"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError("resume refused: tasks, prompts, provider, code, or runtime changed")
    else:
        if results_path.exists():
            raise ValueError("results exist without their run manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshots = {"scenarios.jsonl": scenarios, "requests.jsonl": requests}
    for name, expected_rows in snapshots.items():
        path = output_dir / name
        if path.exists():
            if list(read_jsonl(path)) != expected_rows:
                raise ValueError("saved task/prompt snapshot mismatch")
        else:
            write_jsonl(path, expected_rows)
    by_source = {s["source_id"]: s for s in scenarios}
    by_episode = {r["episode_id"]: r for r in requests}
    completed = {}
    if results_path.exists():
        data = results_path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise ValueError("partial final result line; preserve and repair before resuming")
        for row in read_jsonl(results_path):
            episode_id = row.get("episode_id")
            record_hash = row.get("record_sha256")
            unsigned = {k: v for k, v in row.items() if k != "record_sha256"}
            if (
                episode_id not in by_episode
                or episode_id in completed
                or row.get("run_id") != manifest["run_id"]
                or digest(unsigned) != record_hash
            ):
                raise ValueError("duplicate, foreign, or corrupted saved result")
            request = by_episode[episode_id]
            if row.get("request_sha256") != digest(request):
                raise ValueError("saved result request mismatch")
            completed[episode_id] = row
    with results_path.open("a", encoding="utf-8") as handle:
        for request in requests:
            if request["episode_id"] in completed:
                continue
            response = provider.respond(request)
            validate_response(response)
            if any(response.get(k) != request[k] for k in ("condition_id", "sample_id")):
                raise ValueError("provider returned a response for the wrong episode")
            row = {
                "schema_version": SCHEMA_VERSION,
                "run_id": manifest["run_id"],
                **{
                    k: request[k]
                    for k in (
                        "episode_id",
                        "condition_id",
                        "sample_id",
                        "source_id",
                        "skeleton_id",
                        "family",
                        "permission",
                        "format",
                        "wording_id",
                        "order_id",
                        "prompt_sha256",
                        "scenario_sha256",
                        "review",
                    )
                },
                "request_sha256": digest(request),
                "response": response,
                "score": _score(by_source[request["source_id"]], request, response),
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
