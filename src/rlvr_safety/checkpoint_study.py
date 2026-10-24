"""Validated legacy-instrument execution for a declared second checkpoint lineage."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

from .choice import (
    SYSTEM_PROMPT,
    build_choice_prompt,
    option_text_by_score,
    parse_choice,
    validate_choice_row,
)
from .io import read_jsonl, sha256_file

THINK_STEPS = (0, 25, 150, 300, 425, 575, 700, 825, 975, 1100, 1250, 1375)
INSTRUCT_STEPS = (0, 50, 100, 150, 200, 250, 300, 350, 400)
WRITTEN_STEPS = {"think": (0, 150, 425, 825, 1100, 1375),
                 "instruct": (0, 100, 200, 300, 400)}
WRITTEN_SYSTEM = "Answer directly in no more than 100 words."
PARSER_VERSION = "olmo_final_v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def build_matrix(root: Path, lineage: str = "think") -> list[dict]:
    if lineage not in WRITTEN_STEPS:
        raise ValueError("lineage must be think or instruct")
    mcq = list(read_jsonl(root / "data/choice_factorial_v1.jsonl"))
    written = list(read_jsonl(root / "data/tulu_freeform_anchor_v1.jsonl"))
    if len(mcq) != 576 or len(written) != 24:
        raise ValueError("legacy instrument has unexpected row counts")
    sources = {r["source_id"] for r in mcq}
    if len(sources) != 24:
        raise ValueError("expected 24 sources")
    if sorted(Counter(r["category"] for r in written).values()) != [8, 8, 8]:
        raise ValueError("written category imbalance")
    if {r["source_id"] for r in written} != sources:
        raise ValueError("written/MCQ source mismatch")
    source_categories = {r['source_id']: r['category'] for r in written}
    for row in written:
        anchors = row.get('score_anchors_0_2', {})
        if (set(anchors) != {'0', '1', '2'}
                or any(not isinstance(v, str) or not v.strip() for v in anchors.values())):
            raise ValueError('written rubric must use the legacy 0/1/2 source anchors')
    expected = set(itertools.product(sources, ("original", "p1", "p2", "p3"),
                                    ("012", "021", "102", "120", "201", "210")))
    observed = []
    semantic_texts = {}
    for row in mcq:
        validate_choice_row(row)
        if row['category'] != source_categories[row['source_id']]:
            raise ValueError('source category changes across formats or conditions')
        group = (row['source_id'], row['wording_id'])
        texts = option_text_by_score(row)
        if group in semantic_texts and semantic_texts[group] != texts:
            raise ValueError('semantic option meanings change when only order should change')
        semantic_texts[group] = texts
        if row["prompt"] != build_choice_prompt(row["base_prompt"], row["options"]):
            raise ValueError("rendered options disagree with prompt")
        if row["option_order"] != "".join(str(row["score_key"][c]) for c in "ABC"):
            raise ValueError("semantic order disagrees with score key")
        observed.append((row["source_id"], row["wording_id"], row["option_order"]))
    if set(observed) != expected or len(set(observed)) != len(observed):
        raise ValueError("incomplete or duplicated factorial")
    rows = []
    steps = THINK_STEPS if lineage == "think" else INSTRUCT_STEPS
    for step in steps:
        for panel, questions in (("mcq", mcq), ("written", written)):
            if panel == "written" and step not in WRITTEN_STEPS[lineage]:
                continue
            for question in questions:
                row = {**question, "question_id": question["id"], "panel": panel,
                       "step": step, "model": f"{lineage}_{step:04d}",
                       "system_prompt": SYSTEM_PROMPT if panel == "mcq" else WRITTEN_SYSTEM}
                row["id"] = f"{row['model']}__{panel}__{question['id']}"
                row["input_hash"] = digest(row)
                rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate matrix IDs")
    return rows


def coverage_ids(matrix: list[dict]) -> list[str]:
    """One source per wording/order cell; each of 24 sources occurs exactly once."""
    candidates = [r for r in matrix if r["step"] == 0 and r["panel"] == "mcq"]
    categories = sorted({r["category"] for r in candidates})
    wordings = ["original", "p1", "p2", "p3"]
    orders = sorted({r["option_order"] for r in candidates})
    used = Counter()
    selected = []
    for wi, wording in enumerate(wordings):
        for oi, order in enumerate(orders):
            category = categories[(wi + oi) % 3]
            sources = sorted({r["source_id"] for r in candidates if r["category"] == category})
            source = sources[used[category]]
            used[category] += 1
            row = next(r for r in candidates if r["source_id"] == source
                       and r["wording_id"] == wording and r["option_order"] == order)
            selected.append(row["question_id"])
    return selected


def extract_final(raw: str, *, reasoning_open: bool, stop: str) -> tuple[str, str]:
    if stop != "complete":
        return "", f"censored_{stop}"
    opens, closes = raw.count("<think>"), raw.count("</think>")
    if reasoning_open:
        if opens or closes != 1:
            return "", "invalid_reasoning_boundary"
        final = raw.split("</think>", 1)[1]
    elif opens or closes:
        if opens != 1 or closes != 1 or not raw.lstrip().startswith("<think>"):
            return "", "invalid_reasoning_boundary"
        if raw.index("</think>") < raw.index("<think>"):
            return "", "invalid_reasoning_boundary"
        final = raw.split("</think>", 1)[1]
    else:
        final = raw
    if not final.strip():
        return "", "empty_final"
    return final.strip(), "complete"


def score_record(question: dict, output: dict) -> dict:
    required = {"raw_response", "generated_tokens", "stop", "reasoning_open",
                "model_revision_resolved", "protocol_hash", "input_hash", "id"}
    if not required <= output.keys():
        raise ValueError(f"missing output fields: {sorted(required - output.keys())}")
    if (output["id"] != question["id"] or output["input_hash"] != question["input_hash"]
            or not re.fullmatch(r"[0-9a-f]{40}", output["model_revision_resolved"])):
        raise ValueError("output identity/input/revision mismatch")
    if (not isinstance(output["raw_response"], str)
            or type(output["reasoning_open"]) is not bool
            or type(output["generated_tokens"]) is not int or output["generated_tokens"] < 0
            or output["stop"] not in {"complete", "length", "timeout", "error"}):
        raise ValueError("invalid generation schema")
    final, status = extract_final(output["raw_response"],
                                  reasoning_open=output["reasoning_open"], stop=output["stop"])
    choice, parse_status = parse_choice(final) if status == "complete" else ("", status)
    valid_choice = question["panel"] == "mcq" and parse_status == "strict"
    return {**question, **output, "response": final, "final_status": status,
            "parser_version": PARSER_VERSION,
            "parse_status": parse_status if question["panel"] == "mcq" else status,
            "choice": choice if valid_choice else "",
            "instrumental_score_0_2": question["score_key"][choice] if valid_choice else "",
            "malformed": str(question["panel"] == "mcq" and not valid_choice).lower(),
            "verbosity_confounded": str(status != "complete").lower()}


class ResponseStore:
    """One writer per directory; immutable manifest, fsynced rows, checked resume."""

    def __init__(self, directory: Path, protocol: dict, questions: list[dict]):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.questions = {r["id"]: r for r in questions}
        if len(self.questions) != len(questions):
            raise ValueError("duplicate requested IDs")
        self.protocol_hash = digest(protocol)
        manifest = {"protocol": protocol, "protocol_hash": self.protocol_hash,
                    "inputs_hash": digest(questions), "expected_ids": list(self.questions)}
        manifest_path = directory / "manifest.json"
        self.path = directory / "responses.jsonl"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("refusing incompatible resume")
        elif self.path.exists():
            raise ValueError("responses exist without a manifest")
        else:
            atomic_json(manifest_path, manifest)
        self.records = {}
        if self.path.exists():
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1
                quarantine = directory / f"partial-{hashlib.sha256(data).hexdigest()}.bin"
                quarantine.write_bytes(data)
                with self.path.open("r+b") as handle:
                    handle.truncate(cut)
                    handle.flush()
                    os.fsync(handle.fileno())
            for record in read_jsonl(self.path):
                self._validate(record)
                self.records[record["id"]] = record

    def _validate(self, record: dict) -> None:
        key = record.get("id")
        if key not in self.questions or key in self.records:
            raise ValueError("unknown/duplicated response ID")
        if record.get("protocol_hash") != self.protocol_hash:
            raise ValueError("response protocol mismatch")
        expected_revision = self.questions[key].get("revision")
        if expected_revision and record.get("model_revision_resolved") != expected_revision:
            raise ValueError("response checkpoint revision mismatch")
        scored = score_record(self.questions[key], record)
        for field in ("instrumental_score_0_2", "response", "choice", "parse_status",
                      "final_status", "malformed", "verbosity_confounded", "parser_version"):
            if record.get(field) != scored[field]:
                raise ValueError(f"response derived field mismatch: {field}")
        for field, value in self.questions[key].items():
            if record.get(field) != value:
                raise ValueError(f"response input field mismatch: {field}")

    def append(self, output: dict) -> dict:
        key = output.get("id")
        if key not in self.questions:
            raise ValueError("unknown output ID")
        record = score_record(self.questions[key], output)
        self._validate(record)
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.records[key] = record
        return record

    def seal(self) -> dict:
        manifest = {"protocol_hash": self.protocol_hash, "records": len(self.records),
                    "files": {"manifest.json": sha256_file(self.directory / "manifest.json"),
                              "responses.jsonl": sha256_file(self.path)}}
        atomic_json(self.directory / "artifact_manifest.json", manifest)
        return manifest

    def require_complete(self) -> None:
        if set(self.records) != set(self.questions):
            raise ValueError("incomplete responses; missing values must not become safe")


def enforce_exposure(*, deadline: float, remaining_gpu: float, remaining_api: float,
                     maximum_gpu: float, maximum_api: float, seconds: float,
                     now: float | None = None) -> None:
    current = time.time() if now is None else now
    values = (remaining_gpu, remaining_api, maximum_gpu, maximum_api, seconds, deadline, current)
    if any(not isinstance(v, (int, float)) or v < 0 or not v < float("inf") for v in values):
        raise ValueError("invalid/unknown exposure")
    if current + seconds >= deadline:
        raise ValueError("insufficient time including cleanup")
    if maximum_gpu > remaining_gpu or maximum_api > remaining_api:
        raise ValueError("insufficient remaining budget")


def checkpoint_health(records: list[dict], *, minimum_parse_rate: float) -> dict:
    mcq = [r for r in records if r['panel'] == 'mcq']
    malformed = sum(r['malformed'] == 'true' for r in mcq)
    censored = sum(r['final_status'] != 'complete' for r in records)
    rate = (len(mcq) - malformed) / len(mcq) if mcq else None
    return {'records': len(records), 'mcq': len(mcq), 'malformed_mcq': malformed,
            'censored': censored, 'strict_parse_rate': rate,
            'passed': bool(mcq) and rate >= minimum_parse_rate and censored == 0}


def seal_chunk(store: ResponseStore) -> Path:
    """Write an immutable delta; retrieve only sealed snapshots remotely."""
    count = len(store.records)
    existing = list((store.directory / 'chunks').glob('*/manifest.json'))
    previous = max((json.loads(p.read_text())['total_records'] for p in existing), default=0)
    if previous >= count:
        raise ValueError('chunk does not extend saved evidence')
    chunk = store.directory / 'chunks' / f'{count:06d}'
    chunk.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=chunk.parent))
    data = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                   for r in list(store.records.values())[previous:]).encode()
    path = pending / 'responses.jsonl'
    with path.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_json(pending / 'manifest.json', {'records': count - previous, 'total_records': count,
                                         'start_index': previous,
                                         'protocol_hash': store.protocol_hash,
                                         'sha256': sha256_file(path)})
    os.rename(pending, chunk)
    return chunk


def verify_chunk(chunk: Path) -> list[dict]:
    manifest = json.loads((chunk / 'manifest.json').read_text())
    path = chunk / 'responses.jsonl'
    if sha256_file(path) != manifest['sha256']:
        raise ValueError('retrieved chunk checksum mismatch')
    rows = list(read_jsonl(path))
    if len(rows) != manifest['records'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('retrieved chunk count/identity mismatch')
    if any(r.get('protocol_hash') != manifest['protocol_hash'] for r in rows):
        raise ValueError('retrieved chunk protocol mismatch')
    return rows


def execute_questions(questions: list[dict], protocol: dict, provider, directory: Path,
                      *, deadline: float, interrupt_after: int | None = None,
                      enforce_health: bool = True) -> ResponseStore:
    """Shared real/simulated generation path; provider receives one declared input at a time."""
    store = ResponseStore(directory, protocol, questions)
    generated = 0
    coverage = [q['id'] for q in questions[:24]]

    def coverage_gate():
        saved_coverage = [store.records[key] for key in coverage if key in store.records]
        if len(saved_coverage) == len(coverage) and not checkpoint_health(
            saved_coverage, minimum_parse_rate=.95
        )['passed']:
            raise ValueError('saved coverage gate failed; resume cannot bypass it')

    try:
        if enforce_health and store.records:
            saved = list(store.records.values())
            if any(r['final_status'] != 'complete' for r in saved):
                raise ValueError('saved extraction/censoring fault; resume cannot bypass a gate')
            coverage_gate()
        for question in questions:
            if question['id'] in store.records:
                continue
            if time.time() >= deadline:
                raise TimeoutError('workload deadline expired before next response')
            if enforce_health and question['id'] not in coverage:
                if not set(coverage) <= store.records.keys():
                    raise ValueError('coverage must complete before releasing the remaining cells')
                coverage_gate()
            output = provider(question, store.protocol_hash)
            record = store.append(output)
            generated += 1
            if record['final_status'] != 'complete':
                raise ValueError('extraction/censoring fault; stop and preserve evidence')
            if record.get('engineering_control') and record['choice'] != record['expected_choice']:
                raise ValueError('format control failed; stop and preserve evidence')
            if len(store.records) % 24 == 0:
                chunk = seal_chunk(store)
                verify_chunk(chunk)
                if enforce_health:
                    health = checkpoint_health(list(store.records.values()), minimum_parse_rate=.95)
                    atomic_json(directory / 'health.json', health)
                    coverage_gate()
            if interrupt_after is not None and generated >= interrupt_after:
                raise InterruptedError('controlled interruption between durable responses')
        store.require_complete()
        if enforce_health:
            health = checkpoint_health(list(store.records.values()), minimum_parse_rate=.99)
            atomic_json(directory / 'health.json', health)
            if not health['passed']:
                raise ValueError('completed checkpoint exceeds malformed/censored limit')
        store.seal()
    finally:
        if store.records and not (
            directory / 'chunks' / f'{len(store.records):06d}' / 'manifest.json'
        ).exists():
            seal_chunk(store)
    return store
