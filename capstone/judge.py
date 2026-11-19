"""Blinded Gemini labels for written answers, using the frozen rubric freeform-label-2026-10-15b.

The judge sees only the situation, three shuffled lettered operations and the answer: no
permission sentence, checkpoint or arm. Identical (situation, answer) pairs share one task and
one call. Every call is cached under <cache>/<task_id>.json, so reruns cost nothing.
Usage-based spend is summed before each call and stops the run at --ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlvr_safety.agent_tasks.jobs import freeform_prelabel as fp

MODEL = "gemini-3.8-flash"
RUBRIC = fp.RUBRIC_RULES_B_ID
SETTINGS = {
    "temperature": 0,
    "maxOutputTokens": 4096,
    "thinkingConfig": {"thinkingLevel": "low"},
    "responseMimeType": "application/json",
    "responseSchema": {
        "type": "OBJECT",
        "properties": {
            "matches": {
                "type": "STRING",
                "enum": ["A", "B", "C", "multiple", "none", "unjudgeable"],
            },
            "evidence": {"type": "STRING"},
        },
        "required": ["matches", "evidence"],
    },
}
# USD per million tokens (input, output incl. thinking), list prices checked 2026-11-18,
# rounded up so the ceiling stays conservative.
RATES = {
    "gemini-3.1-pro-preview": (2.5, 15.0),
    "gemini-3.8-flash": (1.5, 7.5),
    "gemini-4-argon": (4.0, 20.0),
}
LOCK = threading.Lock()


def call(api_key: str, prompt: str) -> dict:
    assert MODEL in RATES, f"no price recorded for {MODEL}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": SETTINGS,
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(payload).encode(),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def cost(usage: dict) -> float:
    out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    rate_in, rate_out = RATES[MODEL]
    return (usage.get("promptTokenCount", 0) * rate_in + out * rate_out) / 1e6


def render(task: dict) -> str:
    return (
        task["instructions"]
        + "\n\nInput:\n"
        + json.dumps(task["input"], ensure_ascii=False)
        + "\n\nresponse_schema: "
        + json.dumps(task["response_schema"])
    )


def label_task(task: dict, cache: Path, api_key: str, spent: list, ceiling: float) -> dict:
    path = cache / f"{task['task_id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    last_error = None
    prompt = render(task)
    for attempt in range(3):
        with LOCK:
            if spent[0] > ceiling:
                raise RuntimeError(f"judge ceiling ${ceiling} reached")
        try:
            raw = call(api_key, prompt)
        except Exception as exc:  # network or quota; back off and retry
            last_error = str(exc)
            time.sleep(5 * (attempt + 1))
            continue
        usage = raw.get("usageMetadata", {})
        with LOCK:
            spent[0] += cost(usage)
        try:
            text = raw["candidates"][0]["content"]["parts"][-1]["text"]
            payload = fp.validate(task, {"payload": json.loads(text)})
        except (KeyError, IndexError, ValueError) as exc:
            last_error = str(exc)
            prompt = (
                render(task)
                + f"\n\nYour previous reply was rejected: {last_error}. Copy the evidence"
                " character for character from the answer, without added formatting."
            )
            continue
        record = {
            "task_id": task["task_id"],
            "record": payload,
            "model": MODEL,
            "model_version": raw.get("modelVersion"),
            "usage": usage,
            "attempt": attempt,
        }
        path.write_text(json.dumps(record) + "\n")
        return record
    record = {
        "task_id": task["task_id"],
        "record": {"matches": "unjudgeable", "evidence": ""},
        "model": MODEL,
        "judge_failed": last_error,
    }
    path.write_text(json.dumps(record) + "\n")
    return record


def main() -> None:
    global MODEL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--ceiling", type=float, required=True, help="USD for this invocation")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default=MODEL, choices=sorted(RATES))
    p.add_argument("responses", type=Path, nargs="+")
    a = p.parse_args()
    MODEL = a.model
    api_key = os.environ["GEMINI_API_KEY"]
    requests = {json.loads(x)["episode_id"]: json.loads(x) for x in a.requests.open()}
    scenarios = {json.loads(x)["source_id"]: json.loads(x) for x in a.scenarios.open()}
    a.cache.mkdir(parents=True, exist_ok=True)
    spent = [0.0]
    for path in a.responses:
        responses = [json.loads(x) for x in path.open()]
        rows = [
            {**requests[r["episode_id"]], "episode_id": r["episode_id"], "response": r}
            for r in responses
            if requests[r["episode_id"]]["format"] == "option_free"
        ]
        tasks, refs = fp.build_tasks(rows, requests, scenarios, RUBRIC)
        with ThreadPoolExecutor(a.workers) as pool:
            done = list(
                pool.map(lambda t: label_task(t, a.cache, api_key, spent, a.ceiling), tasks)
            )
        by_task = {d["task_id"]: d for d in done}
        out = path.with_name(path.name.replace(".responses.jsonl", ".written-labels.jsonl"))
        counts: Counter = Counter()
        with out.open("w") as f:
            for row in rows:
                perm = row["permission"]
                if row["response"]["stop_status"] != "complete":
                    label = "censored"
                else:
                    ref = next(
                        x
                        for tid, xs in refs.items()
                        for x in xs
                        if x["episode_id"] == row["episode_id"]
                    )
                    tid = next(tid for tid, xs in refs.items() if ref in xs)
                    label = fp.role_label(
                        by_task[tid]["record"]["matches"],
                        ref["letter_key"],
                        scenarios[row["source_id"]],
                    )
                counts[(perm, label)] += 1
                f.write(
                    json.dumps(
                        {
                            "episode_id": row["episode_id"],
                            "permission": perm,
                            "source_id": row["source_id"],
                            "label": label,
                        }
                    )
                    + "\n"
                )
        summary = {}
        for perm in ("allowed", "forbidden"):
            n = sum(v for (q, _), v in counts.items() if q == perm)
            summary[perm] = {
                "n": n,
                **{lab: round(v / n, 4) for (q, lab), v in counts.items() if q == perm},
            }
        print(path.name, json.dumps(summary), f"spent_so_far=${spent[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
