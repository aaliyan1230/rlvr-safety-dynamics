# Agent-task pipeline

One audited path for every job where an AI system produces data: drafting scenarios, judging
unresolved multiple-choice replies, pre-labelling free-text answers. The pipeline does not care
which AI answers; only the *backend* changes.

```
plan  ->  tasks/<id>.json  ->  [any backend]  ->  responses/<id>.json  ->  ingest  ->  audit
```

Each task is self-contained (instructions, input, response schema) and hashed. A response is an
envelope `{task_id, task_sha256, agent, model_reported, backend?, payload}`; ingest checks it names
its task, validates the payload with the job's validator, and appends the accepted record (with
agent, model, response hash, time, operator) to `ingested.jsonl`. Rejected responses go to
`rejections.jsonl` with the reason and are never deleted. `audit` re-hashes everything.

## Backends

| Backend | Use it when | How |
|---|---|---|
| File (default) | A coding agent (Claude Code, Codex, opencode) or a person answers | Read `tasks/<id>.json`, write `responses/<id>.json` with the envelope above |
| `CommandBackend` | Any CLI can answer | Task JSON on stdin, payload JSON on stdout: `CommandBackend(["codex", "exec", "..."], agent="codex", model_reported="...")` |
| `CallableBackend` | An LLM API (Gemini, OpenAI, Anthropic, local model) | Wrap `fn(task) -> JSON text`; see the template below |

```python
from rlvr_safety.agent_tasks.core import CallableBackend, run_backend

def answer(task: dict) -> str:
    prompt = task["instructions"] + "\n\nInput:\n" + json.dumps(task["input"]) \
        + "\n\nresponse_schema:\n" + json.dumps(task["response_schema"])
    return call_my_model(prompt)          # must return the payload JSON text

run_backend(job_dir, CallableBackend(answer, agent="my-api", model_reported="model-id"))
```

## Jobs (`jobs/`)

| Job | Rubric ID | Input | Output |
|---|---|---|---|
| `authoring` | `authoring-2026-10-07` | A brief (family, count, requirements) | Draft scenario/control records, always `review.status = pending`; mechanical checks must pass |
| `mcq_judge` | `mcq-judge-2026-10-09` | Options + reply only (blinded) for replies the MCQ rule could not resolve | Letter or `none`, with a verbatim quote |
| `freeform_prelabel` | `freeform-label-2026-10-09` | Situation without the permission sentence, shuffled lettered options, answer | Blinded label, translated to a role afterwards; not authoritative |

AI judgments never replace human decisions: judge scores are stored separately from the rule
score, and humans label every judged reply in the diagnostic before the judge is trusted.
