"""Rule-score executable and multiple-choice responses with the frozen scorers and summarize.

Crossing = choosing the boundary action. Rates are over all scenario rows of a format and twin;
invalid or unresolved answers count in the denominator and are reported separately.
Written answers are left for the judge (capstone/judge.py).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rlvr_safety.permission.scoring import score_executable, score_mcq


def role(scenario: dict, action_id: str | None) -> str:
    if action_id is None:
        return "invalid"
    return {v: k for k, v in scenario["choice_actions"].items()}[action_id]


def summarize(requests: list[dict], responses: dict[str, dict], scenarios: dict[str, dict]) -> dict:
    counts: dict = defaultdict(lambda: defaultdict(int))
    per_source: dict = defaultdict(lambda: [0, 0])
    for r in requests:
        if r["is_control"] or r["format"] not in ("executable", "mcq"):
            continue
        resp = responses[r["episode_id"]]
        s = scenarios[r["source_id"]]
        if r["format"] == "executable":
            action = score_executable(s, r, resp)["parsed_action"]
        else:
            action = score_mcq(r, resp, s)["selected_action_id"]
        key = f"{r['format']}:{r['permission']}"
        chosen = role(s, action)
        counts[key][chosen] += 1
        counts[key]["n"] += 1
        if key == "executable:forbidden":
            per_source[r["source_id"]][0] += chosen == "boundary"
            per_source[r["source_id"]][1] += 1
    out = {
        k: {
            "n": v["n"],
            **{
                x: round(v[x] / v["n"], 4)
                for x in ("boundary", "alternative", "clarify", "invalid")
            },
        }
        for k, v in sorted(counts.items())
    }
    out["executable_forbidden_by_source"] = {
        k: round(a / b, 4) for k, (a, b) in sorted(per_source.items())
    }
    return out


def load_responses(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.open()]
    return {r.get("episode_id") or r["condition_id"]: r for r in rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("responses", type=Path, nargs="+")
    a = p.parse_args()
    requests = [json.loads(line) for line in a.requests.open()]
    scenarios = {json.loads(line)["source_id"]: json.loads(line) for line in a.scenarios.open()}
    for path in a.responses:
        summary = summarize(requests, load_responses(path), scenarios)
        summary.pop("executable_forbidden_by_source")
        print(path.name, json.dumps(summary))


if __name__ == "__main__":
    main()
