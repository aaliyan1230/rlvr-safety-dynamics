"""H1-H5 for the capstone main runs: seed-level arm means and two-level bootstrap intervals.

Crossing = choosing the boundary action on the held-out 48 confirmation situations
(controls excluded). Action and letters are rule-scored; written plans use the judge labels
(`*.written-labels.jsonl`). Invalid, unjudgeable and censored answers stay in the denominator
and are reported separately. A run's rate is the mean over situations of that situation's
crossing share. Intervals resample situations (shared across runs, so paired) and, inside each
arm, training seeds; both levels of uncertainty are therefore included.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from score import role, score_executable, score_mcq

FORMATS = ("executable", "mcq", "option_free")
NAMES = {"executable": "action", "mcq": "letters", "option_free": "written"}
ARMS = ("blind", "aware", "shuffled")


def run_matrix(requests, scenarios, responses, labels, sources):
    """{(format, permission): array[situation] of crossing share}, plus problem rates."""
    hits: dict = defaultdict(lambda: defaultdict(list))
    problems: dict = defaultdict(int)
    totals: dict = defaultdict(int)
    for r in requests:
        if r["is_control"] or r["format"] not in FORMATS:
            continue
        key = (r["format"], r["permission"])
        s = scenarios[r["source_id"]]
        if r["format"] == "option_free":
            chosen = labels.get(r["episode_id"])
            if chosen is None:
                continue
            bad = chosen in ("unjudgeable", "censored")
        else:
            resp = responses.get(r["episode_id"])
            if resp is None:
                continue
            if r["format"] == "executable":
                action = score_executable(s, r, resp)["parsed_action"]
            else:
                action = score_mcq(r, resp, s)["selected_action_id"]
            chosen = role(s, action)
            bad = chosen == "invalid"
        hits[key][r["source_id"]].append(chosen == "boundary")
        problems[key] += bad
        totals[key] += 1
    mat = {
        k: np.array([np.mean(v[src]) if v[src] else np.nan for src in sources])
        for k, v in hits.items()
    }
    rates = {k: problems[k] / totals[k] for k in totals}
    return mat, rates


def contrasts(m):
    """m: {label: {key: value}} arm means -> named contrasts (percentage points)."""

    def g(arm, fmt, perm="forbidden"):
        return m[arm][(fmt, perm)]

    gap = lambda arm: g(arm, "executable") - g(arm, "option_free")  # noqa: E731
    return {
        "H1 action: blind - shuffled": g("blind", "executable") - g("shuffled", "executable"),
        "H2 letters: blind - shuffled": g("blind", "mcq") - g("shuffled", "mcq"),
        "H3 written: blind - shuffled": g("blind", "option_free") - g("shuffled", "option_free"),
        "H4 (action - written): blind - shuffled": gap("blind") - gap("shuffled"),
        "H5 action: aware - shuffled": g("aware", "executable") - g("shuffled", "executable"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True, help="directory of *.responses.jsonl")
    p.add_argument("--suffix", default="final", help="checkpoint label suffix to analyse")
    p.add_argument("--reps", type=int, default=10000)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    requests = [json.loads(x) for x in a.requests.open()]
    scenarios = {json.loads(x)["source_id"]: json.loads(x) for x in a.scenarios.open()}
    sources = sorted({r["source_id"] for r in requests if not r["is_control"]})

    runs: dict = {}
    pattern = re.compile(rf"^(parent|(blind|aware|shuffled)-s\d+-{a.suffix})\.responses\.jsonl$")
    for path in sorted(a.eval.glob("*.responses.jsonl")):
        match = pattern.match(path.name)
        if not match:
            continue
        label = match.group(1)
        responses = {json.loads(x)["episode_id"]: json.loads(x) for x in path.open()}
        lpath = path.with_name(path.name.replace(".responses.jsonl", ".written-labels.jsonl"))
        labels = (
            {json.loads(x)["episode_id"]: json.loads(x)["label"] for x in lpath.open()}
            if lpath.exists()
            else {}
        )
        runs[label] = run_matrix(requests, scenarios, responses, labels, sources)

    by_arm = {arm: [k for k in runs if k.startswith(arm + "-")] for arm in ARMS}
    keys = sorted({k for mat, _ in runs.values() for k in mat})
    report: dict = {"runs": {}, "arms": {}, "contrasts": {}, "suffix": a.suffix}
    for label, (mat, rates) in runs.items():
        report["runs"][label] = {
            f"{NAMES[f]}:{perm}": {
                "crossing": round(100 * float(np.nanmean(mat[(f, perm)])), 1),
                "invalid_or_unjudgeable": round(100 * rates[(f, perm)], 1),
            }
            for f, perm in keys
            if (f, perm) in mat
        }

    def arm_means(src_idx, seed_pick):
        out = {}
        for arm, labels in by_arm.items():
            chosen = seed_pick(labels)
            out[arm] = {
                k: float(np.nanmean([np.nanmean(runs[lab][0][k][src_idx]) for lab in chosen]))
                for k in keys
            }
        return out

    every = np.arange(len(sources))
    if all(by_arm.values()):
        point = arm_means(every, lambda labs: labs)
        report["arms"] = {
            arm: {f"{NAMES[f]}:{perm}": round(100 * v, 1) for (f, perm), v in m.items()}
            | {"seeds": by_arm[arm]}
            for arm, m in point.items()
        }
        rng = np.random.default_rng(0)
        draws = defaultdict(list)
        for _ in range(a.reps):
            idx = rng.integers(0, len(sources), len(sources))
            boot = arm_means(idx, lambda labs: list(rng.choice(labs, len(labs))))
            for name, v in contrasts(boot).items():
                draws[name].append(v)
        for name, v in contrasts(point).items():
            lo, hi = np.percentile(draws[name], [2.5, 97.5])
            report["contrasts"][name] = {
                "points": round(100 * v, 1),
                "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
            }
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
