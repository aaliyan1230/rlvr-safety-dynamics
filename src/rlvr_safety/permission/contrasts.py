"""Pre-registered cross-format contrasts: say vs do (H1), written vs MCQ (H2), late vs early.

Every contrast is a per-source difference averaged over sources. Inference resamples sources
(paired: the same sources enter both sides), with a one-sided bootstrap p-value for the
directional hypotheses and Holm's correction across the primary family.

H1 uses the identity P(do & not say) - P(say & not do) = P(do) - P(say): in the forbidden twin the
say-do asymmetry equals the executable crossing rate minus the written crossing rate, so it needs
no paired draws and applies unchanged to sampled runs with several draws per cell.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable

from .analysis import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    MIN_SOURCES_FOR_INTERVAL,
    _outcome_executable,
    _outcome_mcq_rule,
    _per_source,
    make_outcome_labels,
    make_outcome_mcq_judge,
)

CONTRASTS_VERSION = "permission-contrasts-2026-10-15"


def source_rates(rows: list[dict], fmt: str, outcome: Callable, scenarios: dict) -> dict:
    """{source: {"allowed": rate, "forbidden": rate}} for one format, scenarios only."""
    picked = [r for r in rows if not r["is_control"] and r["format"] == fmt]
    grouped = _per_source(picked, outcome, scenarios)
    return {
        source: {p: statistics.fmean(v) for p, v in cells.items() if v}
        for source, cells in grouped.items()
    }


def bootstrap(values: dict[str, float], *, seed: int = BOOTSTRAP_SEED,
              reps: int = BOOTSTRAP_REPS) -> dict:
    """Mean over sources, percentile 95% interval and one-sided p for H: mean > 0."""
    sources = sorted(values)
    result = {"sources": len(sources), "estimate": None, "interval_95": None, "p_one_sided": None}
    if not sources:
        return result
    data = [values[s] for s in sources]
    result["estimate"] = statistics.fmean(data)
    if len(sources) < MIN_SOURCES_FOR_INTERVAL:
        return result
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(data, k=len(data))) for _ in range(reps))
    result["interval_95"] = [draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]]
    result["p_one_sided"] = (1 + sum(d <= 0 for d in draws)) / (reps + 1)
    return result


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values (monotone, capped at 1); missing p stays missing."""
    present = sorted((p, k) for k, p in p_values.items() if p is not None)
    m, running, adjusted = len(present), 0.0, {k: None for k in p_values}
    for rank, (p, key) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def _sens(rates: dict) -> dict[str, float]:
    return {s: c["allowed"] - c["forbidden"] for s, c in rates.items()
            if "allowed" in c and "forbidden" in c}


def model_contrasts(rows: list[dict], scenarios: dict, labels: dict[str, str],
                    judge: dict | None = None) -> dict:
    """H1, H2 and secondary contrasts for one model; ``labels`` maps episode_id to a role."""
    written = source_rates(rows, "option_free", make_outcome_labels(labels), scenarios)
    executable = source_rates(rows, "executable", _outcome_executable, scenarios)
    mcq_outcome = make_outcome_mcq_judge(judge) if judge else _outcome_mcq_rule
    mcq = source_rates(rows, "mcq", mcq_outcome, scenarios)
    h1 = {s: executable[s]["forbidden"] - written[s]["forbidden"]
          for s in written.keys() & executable.keys()
          if "forbidden" in written[s] and "forbidden" in executable[s]}
    sens = {"written": _sens(written), "executable": _sens(executable), "mcq": _sens(mcq)}
    h2 = {s: sens["written"][s] - sens["mcq"][s]
          for s in sens["written"].keys() & sens["mcq"].keys()}
    exe_vs_mcq = {s: sens["executable"][s] - sens["mcq"][s]
                  for s in sens["executable"].keys() & sens["mcq"].keys()}
    families = sorted({scenarios[s]["family"] for s in h1})
    return {
        "H1_say_do_asymmetry": bootstrap(h1),
        "H2_written_minus_mcq_sensitivity": bootstrap(h2),
        "secondary": {
            "executable_minus_mcq_sensitivity": bootstrap(exe_vs_mcq),
            "sensitivity": {f: bootstrap(v) for f, v in sens.items()},
            "forbidden_crossing": {
                "written": bootstrap({s: c["forbidden"] for s, c in written.items()
                                      if "forbidden" in c}),
                "executable": bootstrap({s: c["forbidden"] for s, c in executable.items()
                                         if "forbidden" in c}),
            },
            "by_family": {
                fam: {
                    "H1": bootstrap({s: v for s, v in h1.items()
                                     if scenarios[s]["family"] == fam}),
                    "H2": bootstrap({s: v for s, v in h2.items()
                                     if scenarios[s]["family"] == fam}),
                }
                for fam in families
            },
        },
        "per_source": {
            s: {"h1": h1.get(s), "h2": h2.get(s),
                "written": written.get(s), "executable": executable.get(s), "mcq": mcq.get(s)}
            for s in sorted(set(written) | set(executable) | set(mcq))
        },
        "_sens": sens,
    }


def report(
    per_model: dict[str, dict], *, early: str | None = None, late: str | None = None
) -> dict:
    """Combine model contrasts, apply Holm across H1/H2 x models, add late - early per format."""
    primaries = {}
    for label, result in per_model.items():
        primaries[f"H1|{label}"] = result["H1_say_do_asymmetry"]["p_one_sided"]
        primaries[f"H2|{label}"] = result["H2_written_minus_mcq_sensitivity"]["p_one_sided"]
    adjusted = holm(primaries)
    out = {"version": CONTRASTS_VERSION, "alpha_one_sided": 0.025, "models": {}, "primary": {}}
    for key, p in adjusted.items():
        hypothesis, label = key.split("|")
        name = "H1_say_do_asymmetry" if hypothesis == "H1" else "H2_written_minus_mcq_sensitivity"
        result = per_model[label][name]
        out["primary"][key] = {**result, "p_holm": p,
                               "supported": p is not None and p < 0.025}
    for label, result in per_model.items():
        out["models"][label] = {k: v for k, v in result.items() if k != "_sens"}
    if early and late and early in per_model and late in per_model:
        change = {}
        for fmt in ("written", "mcq", "executable"):
            a, b = per_model[early]["_sens"][fmt], per_model[late]["_sens"][fmt]
            deltas = {s: b[s] - a[s] for s in a.keys() & b.keys()}
            boot = bootstrap(deltas)
            change[fmt] = {k: boot[k] for k in ("sources", "estimate", "interval_95")}
        out["late_minus_early_sensitivity"] = change
    return out
