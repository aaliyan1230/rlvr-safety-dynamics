#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def model_key(path: Path) -> dict[str, str]:
    return {row["model_alias"]: row["model"] for row in read_csv(path)}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score_counts(values):
    counts = Counter(values)
    return f"{counts[0]}x0, {counts[1]}x1, {counts[2]}x2"


def manual_freeform_table(path: Path, key_path: Path) -> list[str]:
    aliases = model_key(key_path)
    by_model = defaultdict(list)
    for row in read_csv(path):
        if row["category"] == "benign_control" or row["instrumental_score_0_2"] == "":
            continue
        by_model[row["model"]].append(row)

    lines = ["## Table 1: Free-Form Manual Scores", ""]
    lines.append("| Model | Mean risk | Counts | Confounded risk rows |")
    lines.append("|---|---:|---|---:|")
    for alias in sorted(by_model):
        rows = by_model[alias]
        values = [int(row["instrumental_score_0_2"]) for row in rows]
        confounded = sum(row.get("verbosity_confounded") == "true" for row in rows)
        lines.append(
            f"| {aliases.get(alias, alias)} | {mean(values):.2f} | {score_counts(values)} | {confounded}/{len(rows)} |"
        )
    return lines


def choice_table(path: Path, key_path: Path) -> list[str]:
    aliases = model_key(key_path)
    by_model = defaultdict(list)
    for row in read_csv(path):
        by_model[row["model"]].append(row)

    lines = ["", "## Table 2: Structured Choice Scores", ""]
    lines.append("| Model | Mean parsed risk | Counts | Malformed | Token-cap rows |")
    lines.append("|---|---:|---|---:|---:|")
    for alias in sorted(by_model):
        rows = by_model[alias]
        values = [int(row["instrumental_score_0_2"]) for row in rows if row["instrumental_score_0_2"] != ""]
        malformed = sum(row.get("malformed") == "true" for row in rows)
        capped = sum(row.get("verbosity_confounded") == "true" for row in rows)
        lines.append(
            f"| {aliases.get(alias, alias)} | {mean(values):.2f} | {score_counts(values)} | {malformed}/{len(rows)} | {capped}/{len(rows)} |"
        )
    return lines


def gemini_judgeability_table(paths: list[Path]) -> list[str]:
    by_model = defaultdict(list)
    for path in paths:
        label = path.stem
        for row in read_jsonl(path):
            item = dict(row)
            item["_source"] = label
            by_model[(label, row["model"])].append(item)

    lines = ["", "## Table 3: Gemini Judgeability", ""]
    lines.append("| Source | Model | Mean Gemini risk | Mean judgeability problem | Needs review | Failure modes |")
    lines.append("|---|---|---:|---:|---:|---|")
    for (source, model), rows in sorted(by_model.items()):
        risk = [int(row["instrumental_score_0_2"]) for row in rows]
        judgeability = [int(row["judgeability_score_0_2"]) for row in rows]
        review = sum(bool(row["needs_human_review"]) for row in rows)
        failures = Counter(row["primary_failure_mode"] for row in rows)
        failure_text = ", ".join(f"{name}:{count}" for name, count in sorted(failures.items()))
        lines.append(
            f"| {source} | {model} | {mean(risk):.2f} | {mean(judgeability):.2f} | {review}/{len(rows)} | {failure_text} |"
        )
    return lines


def agreement_table(path: Path, manual_path: Path, key_path: Path | None) -> list[str]:
    aliases = {}
    if key_path is not None:
        aliases = {row["model"]: row["model_alias"] for row in read_csv(key_path)}
    manual = {}
    for row in read_csv(manual_path):
        if row.get("instrumental_score_0_2", "") != "":
            manual[(row["id"], row["model"])] = {
                "score": int(row["instrumental_score_0_2"]),
                "category": row.get("category", ""),
            }

    pairs = []
    for row in read_jsonl(path):
        model = aliases.get(row["model"], row["model"])
        key = (row["id"], model)
        if key in manual:
            pairs.append(
                (
                    manual[key]["score"],
                    int(row["instrumental_score_0_2"]),
                    manual[key]["category"],
                )
            )

    lines = ["", f"## Agreement: {path.stem}", ""]
    if not pairs:
        lines.append("No comparable rows.")
        return lines
    exact = sum(a == b for a, b, _ in pairs)
    binary = sum((a > 0) == (b > 0) for a, b, _ in pairs)
    risk_pairs = [(a, b) for a, b, category in pairs if category != "benign_control"]
    risk_exact = sum(a == b for a, b in risk_pairs)
    risk_binary = sum((a > 0) == (b > 0) for a, b in risk_pairs)
    lines.append(f"* Comparable rows (all): {len(pairs)}")
    lines.append(f"* Exact agreement (all): {exact}/{len(pairs)} ({exact / len(pairs):.2%})")
    lines.append(
        f"* Binary nonzero-risk agreement (all): {binary}/{len(pairs)} "
        f"({binary / len(pairs):.2%})"
    )
    if risk_pairs != [(a, b) for a, b, _ in pairs]:
        lines.append(
            f"* Exact agreement (risk prompts only): {risk_exact}/{len(risk_pairs)} "
            f"({risk_exact / len(risk_pairs):.2%})"
        )
        lines.append(
            f"* Binary agreement (risk prompts only): {risk_binary}/{len(risk_pairs)} "
            f"({risk_binary / len(risk_pairs):.2%})"
        )
    lines.extend(["", "| Manual | Gemini | Count |", "|---:|---:|---:|"])
    for (manual_score, gemini_score), count in sorted(
        Counter((a, b) for a, b, _ in pairs).items()
    ):
        lines.append(f"| {manual_score} | {gemini_score} | {count} |")
    return lines


def paraphrase_comparison_table(orig_scores: Path, orig_key: Path, para_scores: Path) -> list[str]:
    orig_by_source: dict[tuple[str, str], int] = {}
    for row in read_csv(orig_scores):
        score = row.get("instrumental_score_0_2", "")
        if score == "":
            continue
        model_key_map = model_key(orig_key)
        model = model_key_map.get(row["model"], row["model"])
        orig_by_source[(row["source_id"], model)] = int(score)

    paired = []
    for row in read_csv(para_scores):
        score = row.get("instrumental_score_0_2", "")
        if score == "":
            continue
        key = (row["source_id"], row["model"])
        orig_score = orig_by_source.get(key)
        if orig_score is not None:
            paired.append({
                "model": row["model"],
                "category": row["category"],
                "orig": orig_score,
                "para": int(score),
            })

    if not paired:
        return ["", "## Table 4: Paraphrase Comparison", "", "No paired rows available.", ""]

    by_model = defaultdict(list)
    for p in paired:
        by_model[p["model"]].append(p)

    lines = [
        "",
        "## Table 4: Legacy Confounded Protocol Shift (Original vs P1)",
        "",
        "**Diagnostic only.** Wording and semantic option position changed together, so these "
        "deltas do not identify a wording effect.",
        "",
    ]
    lines.append("| Model | Original mean | Paraphrase mean | Delta | Mean abs delta | Items changed |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in sorted(by_model):
        rows = by_model[model]
        orig_mean = mean(r["orig"] for r in rows)
        para_mean = mean(r["para"] for r in rows)
        abs_delta = mean(abs(r["para"] - r["orig"]) for r in rows)
        changed = sum(1 for r in rows if r["para"] != r["orig"])
        lines.append(
            f"| {model} | {orig_mean:.2f} | {para_mean:.2f} | "
            f"{para_mean - orig_mean:+.2f} | {abs_delta:.2f} | {changed}/{len(rows)} |"
        )
    return lines


def paraphrase_combined_table(orig_scores: Path, orig_key: Path, para1_scores: Path, para2_scores: Path) -> list[str]:
    aliases = model_key(orig_key)

    def paired_rows(para_path: Path) -> dict[tuple[str, str], dict]:
        orig_by_source: dict[tuple[str, str], int] = {}
        for row in read_csv(orig_scores):
            score = row.get("instrumental_score_0_2", "")
            if score == "":
                continue
            model = aliases.get(row["model"], row["model"])
            orig_by_source[(row["source_id"], model)] = int(score)

        paired = {}
        for row in read_csv(para_path):
            score = row.get("instrumental_score_0_2", "")
            if score == "":
                continue
            key = (row["source_id"], row["model"])
            orig_score = orig_by_source.get(key)
            if orig_score is not None:
                paired[key] = {
                    "model": row["model"],
                    "category": row["category"],
                    "orig": orig_score,
                    "para": int(score),
                }
        return paired

    seed1_paired = paired_rows(para1_scores)
    seed2_paired = paired_rows(para2_scores)

    all_models = sorted(set(p["model"] for p in seed1_paired.values()))

    lines = [
        "",
        "## Table 5: Legacy Confounded Protocol Shift (P1 and P2)",
        "",
        "**Diagnostic only.** Both candidate packs fixed safe=A, ambiguous=B, and risky=C "
        "while the original varied score positions. The values combine wording and position effects.",
        "",
    ]
    lines.append("| Model | Orig | Seed 1 mean | Seed 2 mean | Seed 1 delta | Seed 2 delta | Seed 1 flip | Seed 2 flip | Seed 1 MAD | Seed 2 MAD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    per_model = defaultdict(list)
    for model in all_models:
        per_model[model] = {
            "orig": mean(p["orig"] for k, p in seed1_paired.items() if p["model"] == model),
            "seed1_mean": mean(p["para"] for k, p in seed1_paired.items() if p["model"] == model),
            "seed2_mean": mean(p["para"] for k, p in seed2_paired.items() if p["model"] == model),
            "seed1_flip": sum(1 for k, p in seed1_paired.items() if p["model"] == model and p["para"] != p["orig"]),
            "seed2_flip": sum(1 for k, p in seed2_paired.items() if p["model"] == model and p["para"] != p["orig"]),
            "seed1_n": sum(1 for p in seed1_paired.values() if p["model"] == model),
            "seed2_n": sum(1 for p in seed2_paired.values() if p["model"] == model),
            "seed1_abs": mean(abs(p["para"] - p["orig"]) for k, p in seed1_paired.items() if p["model"] == model),
            "seed2_abs": mean(abs(p["para"] - p["orig"]) for k, p in seed2_paired.items() if p["model"] == model),
        }

    for model in all_models:
        m = per_model[model]
        s1d = m["seed1_mean"] - m["orig"]
        s2d = m["seed2_mean"] - m["orig"]
        lines.append(
            f"| {model} | {m['orig']:.2f} | {m['seed1_mean']:.2f} | {m['seed2_mean']:.2f} | "
            f"{s1d:+.2f} | {s2d:+.2f} | {m['seed1_flip']}/{m['seed1_n']} | {m['seed2_flip']}/{m['seed2_n']} | "
            f"{m['seed1_abs']:.2f} | {m['seed2_abs']:.2f} |"
        )

    lines.append("")
    lines.append("### Category-Level Deltas")
    lines.append("")
    lines.append("| Model | Category | Orig | Seed 1 mean | Seed 2 mean | Seed 1 delta | Seed 2 delta | Seed 1 flip | Seed 2 flip |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for model in all_models:
        cats = sorted(set(
            p["category"] for p in seed1_paired.values() if p["model"] == model
        ))
        for cat in cats:
            s1_rows = [(k, p) for k, p in seed1_paired.items() if p["model"] == model and p["category"] == cat]
            s2_rows = [(k, p) for k, p in seed2_paired.items() if p["model"] == model and p["category"] == cat]
            orig_m = mean(p["orig"] for _, p in s1_rows)
            s1_m = mean(p["para"] for _, p in s1_rows)
            s2_m = mean(p["para"] for _, p in s2_rows)
            s1_f = sum(1 for _, p in s1_rows if p["para"] != p["orig"])
            s2_f = sum(1 for _, p in s2_rows if p["para"] != p["orig"])
            lines.append(
                f"| {model} | {cat} | {orig_m:.2f} | {s1_m:.2f} | {s2_m:.2f} | "
                f"{s1_m - orig_m:+.2f} | {s2_m - orig_m:+.2f} | {s1_f}/{len(s1_rows)} | {s2_f}/{len(s2_rows)} |"
            )

    lines.append("")
    lines.append(
        "*P1 was evaluated 2026-07-06 and P2 on 2026-07-08. Their historical checks "
        "confirmed internal label ordering but did not compare candidates with source items.*"
    )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/paper_tables.md"))
    parser.add_argument("--freeform-manual", type=Path, default=Path("artifacts/baseline/freeform_manual_scores.csv"))
    parser.add_argument("--freeform-key", type=Path, default=Path("artifacts/baseline/freeform_model_key.csv"))
    parser.add_argument("--choice-scores", type=Path, default=Path("artifacts/baseline/choice_stage_scores.csv"))
    parser.add_argument("--choice-key", type=Path, default=Path("artifacts/baseline/choice_stage_model_key.csv"))
    parser.add_argument("--gemini-freeform", type=Path, default=Path("artifacts/baseline/gemini_freeform_judgments.jsonl"))
    parser.add_argument("--gemini-choice", type=Path, default=Path("artifacts/baseline/gemini_choice_judgments.jsonl"))
    parser.add_argument("--paraphrase-scores", type=Path, default=Path("artifacts/baseline/paraphrase_p1_scores_confounded.csv"))
    parser.add_argument("--paraphrase-scores-seed2", type=Path, default=Path("artifacts/baseline/paraphrase_p2_scores_confounded.csv"))
    args = parser.parse_args()

    lines = ["# Paper Tables", ""]
    lines.extend(manual_freeform_table(args.freeform_manual, args.freeform_key))
    lines.extend(choice_table(args.choice_scores, args.choice_key))
    lines.extend(gemini_judgeability_table([args.gemini_freeform, args.gemini_choice]))
    lines.extend(agreement_table(args.gemini_freeform, args.freeform_manual, args.freeform_key))
    lines.extend(agreement_table(args.gemini_choice, args.choice_scores, args.choice_key))
    lines.extend(paraphrase_comparison_table(args.choice_scores, args.choice_key, args.paraphrase_scores))
    lines.extend(paraphrase_combined_table(args.choice_scores, args.choice_key, args.paraphrase_scores, args.paraphrase_scores_seed2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote paper tables to {args.out}")


if __name__ == "__main__":
    main()
