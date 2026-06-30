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
            manual[(row["id"], row["model"])] = int(row["instrumental_score_0_2"])

    pairs = []
    for row in read_jsonl(path):
        model = aliases.get(row["model"], row["model"])
        key = (row["id"], model)
        if key in manual:
            pairs.append((manual[key], int(row["instrumental_score_0_2"])))

    lines = ["", f"## Agreement: {path.stem}", ""]
    if not pairs:
        lines.append("No comparable rows.")
        return lines
    exact = sum(a == b for a, b in pairs)
    binary = sum((a > 0) == (b > 0) for a, b in pairs)
    lines.append(f"* Comparable rows: {len(pairs)}")
    lines.append(f"* Exact agreement: {exact}/{len(pairs)} ({exact / len(pairs):.2%})")
    lines.append(f"* Binary nonzero-risk agreement: {binary}/{len(pairs)} ({binary / len(pairs):.2%})")
    lines.extend(["", "| Manual | Gemini | Count |", "|---:|---:|---:|"])
    for (manual_score, gemini_score), count in sorted(Counter(pairs).items()):
        lines.append(f"| {manual_score} | {gemini_score} | {count} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/paper_tables.md"))
    parser.add_argument("--freeform-manual", type=Path, default=Path("results/kaggle_olmo_rlzero_full_v1/manual_scores_olmo3_rlzero_full.scored.csv"))
    parser.add_argument("--freeform-key", type=Path, default=Path("results/kaggle_olmo_rlzero_full_v1/manual_scores_olmo3_rlzero_full.model_key.csv"))
    parser.add_argument("--choice-scores", type=Path, default=Path("results/kaggle_choice_stage_ablation_v2/choice_scores.csv"))
    parser.add_argument("--choice-key", type=Path, default=Path("results/kaggle_choice_stage_ablation_v2/choice_score_model_key.csv"))
    parser.add_argument("--gemini-freeform", type=Path, default=Path("results/gemini_olmo_rlzero_full_v1.jsonl"))
    parser.add_argument("--gemini-choice", type=Path, default=Path("results/gemini_choice_stage_ablation_v2.jsonl"))
    args = parser.parse_args()

    lines = ["# Paper Tables", ""]
    lines.extend(manual_freeform_table(args.freeform_manual, args.freeform_key))
    lines.extend(choice_table(args.choice_scores, args.choice_key))
    lines.extend(gemini_judgeability_table([args.gemini_freeform, args.gemini_choice]))
    lines.extend(agreement_table(args.gemini_freeform, args.freeform_manual, args.freeform_key))
    lines.extend(agreement_table(args.gemini_choice, args.choice_scores, args.choice_key))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote paper tables to {args.out}")


if __name__ == "__main__":
    main()
