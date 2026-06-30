#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from gemini_common import load_jsonl


def load_manual_csv(path: Path, score_field: str) -> dict[tuple[str, str], int]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(score_field, "") == "":
                continue
            rows[(row.get("id", ""), row.get("model", ""))] = int(row[score_field])
    return rows


def load_model_key(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    aliases = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = row.get("model_alias", "")
            model = row.get("model", "")
            if alias and model:
                aliases[model] = alias
    return aliases


def summarize(rows: list[dict]) -> list[str]:
    by_model = defaultdict(list)
    by_failure = defaultdict(Counter)
    for row in rows:
        by_model[row.get("model", "")].append(row)
        by_failure[row.get("model", "")][row.get("primary_failure_mode", "unknown")] += 1

    lines = ["# Judgeability Analysis", ""]
    lines.append("| Model | Mean risk | Mean judgeability problem | Needs review | Failure modes |")
    lines.append("|---|---:|---:|---:|---|")
    for model in sorted(by_model):
        model_rows = by_model[model]
        risk = [int(row["instrumental_score_0_2"]) for row in model_rows]
        judgeability = [int(row["judgeability_score_0_2"]) for row in model_rows]
        review = sum(str(row.get("needs_human_review")).lower() == "true" for row in model_rows)
        failures = ", ".join(f"{key}:{value}" for key, value in sorted(by_failure[model].items()))
        lines.append(
            f"| `{model}` | {sum(risk)/len(risk):.2f} | {sum(judgeability)/len(judgeability):.2f} | "
            f"{review}/{len(model_rows)} | {failures} |"
        )
    return lines


def agreement_lines(rows: list[dict], manual: dict[tuple[str, str], int], model_key: dict[str, str]) -> list[str]:
    comparable = []
    for row in rows:
        model = model_key.get(row.get("model", ""), row.get("model", ""))
        key = (row.get("id", ""), model)
        if key in manual:
            comparable.append((manual[key], int(row["instrumental_score_0_2"])))
    lines = ["", "## Manual Agreement", ""]
    if not comparable:
        lines.append("No comparable manual rows found for exact `(id, model)` keys.")
        return lines
    exact = sum(a == b for a, b in comparable)
    binary = sum((a > 0) == (b > 0) for a, b in comparable)
    lines.append(f"- Comparable rows: {len(comparable)}")
    lines.append(f"- Exact 0/1/2 agreement: {exact}/{len(comparable)} ({exact/len(comparable):.2%})")
    lines.append(f"- Binary nonzero-risk agreement: {binary}/{len(comparable)} ({binary/len(comparable):.2%})")
    matrix = Counter(comparable)
    lines.extend(["", "| Manual | Gemini | Count |", "|---:|---:|---:|"])
    for (manual_score, gemini_score), count in sorted(matrix.items()):
        lines.append(f"| {manual_score} | {gemini_score} | {count} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-results", type=Path, nargs="+", required=True)
    parser.add_argument("--manual-scores", type=Path)
    parser.add_argument("--model-key", type=Path)
    parser.add_argument("--manual-score-field", default="instrumental_score_0_2")
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.judge_results:
        rows.extend(load_jsonl(path))

    lines = summarize(rows)
    if args.manual_scores:
        manual = load_manual_csv(args.manual_scores, args.manual_score_field)
        model_key = load_model_key(args.model_key)
        lines.extend(agreement_lines(rows, manual, model_key))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote judgeability analysis to {args.out_md}")


if __name__ == "__main__":
    main()
