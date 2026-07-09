#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def load_model_key(path: Path) -> dict[str, str]:
    return {row["model_alias"]: row["model"] for row in read_csv(path)}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def load_scores(path: Path, model_key: dict[str, str] | None = None) -> dict[tuple[str, str], int]:
    scores = {}
    for row in read_csv(path):
        score = row.get("instrumental_score_0_2", "")
        if score == "":
            continue
        model = row["model"]
        if model_key:
            model = model_key.get(model, model)
        source_id = row.get("source_id", row.get("id", ""))
        scores[(source_id, model)] = int(score)
    return scores


def load_choice_scores_csv(path: Path, model_key: dict[str, str] | None = None) -> list[dict]:
    rows = []
    for row in read_csv(path):
        score = row.get("instrumental_score_0_2", "")
        if score == "":
            continue
        model = row["model"]
        if model_key:
            model = model_key.get(model, model)
        rows.append({
            "id": row.get("id", ""),
            "source_id": row.get("source_id", ""),
            "model": model,
            "category": row.get("category", ""),
            "score": int(score),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-scores", type=Path, required=True)
    parser.add_argument("--original-model-key", type=Path, default=None)
    parser.add_argument("--paraphrase-scores", type=Path, required=True)
    parser.add_argument("--paraphrase-model-key", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=Path("reports/prompt_sensitivity.md"))
    args = parser.parse_args()

    orig_key = load_model_key(args.original_model_key) if args.original_model_key else {}
    para_key = load_model_key(args.paraphrase_model_key) if args.paraphrase_model_key else {}

    orig_rows = load_choice_scores_csv(args.original_scores, orig_key)
    para_rows = load_choice_scores_csv(args.paraphrase_scores, para_key)

    # Build lookup: (source_id, model) -> score for original
    orig_lookup = {}
    for row in orig_rows:
        orig_lookup[(row["source_id"], row["model"])] = row

    # Join paraphrase rows to original by (source_id, model)
    paired = []
    for p in para_rows:
        key = (p["source_id"], p["model"])
        o = orig_lookup.get(key)
        if o is not None:
            paired.append({
                "source_id": key[0],
                "model": key[1],
                "category": p["category"],
                "original_score": o["score"],
                "paraphrase_score": p["score"],
                "delta": p["score"] - o["score"],
                "abs_delta": abs(p["score"] - o["score"]),
                "flipped": p["score"] != o["score"],
            })

    if not paired:
        print("ERROR: No paired rows. Check that model names and source_ids align.")
        return 1

    lines = ["# Prompt Sensitivity Analysis: Original vs Paraphrase", ""]
    lines.append(f"Paired rows: {len(paired)}")
    lines.append("")

    # --- Model means ---
    by_model = defaultdict(list)
    for p in paired:
        by_model[p["model"]].append(p)

    lines.append("## Per-Model Aggregate")
    lines.append("")
    lines.append("| Model | Original mean | Paraphrase mean | Delta | Mean abs delta | Items flipped |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in sorted(by_model):
        rows = by_model[model]
        orig_mean = mean(r["original_score"] for r in rows)
        para_mean = mean(r["paraphrase_score"] for r in rows)
        mean_abs_delta = mean(r["abs_delta"] for r in rows)
        flipped = sum(1 for r in rows if r["flipped"])
        lines.append(
            f"| {model} | {orig_mean:.2f} | {para_mean:.2f} | +{para_mean - orig_mean:.2f} | "
            f"{mean_abs_delta:.2f} | {flipped}/{len(rows)} |"
        )
    lines.append("")

    # --- Category deltas ---
    lines.append("## Category Deltas by Model")
    lines.append("")
    lines.append("| Model | Category | Original | Paraphrase | Delta | Flipped |")
    lines.append("|---|---|---:|---:|---:|---:|")
    by_model_cat = defaultdict(list)
    for p in paired:
        by_model_cat[(p["model"], p["category"])].append(p)
    for (model, category), rows in sorted(by_model_cat.items()):
        orig_mean = mean(r["original_score"] for r in rows)
        para_mean = mean(r["paraphrase_score"] for r in rows)
        flipped = sum(1 for r in rows if r["flipped"])
        lines.append(
            f"| {model} | {category} | {orig_mean:.2f} | {para_mean:.2f} | "
            f"{para_mean - orig_mean:+.2f} | {flipped}/{len(rows)} |"
        )
    lines.append("")

    # --- Flip matrix (how scores changed) ---
    lines.append("## Score Change Matrix")
    lines.append("")
    lines.append("| Model | 0->0 | 0->1 | 0->2 | 1->0 | 1->1 | 1->2 | 2->0 | 2->1 | 2->2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in sorted(by_model):
        rows = by_model[model]
        matrix = Counter()
        for r in rows:
            matrix[(r["original_score"], r["paraphrase_score"])] += 1
        cells = [matrix.get((o, p), 0) for o in range(3) for p in range(3)]
        lines.append(f"| {model} | " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")

    # --- Spread comparison: model-stage vs prompt/source ---
    lines.append("## Spread: Model-Stage vs Prompt-Source Variability")
    lines.append("")
    lines.append("Variance decomposition: how much of the score variance is explained by model identity vs prompt wording.")
    lines.append("")

    # Variance by model (across prompts, both original and paraphrase)
    from_model = []
    for model in sorted(by_model):
        rows = by_model[model]
        orig_vals = [r["original_score"] for r in rows]
        para_vals = [r["paraphrase_score"] for r in rows]
        mean_o = mean(orig_vals)
        mean_p = mean(para_vals)
        orig_var = sum((v - mean_o) ** 2 for v in orig_vals) / len(orig_vals) if len(orig_vals) > 1 else 0
        para_var = sum((v - mean_p) ** 2 for v in para_vals) / len(para_vals) if len(para_vals) > 1 else 0
        from_model.append((model, orig_var, para_var, mean_o, mean_p))

    lines.append("| Model | Original variance | Paraphrase variance |")
    lines.append("|---|---:|---:|")
    for model, ov, pv, *_ in from_model:
        lines.append(f"| {model} | {ov:.3f} | {pv:.3f} |")
    lines.append("")

    # Variance across models within each prompt set
    lines.append("| Source | Variance across models |")
    lines.append("|---|---:|")
    for group_name, rows_group in [("original", orig_rows), ("paraphrase", para_rows)]:
        by_source = defaultdict(list)
        for row in rows_group:
            by_source[row["model"]].append(row["score"])
        model_means = [mean(v) for v in by_source.values()]
        var_across = mean((m - mean(model_means)) ** 2 for m in model_means) if len(model_means) > 1 else 0
        lines.append(f"| {group_name} | {var_across:.3f} |")
    lines.append("")

    # Per-source_id variance across models
    lines.append("## Item-Level Agreement: How Often Did Individual Items Change?")
    lines.append("")
    for model in sorted(by_model):
        rows = by_model[model]
        flipped = [r for r in rows if r["flipped"]]
        lines.append(f"- **{model}**: {len(flipped)}/{len(rows)} items changed score")
        if flipped:
            lines.append("  - Increased: " + ", ".join(
                f"{r['source_id']} ({r['original_score']}->{r['paraphrase_score']})"
                for r in sorted(flipped, key=lambda x: x["delta"], reverse=True) if r["delta"] > 0
            ))
            lines.append("  - Decreased: " + ", ".join(
                f"{r['source_id']} ({r['original_score']}->{r['paraphrase_score']})"
                for r in sorted(flipped, key=lambda x: x["delta"]) if r["delta"] < 0
            ))
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote prompt sensitivity analysis to {args.out_md}")


if __name__ == "__main__":
    main()
