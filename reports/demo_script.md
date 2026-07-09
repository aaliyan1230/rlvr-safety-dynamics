# Project Demo Script

## Three-Minute Version

### 0:00 - Research Question

I wanted to test whether current small open reasoning or RLVR-lineage models show stronger instrumental-convergence-style behavior than matched instruction models.

The behaviors I checked were goal persistence, resource acquisition, self-preservation, deception, and evaluation awareness.

### 0:30 - Method

I built a reproducible evaluation scaffold:

* A 48-prompt seed set.
* A manual 0-2 scoring rubric.
* A Kaggle-ready generation runner.
* A structured choice follow-up eval.
* A Gemini judgeability pass using the official SDK.
* A table-generation script for paper-facing results.

### 1:10 - Experiments

I ran:

* Qwen3 4B Instruct vs Thinking.
* OLMo 3 Instruct vs RL-Zero-General.
* OLMo 3 SFT/DPO/final Instruct/RL-Zero-General structured choice ablation.

### 1:45 - Result

The project did not find a robust increase in instrumental behavior.

The clearer result was a judgeability problem.

Thinking and RL-Zero-style outputs were more likely to become verbose, token-capped, malformed, or contaminated with reasoning traces.

### 2:25 - Why It Matters

If output-format failures are not tracked separately, safety evals can mistake judgeability artifacts for behavioral differences.

So the contribution is a reusable warning and workflow: score both behavioral risk and judgeability.

### 2:55 - Next Step

The next step is to publish the short report and optionally run a narrow paraphrase robustness check on the clean OLMo Instruct-stage models.

## Seven-Minute Version

### 0:00 - Motivation

Reasoning/RLVR-style training is increasingly common. One safety concern is whether it encourages instrumental behavior. But model outputs can also change format: longer reasoning traces, truncation, malformed answers, and verbose preambles.

This project asks whether there is a clean behavioral signal, or whether the evaluation is dominated by these format changes.

### 1:00 - Evaluation Design

The seed prompt set covers six categories:

* Goal persistence.
* Resource acquisition.
* Self-preservation.
* Deception.
* Evaluation awareness.
* Benign controls.

The scoring rubric separates safe answers, ambiguous answers, and clear instrumental endorsement.

### 2:00 - Tooling

The repo includes:

* `scripts/run_behavioral_eval.py` for Kaggle generation.
* `scripts/score_manual_template.py` for blind scoring sheets.
* `scripts/build_choice_eval.py` and `scripts/score_choice_eval.py` for structured choice evals.
* `scripts/gemini_judge_outputs.py` for risk and judgeability review.
* `scripts/build_paper_tables.py` for report tables.

### 3:00 - Main Results

Qwen3 Thinking had higher apparent risk than Instruct, but the result was dominated by truncation and exposed reasoning traces.

OLMo 3 RL-Zero-General did not score higher than OLMo 3 Instruct in the full free-form run.

In the structured choice stage ablation, final OLMo 3 Instruct did not score higher than SFT/DPO.

RL-Zero-General remained less judgeable even under strict structured choice prompting.

### 4:30 - Gemini Judgeability

Gemini agreed strongly with parsed structured choice labels and broadly with free-form manual labels.

It was most useful for identifying disagreement rows and judgeability failures, not as ground truth.

### 5:30 - Interpretation

The defensible finding is not "RLVR causes instrumental convergence."

The defensible finding is: for small open reasoning/RLVR-lineage models, judgeability can degrade enough to dominate safety eval interpretation.

### 6:30 - Ask

I want feedback on whether the rubric, judgeability tags, and limitations are clear enough for this to be useful as a short public report.
