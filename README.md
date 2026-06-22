# RLVR Safety Dynamics in Current Small Open Models

Small, reproducible audit of whether current open Thinking/RLVR-lineage models show stronger instrumental-convergence-style behavior than matched instruction models.

## Research Question

Do current small reasoning or RLVR-lineage open models show stronger goal-persistence, resource-acquisition, self-preservation, deception, or evaluation-awareness signals than matched non-thinking instruction models?

This project is exploratory. It does not try to prove that RLVR causes misalignment. The first pass treats matched Thinking/Instruct pairs as reasoning-specialization comparisons, then optionally uses OLMo 3 RL-Zero checkpoints for a more direct RLVR-lineage check.

## First Model Pair

Primary Kaggle-feasible pair:

| Role | Model |
|---|---|
| Baseline | `Qwen/Qwen3-4B-Instruct-2507` |
| Reasoning-specialized | `Qwen/Qwen3-4B-Thinking-2507` |

Second-stage pair if compute allows:

| Role | Model |
|---|---|
| Baseline | `allenai/Olmo-3-7B-Instruct` |
| Reasoning/RLVR-lineage | `allenai/Olmo-3-7B-Think` or `allenai/Olmo-3-7B-RL-Zero-Mix` |

## What This Repo Currently Contains

This initial scaffold is meant to pass a "is this project real and runnable?" gate before applying for compute funding.

* `data/prompts_seed.jsonl`: seed prompt set with categories, expected risk dimensions, and paraphrase-group metadata.
* `data/scoring_rubric.md`: manual scoring rubric for instrumental endorsement and safe redirection.
* `configs/models.json`: model pairs and inference settings.
* `scripts/validate_prompts.py`: schema and category-count checks for the prompt dataset.
* `scripts/run_behavioral_eval.py`: Hugging Face generation runner for a first Kaggle pass.
* `scripts/score_manual_template.py`: creates a CSV template for manual scoring.
* `docs/unit2_project_log.md`: BlueDot Unit 2 project summary, assumptions, cruxes, and first-cheap-test plan.

## MVP Plan

1. Validate the seed prompt set.
2. Run Qwen3 4B Instruct and Thinking on 48-60 prompts with fixed generation settings.
3. Manually score outputs before using any LLM judge.
4. Add paraphrases for the highest-signal prompts only.
5. Collect final-token or answer-token hidden states for a small subset if the behavioral signal is nontrivial.

## BlueDot Unit 2 Check

The Unit 2 prompts are answered in `docs/unit2_project_log.md`:

* 3-5 sentence project summary
* Long list of assumptions
* High-stakes/high-uncertainty cruxes
* First cheap test with prediction and change-course criteria

## Quickstart

Run local non-GPU gates:

```bash
make validate
make compile
make smoke-score
```

Validate the prompt file:

```bash
python scripts/validate_prompts.py --prompts data/prompts_seed.jsonl
```

Create a manual scoring sheet from generated outputs:

```bash
python scripts/score_manual_template.py \
  --generations results/generations_qwen3_4b_sample.jsonl \
  --out results/manual_scores_template.csv
```

Run generation on Kaggle or another GPU machine:

```bash
python scripts/run_behavioral_eval.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --prompts data/prompts_seed.jsonl \
  --out results/generations_qwen3_4b_instruct.jsonl \
  --load-in-4bit
```

## Limitations

* Instrumental-convergence-style prompts are subjective and can be prompt-sensitive.
* Thinking/Instruct comparisons do not isolate RLVR causally.
* Thinking models may produce longer outputs, so response length must be reported as a control.
* Activation analysis is exploratory and should not be overinterpreted.

## Intended Deliverables

* Public dataset and prompt schema.
* Reproducible generation scripts.
* Transparent manual scoring rubric.
* Results tables with examples and failure analysis.
* Optional activation-level separability analysis.
