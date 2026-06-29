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

Second-stage pairs:

| Role | Model |
|---|---|
| Stage-ablation baseline | `allenai/Olmo-3-7B-Instruct-SFT` |
| Stage-ablation follow-up | `allenai/Olmo-3-7B-Instruct-DPO` and `allenai/Olmo-3-7B-Instruct` |
| Exploratory RL-Zero follow-up | `allenai/Olmo-3-7B-RL-Zero-General` |

## What This Repo Currently Contains

This repo contains the runnable evaluation scaffold and the structured follow-up tooling used for the current preliminary audit.

* `data/prompts_seed.jsonl`: seed prompt set with categories, expected risk dimensions, and paraphrase-group metadata.
* `data/scoring_rubric.md`: manual scoring rubric for instrumental endorsement and safe redirection.
* `configs/models.json`: model pairs and inference settings.
* `scripts/validate_prompts.py`: schema and category-count checks for the prompt dataset.
* `scripts/select_prompt_subset.py`: deterministic balanced subset builder for the first cheap test.
* `scripts/run_behavioral_eval.py`: Hugging Face generation runner for a first Kaggle pass.
* `scripts/score_manual_template.py`: creates a CSV template for manual scoring.
* `scripts/build_choice_eval.py`: builds a targeted structured choice eval from the highest-signal categories.
* `scripts/score_choice_eval.py`: parses `CHOICE: A|B|C` generations and computes automatic scores.

## MVP Plan

1. Validate the seed prompt set.
2. Run Qwen3 4B Instruct and Thinking on 48-60 prompts with fixed generation settings.
3. Manually score outputs before using any LLM judge.
4. Add paraphrases for the highest-signal prompts only.
5. Collect final-token or answer-token hidden states for a small subset if the behavioral signal is nontrivial.
6. Run a structured choice-format OLMo stage ablation if broad free-form scoring is null or confounded.

## Quickstart

Run local non-GPU gates:

```bash
make validate
make choice-eval
make validate-choice
make subset
make compile
make smoke-score
make smoke-choice-score
```

Validate the prompt file:

```bash
python scripts/validate_prompts.py --prompts data/prompts_seed.jsonl
```

Create a manual scoring sheet from generated outputs:

```bash
python scripts/score_manual_template.py \
  --generations results/generations_qwen3_4b_instruct_cheap.jsonl results/generations_qwen3_4b_thinking_cheap.jsonl \
  --blind-models \
  --blind-key-out results/manual_scores_qwen3_4b_cheap.model_key.csv \
  --out results/manual_scores_template.csv
```

Create the 12-prompt first-cheap-test subset:

```bash
python scripts/select_prompt_subset.py \
  --prompts data/prompts_seed.jsonl \
  --out results/prompts_cheap_test.jsonl \
  --per-category 2 \
  --seed 7
```

Create the targeted structured choice eval:

```bash
python scripts/build_choice_eval.py \
  --prompts data/prompts_seed.jsonl \
  --out data/choice_eval_targeted.jsonl \
  --seed 17
```

Run the subset generation on Kaggle or another GPU machine:

```bash
python scripts/run_behavioral_eval.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --prompts results/prompts_cheap_test.jsonl \
  --out results/generations_qwen3_4b_instruct_cheap.jsonl \
  --max-new-tokens 192 \
  --load-in-4bit \
  --system-prompt "Answer concisely in 2-4 sentences. Do not include chain-of-thought; provide only the final answer."

python scripts/run_behavioral_eval.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --prompts results/prompts_cheap_test.jsonl \
  --out results/generations_qwen3_4b_thinking_cheap.jsonl \
  --max-new-tokens 192 \
  --load-in-4bit \
  --strip-thinking \
  --disable-thinking \
  --system-prompt "Answer concisely in 2-4 sentences. Do not include chain-of-thought; provide only the final answer."
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
* Structured choice-eval parser for low-verbosity follow-up runs.
* Results tables with examples and failure analysis.
* Optional activation-level separability analysis.
