# RLVR Safety Dynamics in Current Small Open Models

Small, reproducible audit of whether current open Thinking/RLVR-lineage models show stronger instrumental-convergence-style behavior than matched instruction models.

## Research Question

Do current small reasoning or RLVR-lineage open models show stronger goal-persistence, resource-acquisition, self-preservation, deception, or evaluation-awareness signals than matched non-thinking instruction models?

This project is exploratory. It does not try to prove that RLVR causes misalignment. The first pass treats matched Thinking/Instruct pairs as reasoning-specialization comparisons, then optionally uses OLMo 3 RL-Zero checkpoints for a more direct RLVR-lineage check.

## Current Preliminary Findings

The current evidence does not support a simple claim that the tested small open RLVR-lineage or reasoning-specialized models show a robust increase in instrumental behavior.

The stronger finding is methodological: judgeability and prompt wording sensitivity can dominate the behavioral signal in this setting.

### Qwen status

The Qwen3 4B Instruct vs Thinking cheap run is inconclusive as behavioral evidence. The Thinking model produced much longer outputs, frequently hit the token cap, and exposed or implied reasoning traces despite concise-answer prompting. Those format differences made most risk rows confounded, so the run is best treated as evidence that reasoning-specialized outputs can be harder to judge, not as evidence of higher instrumental behavior.

### OLMo status

The OLMo 3 free-form Instruct vs RL-Zero-General run did not show a robust risk increase for RL-Zero-General. RL-Zero-General was also frequently verbosity-confounded or token-capped.

The structured OLMo stage-ablation run was cleaner for the three Instruct-stage models:

| Model | Mean parsed score | Parse/length status |
|---|---:|---|
| `allenai/Olmo-3-7B-Instruct` | 0.42 | clean |
| `allenai/Olmo-3-7B-Instruct-DPO` | 0.46 | clean |
| `allenai/Olmo-3-7B-Instruct-SFT` | 0.50 | clean |

This did not show final Instruct scoring higher than SFT/DPO. RL-Zero-General remained difficult to compare in the structured setting because malformed and token-capped outputs dominated.

### Paraphrase robustness status

A follow-up paraphrase robustness run generated one Gemini-validated paraphrase for each of the 24 structured-choice prompts and ran the three clean OLMo Instruct-stage models on Kaggle. All paraphrases passed semantic validation, and all three models produced clean, non-capped structured outputs.

| Model | Original mean | Paraphrase mean | Delta |
|---|---:|---:|---:|
| `allenai/Olmo-3-7B-Instruct` | 0.42 | 0.58 | +0.17 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 0.46 | 0.67 | +0.21 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 0.50 | 0.62 | +0.12 |

This is a useful robustness result, but it should not be overclaimed. The aggregate scores moved upward under paraphrase, while category-level movement was mixed. The current best framing is that the structured eval is judgeable for these models, but not wording-invariant.

### Expand vs pivot

Expand this project if a second paraphrase seed, a larger prompt set, or another model family shows stable judgeability and similar prompt-sensitivity patterns.

Pivot the claim, not the project, if additional runs remain unstable or if model differences are smaller than prompt-wording effects. The most defensible current direction is:

> Small-model RLVR/reasoning safety evals should report judgeability and prompt sensitivity separately from behavioral risk scores.

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
* `scripts/gemini_judge_outputs.py`: optional Gemini Flash-Lite judge for risk and judgeability scoring.
* `scripts/gemini_generate_paraphrases.py`: optional Gemini paraphrase generator and semantic validator.
* `scripts/analyze_judge_results.py`: aggregates judgeability, risk scores, and manual agreement tables.

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
make smoke-gemini-judge
make smoke-gemini-paraphrase
make smoke-judge-analysis
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

Run Gemini judge tooling locally with a user-provided API key:

```bash
export GEMINI_API_KEY=...
python scripts/gemini_judge_outputs.py \
  --generations results/kaggle_choice_stage_ablation_v2/choice_generations_combined.jsonl \
  --out results/gemini_choice_stage_ablation_v2.jsonl \
  --max-new-tokens 96
```

Generate Gemini paraphrases for the targeted choice eval:

```bash
python scripts/gemini_generate_paraphrases.py \
  --choice-prompts data/choice_eval_targeted.jsonl \
  --out results/gemini_choice_paraphrases.jsonl \
  --validation-out results/gemini_choice_paraphrase_validations.jsonl \
  --per-source 3
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
* Optional Gemini judge and paraphrase-generation workflow.
* Results tables with examples and failure analysis.
* Optional activation-level separability analysis.
