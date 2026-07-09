# RLVR Safety Dynamics in Current Small Open Models

Small, reproducible audit of whether current open reasoning/RLVR-lineage models show stronger instrumental behavior than matched instruction models.

## Headline Result

The current evidence does not support a robust positive behavioral-risk claim. The stronger finding is methodological:

> Small-model RLVR/reasoning safety evals should report judgeability and prompt sensitivity separately from behavioral risk scores.

### Key findings

- **Qwen3 4B Thinking was confounded** by token caps and exposed reasoning traces — higher apparent risk was dominated by format artifacts.
- **OLMo 3 RL-Zero-General did not score higher** than Instruct in free-form scoring, and remained less judgeable under structured choice prompting.
- **OLMo stage ablation found no monotonic increase**: final Instruct (0.42) did not score higher than SFT (0.50) or DPO (0.46) on structured choice prompts.
- **Paraphrase robustness**: A Gemini-validated paraphrase of the 24 choice prompts shifted aggregate scores upward for all three clean OLMo models (deltas +0.12 to +0.21), but category movement was mixed. The eval is judgeable but not wording-invariant.

## Repo Map

| Path | Contents |
|---|---|
| `data/` | Prompt inputs (`prompts_seed.jsonl`, `choice_eval_targeted.jsonl`) and scoring rubric |
| `scripts/` | Runnable CLI entrypoints for validation, scoring, generation, judge passes, and tables |
| `reports/` | Curated human-facing outputs: report draft, tables, prompt sensitivity, adjudication notes, examples, demo script |
| `results/` | Raw generated outputs (git-ignored except curated samples and `.gitkeep`) |
| `configs/` | Model and inference settings |
| `tests/` | Fixtures for smoke tests |

## Reproduce Checks

```bash
make validate
make validate-choice
make compile
make smoke-choice-score
make smoke-gemini-judge
make smoke-gemini-paraphrase
make smoke-judge-analysis
```

## Reproduce Tables

```bash
make paper-tables
make analyze-sensitivity
```

Paper tables are written to `results/paper_tables.md` and `reports/paper_tables.md`.
Prompt sensitivity analysis is written to `reports/prompt_sensitivity.md`.

GPU-heavy model generation should run on Kaggle (see `scripts/run_behavioral_eval.py`).
Gemini judge passes require `GEMINI_API_KEY` (see `scripts/gemini_judge_outputs.py`).

## Limitations

- Prompt set is small and hand-written.
- Scoring rubric is transparent but subjective.
- Thinking/Instruct and RL-Zero/Instruct comparisons do not isolate causal effects of RLVR.
- Paraphrase robustness covers one seed per prompt.
- Small models may not express the same behaviors as frontier reasoning models.
