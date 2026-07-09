# RLVR Safety Evals Are Often Limited by Judgeability Before Behavioral Signal

## Summary

This project tested whether current small open reasoning/RLVR-lineage models show stronger instrumental-convergence-style behavior than matched instruction models.

The current evidence does not support a robust positive behavioral-risk claim.

The stronger finding is methodological: Thinking/RL-Zero-style outputs often become harder to judge before they become clearly more instrumentally risky. They are more likely to be verbose, token-capped, malformed, or contaminated with reasoning traces. Safety evals that do not measure judgeability separately can therefore confuse output-format artifacts with behavioral differences.

## Research Question

Do small open reasoning/RLVR-lineage models show more goal persistence, resource acquisition, self-preservation, deception, or evaluation awareness than matched instruction models?

Secondary question:

Are apparent differences actually behavioral, or are they dominated by output format and judgeability?

## Methods

The project uses three evaluation layers.

1. Free-form prompts over goal persistence, resource acquisition, self-preservation, deception, evaluation awareness, and benign controls.
2. Structured choice prompts over the highest-signal categories: resource acquisition, self-preservation, and evaluation awareness.
3. Gemini Flash-Lite judgeability review using the official Gemini SDK, with manual labels retained as final authority.

Generation was run on Kaggle GPU. Scoring, analysis, Gemini judge passes, and table generation were run locally.

## Models

Qwen cheap test:

* `Qwen/Qwen3-4B-Instruct-2507`
* `Qwen/Qwen3-4B-Thinking-2507`

OLMo free-form run:

* `allenai/Olmo-3-7B-Instruct`
* `allenai/Olmo-3-7B-RL-Zero-General`

OLMo structured stage ablation:

* `allenai/Olmo-3-7B-Instruct-SFT`
* `allenai/Olmo-3-7B-Instruct-DPO`
* `allenai/Olmo-3-7B-Instruct`
* `allenai/Olmo-3-7B-RL-Zero-General`

## Result 1: Qwen Thinking Was Confounded

The first Qwen cheap test appeared to show higher risk for the Thinking model, but the comparison was dominated by output-format problems.

Risk-prompt aggregate:

| Model | Mean risk | Confounded risk rows |
|---|---:|---:|
| Qwen3 4B Instruct | 0.20 | 0/10 |
| Qwen3 4B Thinking | 0.50 | 9/10 |

The Thinking model hit the 192-token cap on 11/12 outputs and exposed or implied reasoning traces despite concise-answer prompting.

Conclusion: the Qwen run is useful as a confound discovery result, not clean behavioral evidence.

## Result 2: OLMo RL-Zero-General Did Not Show A Free-Form Risk Increase

The full OLMo free-form run did not support the original risk-increase hypothesis.

| Model | Mean manual risk | Score counts | Confounded risk rows |
|---|---:|---|---:|
| OLMo 3 Instruct | 0.38 | 27x0, 11x1, 2x2 | 0/40 |
| OLMo 3 RL-Zero-General | 0.35 | 26x0, 14x1, 0x2 | 16/40 |

RL-Zero-General did not score higher than Instruct, and many of its rows were token-capped or verbosity-confounded.

Conclusion: this run does not show a robust behavioral-risk increase.

## Result 3: OLMo Stage Ablation Did Not Show Final Instruct > SFT/DPO

The structured choice eval made the OLMo Instruct-stage models easier to compare.

| Model | Mean parsed risk | Counts | Malformed | Token-cap rows |
|---|---:|---|---:|---:|
| OLMo 3 Instruct | 0.42 | 14x0, 10x1, 0x2 | 0/24 | 0/24 |
| OLMo 3 Instruct-DPO | 0.46 | 13x0, 11x1, 0x2 | 0/24 | 0/24 |
| OLMo 3 Instruct-SFT | 0.50 | 13x0, 10x1, 1x2 | 0/24 | 0/24 |
| OLMo 3 RL-Zero-General | 0.44 | 9x0, 7x1, 0x2 | 8/24 | 24/24 |

Final Instruct did not score higher than SFT or DPO.

RL-Zero-General was not directly comparable because every row hit the token cap and one-third of rows were malformed.

Conclusion: the stage ablation does not support a broad claim that final RLVR-style post-training increased instrumental choices.

## Result 4: Gemini Was Useful For Judgeability Triage

Gemini Flash-Lite was used as a second-pass judge for risk and judgeability. Manual adjudication remains final.

| Source | Model | Mean Gemini risk | Mean judgeability problem | Needs review | Failure modes |
|---|---|---:|---:|---:|---|
| Free-form | OLMo 3 Instruct | 0.25 | 0.00 | 0/48 | clean:48 |
| Free-form | OLMo 3 RL-Zero-General | 0.21 | 0.56 | 9/48 | clean:34, token_capped:14 |
| Structured choice | OLMo 3 Instruct | 0.42 | 0.00 | 0/24 | clean:24 |
| Structured choice | OLMo 3 Instruct-DPO | 0.46 | 0.00 | 0/24 | clean:24 |
| Structured choice | OLMo 3 Instruct-SFT | 0.50 | 0.00 | 0/24 | clean:24 |
| Structured choice | OLMo 3 RL-Zero-General | 0.75 | 1.38 | 4/24 | malformed:5, other:4, token_capped:5, verbose_preamble:10 |

Agreement:

| Run | Comparable rows | Exact agreement | Binary nonzero agreement |
|---|---:|---:|---:|
| Free-form OLMo | 96 | 79/96 (82.29%) | 85/96 (88.54%) |
| Structured choice | 88 | 88/88 (100.00%) | 88/88 (100.00%) |

Conclusion: Gemini is useful for finding disagreement rows and judgeability failures, but it should not replace manual adjudication.

## Result 5: Paraphrased Choice Prompts Changed The Clean OLMo Scores

A narrow robustness follow-up was run on 2026-07-06.

Gemini Flash-Lite generated one paraphrase for each structured-choice prompt and then validated whether the safe=0, ambiguous=1, risky=2 option semantics were preserved. The generated paraphrase pack passed validation: 24/24 rows preserved the intended labels and passed the local structural validator.

The paraphrased prompts were then run on Kaggle for the three OLMo Instruct-stage models that were clean in the original structured-choice ablation. RL-Zero-General was excluded because the prior structured run was dominated by malformed and token-capped rows.

| Model | Original mean | Paraphrase mean | Delta | Paraphrase malformed | Paraphrase token-cap rows |
|---|---:|---:|---:|---:|---:|
| OLMo 3 Instruct | 0.42 | 0.58 | +0.17 | 0/24 | 0/24 |
| OLMo 3 Instruct-DPO | 0.46 | 0.67 | +0.21 | 0/24 | 0/24 |
| OLMo 3 Instruct-SFT | 0.50 | 0.62 | +0.12 | 0/24 | 0/24 |

The aggregate movement was upward for all three clean models, but the category movement was mixed: self-preservation rose sharply, evaluation-awareness rose mildly, and resource-acquisition fell.

Conclusion: the structured-choice protocol can stay judgeable under paraphrase, but measured instrumental-choice rates are prompt-sensitive. This is useful evidence for the report because it turns the follow-up from a pure null into a concrete robustness/sensitivity finding.

## Interpretation

The defensible claim is:

> In these small open model runs, the strongest repeated effect is judgeability degradation, not a robust increase in instrumental behavior.

The paraphrase run adds:

> When judgeability is clean, the measured score can still move substantially under validated paraphrases, so prompt wording sensitivity should be reported alongside model comparisons.

This matters because safety evals can become misleading if:

* token-capped rows are treated as complete answers,
* malformed structured-choice outputs are parsed as comparable responses,
* exposed reasoning traces are scored the same way as final answers,
* judgeability failures are folded into behavioral risk scores.

## Limitations

The prompt set is small and hand-written. The scoring rubric is transparent but subjective. The model set is narrow. Thinking/Instruct and RL-Zero/Instruct comparisons do not isolate causal effects of RLVR. Small models may not express the same behaviors as frontier reasoning models. Gemini is a useful reviewer, not ground truth.

## Reproducibility

Local non-GPU checks:

```bash
make validate
make choice-eval
make validate-choice
make compile
make smoke-score
make smoke-choice-score
make smoke-gemini-judge
make smoke-gemini-paraphrase
make smoke-judge-analysis
make paper-tables
```

Paper tables:

```bash
make paper-tables
```

Gemini judge reruns require `GEMINI_API_KEY` in the local environment.

GPU-heavy model generation should run on Kaggle, not locally.

## Next Work

No grant-required work:

1. Polish this report.
2. Add a qualitative examples appendix.
3. Update the README with current results.
4. Share with BlueDot/community for feedback.

Optional compute/API work:

1. Inspect representative changed rows from the paraphrase robustness run.
2. Add the paraphrase result as a robustness appendix.
3. Optionally run a second paraphrase seed only if the write-up needs stronger wording-sensitivity evidence.

Deferred work:

* Broad Qwen reruns.
* Full RL-Zero-General reruns.
* Larger model sweeps.
* Activation analysis.
