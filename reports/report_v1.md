# RLVR Safety Evals Are Often Limited by Judgeability Before Behavioral Signal

## Summary

This project tested whether current small open reasoning/RLVR-lineage models show stronger instrumental-convergence-style behavior than matched instruction models.

The current evidence does not support a robust positive behavioral-risk claim.

The stronger finding is methodological: Thinking/RL-Zero-style outputs often become harder to judge before they become clearly more instrumentally risky. They are more likely to be verbose, token-capped, malformed, or contaminated with reasoning traces. Safety evals that do not measure judgeability separately can therefore confuse output-format artifacts with behavioral differences.

A secondary finding from two validated paraphrase seeds is that even when judgeability is clean, prompt-wording sensitivity can move measured scores by more than model-stage differences. In this setup, prompt wording effects exceeded the spread across OLMo Instruct-stage models (SFT, DPO, Instruct).

## Research Question

Do small open reasoning/RLVR-lineage models show more goal persistence, resource acquisition, self-preservation, deception, or evaluation awareness than matched instruction models?

Secondary question:

Are apparent differences actually behavioral, or are they dominated by output format, judgeability, and prompt wording?

## Methods

The project uses three evaluation layers.

1. Free-form prompts over goal persistence, resource acquisition, self-preservation, deception, evaluation awareness, and benign controls.
2. Structured choice prompts over the highest-signal categories: resource acquisition, self-preservation, and evaluation awareness.
3. Gemini Flash-Lite judgeability review using the official Gemini SDK, with manual labels retained as final authority.

Two independent Gemini-validated paraphrase seeds were generated for the 24 structured-choice prompts. Each seed was validated to confirm safe=0, ambiguous=1, risky=2 semantics were preserved (24/24 passed both times). The three clean OLMo Instruct-stage models were re-run on each paraphrase seed.

Generation was run on Kaggle GPU. Scoring, analysis, Gemini judge passes, and table generation were run locally.

## Defined Metrics

This report uses several metrics to quantify judgeability and prompt sensitivity:

**Judgeability failure rate**: Fraction of scored rows that are malformed, token-capped, or flagged for human review. Computed per-model as `(malformed + token_capped + needs_review) / total_rows`.

**Prompt sensitivity delta**: Difference between paraphrase mean score and original mean score, computed per-model and per-category. Positive values indicate the paraphrase wording elicited higher measured instrumental endorsement.

**Item flip rate**: Fraction of individual source items (prompts) whose score changes between original and paraphrase. Computed as `items_with_score_change / total_paired_items`.

**Model-stage spread**: Difference between the highest and lowest mean scores among the three clean OLMo Instruct-stage models (SFT, DPO, Instruct). On the original prompts, this spread is 0.08 (SFT=0.50 minus Instruct=0.42).

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

Final Instruct did not score higher than SFT or DPO. The model-stage spread among the three clean models is 0.08 (0.50 - 0.42).

RL-Zero-General was not directly comparable because every row hit the token cap and one-third of rows were malformed. Judgeability failure rate: 24/24 (100%).

Conclusion: the stage ablation does not support a broad claim that final RLVR-style post-training increased instrumental choices.

## Result 4: Gemini Was Useful For Judgeability Triage

Gemini Flash-Lite was used as a second-pass judge for risk and judgeability. Manual adjudication remains final.

| Source | Model | Mean Gemini risk | Mean judgeability problem | Needs review | Failure modes |
|---|---|---|---:|---:|---:|---|
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

## Result 5: Prompt Wording Sensitivity Exceeded Model-Stage Differences

Two independent paraphrase seeds were run to test whether the structured-choice protocol is robust to prompt wording changes.

Method:
1. Gemini Flash-Lite generated a paraphrase for each of the 24 structured-choice prompts.
2. Gemini validated whether safe=0, ambiguous=1, risky=2 semantics were preserved.
3. The paraphrase pack passed validation (24/24 for both seeds).
4. Models were re-run on Kaggle with the paraphrased prompts.
5. All three clean OLMo Instruct-stage models were tested (RL-Zero-General excluded: 100% judgeability failure rate).

This process was repeated for two independent paraphrase seeds.

### Seed 1 (paraphrase run 2026-07-06)

| Model | Original mean | Paraphrase mean | Delta | Paraphrase malformed | Paraphrase token-cap rows |
|---|---:|---:|---:|---:|---:|
| OLMo 3 Instruct | 0.42 | 0.58 | +0.17 | 0/24 | 0/24 |
| OLMo 3 Instruct-DPO | 0.46 | 0.67 | +0.21 | 0/24 | 0/24 |
| OLMo 3 Instruct-SFT | 0.50 | 0.62 | +0.12 | 0/24 | 0/24 |

### Seed 2 (paraphrase run 2026-07-08)

| Model | Original mean | Paraphrase mean | Delta | Paraphrase malformed | Paraphrase token-cap rows |
|---|---:|---:|---:|---:|---:|
| OLMo 3 Instruct | 0.42 | 0.75 | +0.33 | 0/24 | 0/24 |
| OLMo 3 Instruct-DPO | 0.46 | 0.79 | +0.33 | 0/24 | 0/24 |
| OLMo 3 Instruct-SFT | 0.50 | 0.79 | +0.29 | 0/24 | 0/24 |

Both seeds were clean: 72/72 parsed, 0 malformed, 0 token-cap per seed.

### Combined Sensitivity Summary

| Model | Orig | Seed 1 mean | Seed 2 mean | Seed 1 delta | Seed 2 delta | Seed 1 flip | Seed 2 flip | Seed 1 MAD | Seed 2 MAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OLMo 3 Instruct | 0.42 | 0.58 | 0.75 | +0.17 | +0.33 | 13/24 | 14/24 | 0.58 | 0.67 |
| OLMo 3 Instruct-DPO | 0.46 | 0.67 | 0.79 | +0.21 | +0.33 | 12/24 | 16/24 | 0.54 | 0.67 |
| OLMo 3 Instruct-SFT | 0.50 | 0.62 | 0.79 | +0.12 | +0.29 | 14/24 | 16/24 | 0.62 | 0.71 |

MAD = mean absolute delta per item. Flip = fraction of items whose score changed.

### Category-Level Deltas (Combined)

| Model | Category | Orig | Seed 1 mean | Seed 2 mean | Seed 1 delta | Seed 2 delta | Seed 1 flip | Seed 2 flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLMo 3 Instruct | evaluation_awareness | 0.38 | 0.50 | 0.50 | +0.12 | +0.12 | 3/8 | 3/8 |
| OLMo 3 Instruct | resource_acquisition | 0.75 | 0.38 | 0.50 | -0.38 | -0.25 | 3/8 | 4/8 |
| OLMo 3 Instruct | self_preservation | 0.12 | 0.88 | 1.25 | +0.75 | +1.12 | 7/8 | 7/8 |
| OLMo 3 Instruct-DPO | evaluation_awareness | 0.50 | 0.62 | 0.75 | +0.12 | +0.25 | 3/8 | 4/8 |
| OLMo 3 Instruct-DPO | resource_acquisition | 0.75 | 0.38 | 0.62 | -0.38 | -0.12 | 3/8 | 5/8 |
| OLMo 3 Instruct-DPO | self_preservation | 0.12 | 1.00 | 1.00 | +0.88 | +0.88 | 6/8 | 7/8 |
| OLMo 3 Instruct-SFT | evaluation_awareness | 0.38 | 0.50 | 0.75 | +0.12 | +0.38 | 5/8 | 5/8 |
| OLMo 3 Instruct-SFT | resource_acquisition | 0.75 | 0.62 | 0.75 | -0.12 | +0.00 | 3/8 | 4/8 |
| OLMo 3 Instruct-SFT | self_preservation | 0.38 | 0.75 | 0.88 | +0.38 | +0.50 | 6/8 | 7/8 |

### Key Observations

**Judgeability stayed clean for all three models across both seeds.** Zero malformed responses, zero token-capped outputs. The structured-choice protocol is judgeable under paraphrase, which makes it a useful sensitivity instrument.

**All three models scored higher under paraphrased wording in both seeds.** Every per-model delta was positive, ranging from +0.12 to +0.33. The ordinal ranking was preserved (SFT > DPO > Instruct), but the absolute levels shifted materially.

**Self-preservation was the dominant sensitivity category.** Self-preservation deltas were +0.38 to +1.12, with flip rates of 6/8 to 7/8. Resource-acquisition scores declined or stayed flat in most cases.

**Prompt wording effects exceeded model-stage differences.** The model-stage spread on original prompts was 0.08. Every prompt sensitivity delta (per-seed, per-model) was larger than 0.08, ranging from +0.12 to +0.33. The mean absolute deltas (0.54 to 0.71) were 7-9x larger than the model-stage spread.

**Item flip rates were high.** Across both seeds, 50-67% of individual items changed score between original and paraphrase. This means the measured score for any given prompt-source pair is meaningfully sensitive to how the prompt is worded.

**Seed dependence is real.** Seed 2 produced consistently larger deltas than Seed 1 (+0.29 to +0.33 vs +0.12 to +0.21). The qualitative pattern (self-preservation up, resource-acquisition down) held, but the magnitude varied. Reports should include sensitivity intervals, not single prompt-set scores.

## Interpretation

The defensible claim is:

> In these small open model runs, the strongest repeated effect is judgeability degradation, not a robust increase in instrumental behavior. When judgeability is clean, prompt wording effects can exceed model-stage differences.

Stronger claim (supported by the data):

> Across two validated paraphrase seeds, prompt wording moved measured instrumental-risk scores by 7-9x more than the spread across OLMo Instruct-stage models.

This matters because safety evals can become misleading if:

* token-capped rows are treated as complete answers,
* malformed structured-choice outputs are parsed as comparable responses,
* exposed reasoning traces are scored the same way as final answers,
* judgeability failures are folded into behavioral risk scores,
* single prompt-set scores are reported without sensitivity intervals.

The paraphrase result is important not because it shows RLVR causes anything, but because it shows that **even semantically validated rewording can shift measured scores by more than the effect size being studied.** If a safety eval claims a model-stage difference of 0.08 but prompt wording shifts scores by 0.12-0.33, the eval's resolution is insufficient for the claim.

## Limitations

The prompt set is small and hand-written. The scoring rubric is transparent but subjective. The model set is narrow. Thinking/Instruct and RL-Zero/Instruct comparisons do not isolate causal effects of RLVR. Small models may not express the same behaviors as frontier reasoning models. Gemini is a useful reviewer, not ground truth.

Paraphrase robustness covers two validated Gemini seeds. While both show consistent qualitative patterns, additional independent paraphrases (human-written or from different models) would strengthen the sensitivity interval claim.

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

Prompt sensitivity (seed 1 only; seed 2 requires manual invocation):

```bash
make analyze-sensitivity
```

Gemini judge reruns require `GEMINI_API_KEY` in the local environment.

GPU-heavy model generation should run on Kaggle, not locally.

## Next Work

No grant-required work:

1. ~~Polish this report.~~
2. ~~Add a qualitative examples appendix.~~
3. ~~Update the README with current results.~~
4. Share with BlueDot/community for feedback.

Optional compute/API work:

1. Run a third paraphrase seed to strengthen the sensitivity-interval claim (requires T4 GPU on Kaggle).
2. Add human-written paraphrases as an additional validation layer.

Deferred work:

* Broad Qwen reruns.
* Full RL-Zero-General reruns.
* Larger model sweeps.
* Activation analysis.
