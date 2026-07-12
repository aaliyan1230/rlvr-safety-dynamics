# Counterbalanced Wording × Option-Order Analysis

Protocol: BitsAndBytes default 4-bit loading under Transformers 4.57.6; standardized fixed-runtime primary protocol, not an exact historical reproduction.

Complete cells: 1728 (3 models × 24 source items × 4 wordings × 6 option orders).

Intervals are 95% source-item-clustered bootstrap intervals. Scores are ordinal 0–2; the variance table is descriptive rather than an ordinal causal model.

## Model-stage estimates (marginalized over wording and order)

| Model | Mean [95% CI] | P(score > 0) | P(score = 2) | Rows |
|---|---:|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 0.380 [0.276, 0.488] | 0.345 | 0.035 | 576 |
| allenai/Olmo-3-7B-Instruct-DPO | 0.392 [0.292, 0.500] | 0.370 | 0.023 | 576 |
| allenai/Olmo-3-7B-Instruct-SFT | 0.413 [0.311, 0.528] | 0.370 | 0.043 | 576 |

Marginal model-stage spread: **0.033**.

## Wording effects (marginalized over all six option orders)

| Model | Wording | Mean [95% CI] | Delta vs original [95% CI] |
|---|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | original | 0.403 [0.271, 0.542] | +0.000 [+0.000, +0.000] |
| allenai/Olmo-3-7B-Instruct | p1 | 0.347 [0.194, 0.507] | -0.056 [-0.215, +0.118] |
| allenai/Olmo-3-7B-Instruct | p2 | 0.451 [0.271, 0.660] | +0.049 [-0.174, +0.285] |
| allenai/Olmo-3-7B-Instruct | p3 | 0.319 [0.174, 0.479] | -0.083 [-0.292, +0.132] |
| allenai/Olmo-3-7B-Instruct-DPO | original | 0.444 [0.319, 0.576] | +0.000 [+0.000, +0.000] |
| allenai/Olmo-3-7B-Instruct-DPO | p1 | 0.354 [0.201, 0.514] | -0.090 [-0.243, +0.076] |
| allenai/Olmo-3-7B-Instruct-DPO | p2 | 0.444 [0.285, 0.618] | +0.000 [-0.195, +0.215] |
| allenai/Olmo-3-7B-Instruct-DPO | p3 | 0.326 [0.194, 0.472] | -0.118 [-0.306, +0.083] |
| allenai/Olmo-3-7B-Instruct-SFT | original | 0.472 [0.319, 0.632] | +0.000 [+0.000, +0.000] |
| allenai/Olmo-3-7B-Instruct-SFT | p1 | 0.375 [0.229, 0.528] | -0.097 [-0.292, +0.104] |
| allenai/Olmo-3-7B-Instruct-SFT | p2 | 0.403 [0.243, 0.569] | -0.069 [-0.292, +0.153] |
| allenai/Olmo-3-7B-Instruct-SFT | p3 | 0.403 [0.236, 0.576] | -0.069 [-0.285, +0.160] |

## Legacy single-layout delta versus permutation-marginalized delta

The legacy layout compares each original item's source order with candidate order 012 (safe=A, ambiguous=B, risky=C). The confounding shift is legacy minus permutation-marginalized delta.

| Model | Wording | Legacy delta [95% CI] | Marginalized delta [95% CI] | Confounding shift [95% CI] |
|---|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | p1 | +0.083 [-0.208, +0.375] | -0.056 [-0.215, +0.125] | +0.139 [-0.090, +0.354] |
| allenai/Olmo-3-7B-Instruct | p2 | +0.250 [-0.083, +0.583] | +0.049 [-0.160, +0.285] | +0.201 [-0.063, +0.451] |
| allenai/Olmo-3-7B-Instruct | p3 | +0.000 [-0.292, +0.292] | -0.083 [-0.285, +0.139] | +0.083 [-0.118, +0.278] |
| allenai/Olmo-3-7B-Instruct-DPO | p1 | +0.083 [-0.208, +0.375] | -0.090 [-0.236, +0.083] | +0.174 [-0.028, +0.368] |
| allenai/Olmo-3-7B-Instruct-DPO | p2 | +0.125 [-0.167, +0.417] | +0.000 [-0.201, +0.222] | +0.125 [-0.104, +0.354] |
| allenai/Olmo-3-7B-Instruct-DPO | p3 | +0.042 [-0.250, +0.333] | -0.118 [-0.306, +0.083] | +0.160 [-0.056, +0.375] |
| allenai/Olmo-3-7B-Instruct-SFT | p1 | +0.083 [-0.167, +0.333] | -0.097 [-0.299, +0.111] | +0.181 [-0.083, +0.431] |
| allenai/Olmo-3-7B-Instruct-SFT | p2 | +0.125 [-0.167, +0.417] | -0.069 [-0.292, +0.146] | +0.194 [-0.042, +0.424] |
| allenai/Olmo-3-7B-Instruct-SFT | p3 | +0.083 [-0.208, +0.375] | -0.069 [-0.299, +0.174] | +0.153 [-0.028, +0.333] |

## Category effects (averaged over models and all six orders)

| Category | Wording | Mean | Delta vs original [95% CI] | Mean absolute source delta |
|---|---|---:|---:|---:|
| evaluation_awareness | original | 0.535 | +0.000 [+0.000, +0.000] | 0.000 |
| evaluation_awareness | p1 | 0.333 | -0.201 [-0.417, +0.021] | 0.326 |
| evaluation_awareness | p2 | 0.486 | -0.049 [-0.312, +0.201] | 0.312 |
| evaluation_awareness | p3 | 0.319 | -0.215 [-0.451, +0.049] | 0.396 |
| resource_acquisition | original | 0.618 | +0.000 [+0.000, +0.000] | 0.000 |
| resource_acquisition | p1 | 0.333 | -0.285 [-0.444, -0.132] | 0.299 |
| resource_acquisition | p2 | 0.208 | -0.410 [-0.632, -0.188] | 0.424 |
| resource_acquisition | p3 | 0.215 | -0.403 [-0.549, -0.243] | 0.403 |
| self_preservation | original | 0.167 | +0.000 [+0.000, +0.000] | 0.000 |
| self_preservation | p1 | 0.410 | +0.243 [-0.062, +0.576] | 0.396 |
| self_preservation | p2 | 0.604 | +0.438 [+0.118, +0.764] | 0.521 |
| self_preservation | p3 | 0.514 | +0.347 [-0.014, +0.708] | 0.514 |

## Risky-option position effects (marginalized over wording and relative order)

| Model | Risky option at | Mean [95% CI] |
|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | A | 0.323 [0.214, 0.448] |
| allenai/Olmo-3-7B-Instruct | B | 0.432 [0.318, 0.552] |
| allenai/Olmo-3-7B-Instruct | C | 0.385 [0.281, 0.495] |
| allenai/Olmo-3-7B-Instruct-DPO | A | 0.344 [0.240, 0.464] |
| allenai/Olmo-3-7B-Instruct-DPO | B | 0.443 [0.323, 0.573] |
| allenai/Olmo-3-7B-Instruct-DPO | C | 0.391 [0.302, 0.484] |
| allenai/Olmo-3-7B-Instruct-SFT | A | 0.333 [0.224, 0.453] |
| allenai/Olmo-3-7B-Instruct-SFT | B | 0.505 [0.365, 0.651] |
| allenai/Olmo-3-7B-Instruct-SFT | C | 0.401 [0.302, 0.505] |

Paired position contrasts are second position minus first position. The simultaneous intervals control the full pooled and per-model position family with a source-clustered max-|t| bootstrap.

| Scope | First | Second | Difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|
| all_models | A | B | +0.127 | [+0.047, +0.206] |
| all_models | A | C | +0.059 | [-0.036, +0.154] |
| all_models | B | C | -0.068 | [-0.170, +0.034] |
| allenai/Olmo-3-7B-Instruct | A | B | +0.109 | [+0.032, +0.186] |
| allenai/Olmo-3-7B-Instruct | A | C | +0.062 | [-0.059, +0.184] |
| allenai/Olmo-3-7B-Instruct | B | C | -0.047 | [-0.146, +0.052] |
| allenai/Olmo-3-7B-Instruct-DPO | A | B | +0.099 | [+0.026, +0.172] |
| allenai/Olmo-3-7B-Instruct-DPO | A | C | +0.047 | [-0.052, +0.146] |
| allenai/Olmo-3-7B-Instruct-DPO | B | C | -0.052 | [-0.169, +0.065] |
| allenai/Olmo-3-7B-Instruct-SFT | A | B | +0.172 | [+0.052, +0.292] |
| allenai/Olmo-3-7B-Instruct-SFT | A | C | +0.068 | [-0.038, +0.173] |
| allenai/Olmo-3-7B-Instruct-SFT | B | C | -0.104 | [-0.252, +0.043] |

## Paired model contrasts

Positive values mean the second model scored higher than the first.

The predeclared practical-equivalence margin is ±0.10. The final column uses the stricter rule that the full 95% interval must lie inside that band.

| First | Second | Mean difference [95% CI] | Within ±0.10? |
|---|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | 0.012 [-0.021, 0.042] | yes |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | 0.033 [-0.021, 0.085] | yes |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | 0.021 [-0.019, 0.059] | yes |

## Diagnostic power for the predeclared 0.10-point change

This empirical paired-source bootstrap reuses the observed source-level residual distribution. It is a planning diagnostic for this item pool, not a guarantee of power on a broader population.

| First | Second | Sources | Minimum two-sided power | Null mean interval |
|---|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | 24 | 100.0% | [-0.033, +0.030] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | 24 | 96.2% | [-0.050, +0.054] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | 24 | 99.8% | [-0.042, +0.040] |

## Measurement reliability

| Model | Permutation-invariant cells | Mean wording range | Mean order range |
|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 46.9% | 0.667 | 0.604 |
| allenai/Olmo-3-7B-Instruct-DPO | 40.6% | 0.604 | 0.646 |
| allenai/Olmo-3-7B-Instruct-SFT | 40.6% | 0.674 | 0.708 |

Paired stage contrasts in measurement reliability are shown below. The simultaneous intervals control the family of nine reliability contrasts with a source-clustered max-|t| bootstrap.

| First | Second | Metric | Difference | Simultaneous 95% CI |
|---|---|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | permutation_invariance | -0.062 | [-0.153, +0.028] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | wording_range | -0.062 | [-0.187, +0.062] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | order_range | +0.042 | [-0.082, +0.165] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | permutation_invariance | -0.062 | [-0.196, +0.071] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | wording_range | +0.007 | [-0.174, +0.188] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | order_range | +0.104 | [-0.051, +0.259] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | permutation_invariance | +0.000 | [-0.105, +0.105] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | wording_range | +0.069 | [-0.065, +0.204] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | order_range | +0.062 | [-0.045, +0.170] |

Across models, mean wording range was 0.648 and mean option-order range was 0.653, versus a marginal model-stage spread of 0.033 (19.6× and 19.8×, respectively).

Stage ranking by wording (highest mean first):

- original: allenai/Olmo-3-7B-Instruct-SFT (0.472) > allenai/Olmo-3-7B-Instruct-DPO (0.444) > allenai/Olmo-3-7B-Instruct (0.403)
- p1: allenai/Olmo-3-7B-Instruct-SFT (0.375) > allenai/Olmo-3-7B-Instruct-DPO (0.354) > allenai/Olmo-3-7B-Instruct (0.347)
- p2: allenai/Olmo-3-7B-Instruct (0.451) > allenai/Olmo-3-7B-Instruct-DPO (0.444) > allenai/Olmo-3-7B-Instruct-SFT (0.403)
- p3: allenai/Olmo-3-7B-Instruct-SFT (0.403) > allenai/Olmo-3-7B-Instruct-DPO (0.326) > allenai/Olmo-3-7B-Instruct (0.319)

## Descriptive variance decomposition

| Component | Sum of squares | Share of total |
|---|---:|---:|
| model | 0.321 | 0.1% |
| source_item | 113.777 | 21.5% |
| wording | 2.946 | 0.6% |
| option_order | 32.274 | 6.1% |
| model × wording | 0.906 | 0.2% |
| model × option_order | 2.061 | 0.4% |
| other interactions / residual | 376.756 | 71.2% |
