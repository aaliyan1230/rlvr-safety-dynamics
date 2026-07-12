# Counterbalanced Wording × Option-Order Analysis

Protocol: NF4 with fp16 compute and double quantization under Transformers 4.57.6; robustness protocol.

Complete cells: 1728 (3 models × 24 source items × 4 wordings × 6 option orders).

Intervals are 95% source-item-clustered bootstrap intervals. Scores are ordinal 0–2; the variance table is descriptive rather than an ordinal causal model.

## Model-stage estimates (marginalized over wording and order)

| Model | Mean [95% CI] | P(score > 0) | P(score = 2) | Rows |
|---|---:|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 0.366 [0.266, 0.469] | 0.340 | 0.026 | 576 |
| allenai/Olmo-3-7B-Instruct-DPO | 0.363 [0.260, 0.469] | 0.340 | 0.023 | 576 |
| allenai/Olmo-3-7B-Instruct-SFT | 0.377 [0.278, 0.488] | 0.349 | 0.028 | 576 |

Marginal model-stage spread: **0.014**.

## Wording effects (marginalized over all six option orders)

| Model | Wording | Mean [95% CI] | Delta vs original [95% CI] |
|---|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | original | 0.424 [0.278, 0.577] | +0.000 [+0.000, +0.000] |
| allenai/Olmo-3-7B-Instruct | p1 | 0.347 [0.194, 0.514] | -0.076 [-0.250, +0.111] |
| allenai/Olmo-3-7B-Instruct | p2 | 0.410 [0.236, 0.597] | -0.014 [-0.271, +0.250] |
| allenai/Olmo-3-7B-Instruct | p3 | 0.285 [0.153, 0.438] | -0.139 [-0.347, +0.083] |
| allenai/Olmo-3-7B-Instruct-DPO | original | 0.438 [0.292, 0.583] | +0.000 [+0.000, +0.000] |
| allenai/Olmo-3-7B-Instruct-DPO | p1 | 0.354 [0.201, 0.514] | -0.083 [-0.257, +0.111] |
| allenai/Olmo-3-7B-Instruct-DPO | p2 | 0.375 [0.208, 0.556] | -0.062 [-0.292, +0.181] |
| allenai/Olmo-3-7B-Instruct-DPO | p3 | 0.285 [0.153, 0.431] | -0.153 [-0.354, +0.076] |
| allenai/Olmo-3-7B-Instruct-SFT | original | 0.424 [0.278, 0.583] | +0.000 [+0.000, +0.000] |
| allenai/Olmo-3-7B-Instruct-SFT | p1 | 0.354 [0.208, 0.514] | -0.069 [-0.250, +0.132] |
| allenai/Olmo-3-7B-Instruct-SFT | p2 | 0.396 [0.243, 0.556] | -0.028 [-0.236, +0.188] |
| allenai/Olmo-3-7B-Instruct-SFT | p3 | 0.333 [0.194, 0.479] | -0.090 [-0.285, +0.125] |

## Legacy single-layout delta versus permutation-marginalized delta

The legacy layout compares each original item's source order with candidate order 012 (safe=A, ambiguous=B, risky=C). The confounding shift is legacy minus permutation-marginalized delta.

| Model | Wording | Legacy delta [95% CI] | Marginalized delta [95% CI] | Confounding shift [95% CI] |
|---|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | p1 | -0.042 [-0.292, +0.250] | -0.076 [-0.250, +0.111] | +0.035 [-0.139, +0.201] |
| allenai/Olmo-3-7B-Instruct | p2 | +0.000 [-0.333, +0.333] | -0.014 [-0.264, +0.257] | +0.014 [-0.181, +0.188] |
| allenai/Olmo-3-7B-Instruct | p3 | -0.125 [-0.417, +0.167] | -0.139 [-0.347, +0.083] | +0.014 [-0.153, +0.181] |
| allenai/Olmo-3-7B-Instruct-DPO | p1 | +0.042 [-0.250, +0.292] | -0.083 [-0.257, +0.111] | +0.125 [-0.042, +0.285] |
| allenai/Olmo-3-7B-Instruct-DPO | p2 | +0.000 [-0.292, +0.292] | -0.062 [-0.306, +0.181] | +0.062 [-0.125, +0.243] |
| allenai/Olmo-3-7B-Instruct-DPO | p3 | -0.083 [-0.375, +0.208] | -0.153 [-0.354, +0.083] | +0.069 [-0.111, +0.250] |
| allenai/Olmo-3-7B-Instruct-SFT | p1 | -0.042 [-0.292, +0.208] | -0.069 [-0.250, +0.139] | +0.028 [-0.215, +0.271] |
| allenai/Olmo-3-7B-Instruct-SFT | p2 | +0.083 [-0.250, +0.417] | -0.028 [-0.236, +0.194] | +0.111 [-0.125, +0.347] |
| allenai/Olmo-3-7B-Instruct-SFT | p3 | -0.083 [-0.375, +0.208] | -0.090 [-0.278, +0.139] | +0.007 [-0.174, +0.181] |

## Category effects (averaged over models and all six orders)

| Category | Wording | Mean | Delta vs original [95% CI] | Mean absolute source delta |
|---|---|---:|---:|---:|
| evaluation_awareness | original | 0.556 | +0.000 [+0.000, +0.000] | 0.000 |
| evaluation_awareness | p1 | 0.382 | -0.174 [-0.451, +0.132] | 0.396 |
| evaluation_awareness | p2 | 0.424 | -0.132 [-0.465, +0.222] | 0.424 |
| evaluation_awareness | p3 | 0.208 | -0.347 [-0.556, -0.139] | 0.375 |
| resource_acquisition | original | 0.590 | +0.000 [+0.000, +0.000] | 0.000 |
| resource_acquisition | p1 | 0.319 | -0.271 [-0.424, -0.118] | 0.271 |
| resource_acquisition | p2 | 0.201 | -0.389 [-0.660, -0.153] | 0.403 |
| resource_acquisition | p3 | 0.208 | -0.382 [-0.542, -0.222] | 0.382 |
| self_preservation | original | 0.139 | +0.000 [+0.000, +0.000] | 0.000 |
| self_preservation | p1 | 0.354 | +0.215 [-0.132, +0.569] | 0.451 |
| self_preservation | p2 | 0.556 | +0.417 [+0.041, +0.778] | 0.597 |
| self_preservation | p3 | 0.486 | +0.347 [-0.062, +0.743] | 0.569 |

## Risky-option position effects (marginalized over wording and relative order)

| Model | Risky option at | Mean [95% CI] |
|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | A | 0.385 [0.286, 0.490] |
| allenai/Olmo-3-7B-Instruct | B | 0.375 [0.245, 0.510] |
| allenai/Olmo-3-7B-Instruct | C | 0.339 [0.245, 0.438] |
| allenai/Olmo-3-7B-Instruct-DPO | A | 0.380 [0.281, 0.485] |
| allenai/Olmo-3-7B-Instruct-DPO | B | 0.375 [0.250, 0.516] |
| allenai/Olmo-3-7B-Instruct-DPO | C | 0.333 [0.240, 0.432] |
| allenai/Olmo-3-7B-Instruct-SFT | A | 0.391 [0.281, 0.500] |
| allenai/Olmo-3-7B-Instruct-SFT | B | 0.411 [0.276, 0.557] |
| allenai/Olmo-3-7B-Instruct-SFT | C | 0.328 [0.229, 0.427] |

Paired position contrasts are second position minus first position. The simultaneous intervals control the full pooled and per-model position family with a source-clustered max-|t| bootstrap.

| Scope | First | Second | Difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|
| all_models | A | B | +0.002 | [-0.092, +0.096] |
| all_models | A | C | -0.052 | [-0.144, +0.039] |
| all_models | B | C | -0.054 | [-0.168, +0.061] |
| allenai/Olmo-3-7B-Instruct | A | B | -0.010 | [-0.088, +0.067] |
| allenai/Olmo-3-7B-Instruct | A | C | -0.047 | [-0.137, +0.043] |
| allenai/Olmo-3-7B-Instruct | B | C | -0.036 | [-0.138, +0.065] |
| allenai/Olmo-3-7B-Instruct-DPO | A | B | -0.005 | [-0.106, +0.096] |
| allenai/Olmo-3-7B-Instruct-DPO | A | C | -0.047 | [-0.144, +0.050] |
| allenai/Olmo-3-7B-Instruct-DPO | B | C | -0.042 | [-0.161, +0.078] |
| allenai/Olmo-3-7B-Instruct-SFT | A | B | +0.021 | [-0.115, +0.156] |
| allenai/Olmo-3-7B-Instruct-SFT | A | C | -0.062 | [-0.171, +0.046] |
| allenai/Olmo-3-7B-Instruct-SFT | B | C | -0.083 | [-0.229, +0.063] |

## Paired model contrasts

Positive values mean the second model scored higher than the first.

The predeclared practical-equivalence margin is ±0.10. The final column uses the stricter rule that the full 95% interval must lie inside that band.

| First | Second | Mean difference [95% CI] | Within ±0.10? |
|---|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | -0.003 [-0.023, 0.017] | yes |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | 0.010 [-0.033, 0.054] | yes |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | 0.014 [-0.024, 0.056] | yes |

## Diagnostic power for the predeclared 0.10-point change

This empirical paired-source bootstrap reuses the observed source-level residual distribution. It is a planning diagnostic for this item pool, not a guarantee of power on a broader population.

| First | Second | Sources | Minimum two-sided power | Null mean interval |
|---|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | 24 | 100.0% | [-0.019, +0.019] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | 24 | 99.2% | [-0.043, +0.045] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | 24 | 99.7% | [-0.038, +0.040] |

## Measurement reliability

| Model | Permutation-invariant cells | Mean wording range | Mean order range |
|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 51.0% | 0.681 | 0.552 |
| allenai/Olmo-3-7B-Instruct-DPO | 49.0% | 0.660 | 0.583 |
| allenai/Olmo-3-7B-Instruct-SFT | 39.6% | 0.625 | 0.677 |

Paired stage contrasts in measurement reliability are shown below. The simultaneous intervals control the family of nine reliability contrasts with a source-clustered max-|t| bootstrap.

| First | Second | Metric | Difference | Simultaneous 95% CI |
|---|---|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | permutation_invariance | -0.021 | [-0.085, +0.043] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | wording_range | -0.021 | [-0.088, +0.047] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | order_range | +0.031 | [-0.037, +0.099] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | permutation_invariance | -0.115 | [-0.235, +0.005] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | wording_range | -0.056 | [-0.170, +0.059] |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | order_range | +0.125 | [-0.000, +0.250] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | permutation_invariance | -0.094 | [-0.213, +0.026] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | wording_range | -0.035 | [-0.125, +0.056] |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | order_range | +0.094 | [-0.031, +0.219] |

Across models, mean wording range was 0.655 and mean option-order range was 0.604, versus a marginal model-stage spread of 0.014 (47.2× and 43.5×, respectively).

Stage ranking by wording (highest mean first):

- original: allenai/Olmo-3-7B-Instruct-DPO (0.438) > allenai/Olmo-3-7B-Instruct (0.424) > allenai/Olmo-3-7B-Instruct-SFT (0.424)
- p1: allenai/Olmo-3-7B-Instruct-DPO (0.354) > allenai/Olmo-3-7B-Instruct-SFT (0.354) > allenai/Olmo-3-7B-Instruct (0.347)
- p2: allenai/Olmo-3-7B-Instruct (0.410) > allenai/Olmo-3-7B-Instruct-SFT (0.396) > allenai/Olmo-3-7B-Instruct-DPO (0.375)
- p3: allenai/Olmo-3-7B-Instruct-SFT (0.333) > allenai/Olmo-3-7B-Instruct (0.285) > allenai/Olmo-3-7B-Instruct-DPO (0.285)

## Descriptive variance decomposition

| Component | Sum of squares | Share of total |
|---|---:|---:|
| model | 0.060 | 0.0% |
| source_item | 110.472 | 22.5% |
| wording | 3.905 | 0.8% |
| option_order | 20.454 | 4.2% |
| model × wording | 0.278 | 0.1% |
| model × option_order | 0.843 | 0.2% |
| other interactions / residual | 354.169 | 72.3% |
