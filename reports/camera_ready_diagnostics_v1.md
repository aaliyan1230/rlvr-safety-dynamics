# Camera-ready diagnostics from saved Tülu responses

These checks use existing generations only. The original frozen analysis and claim gates are unchanged. The fixed-order and original-wording results below are post hoc diagnostics; they do not establish a new confirmatory safety-drift result.

## Recomputed measurement metrics

| Step | Permutation invariance | Mean wording range | Mean order range | Valid / 576 |
|---:|---:|---:|---:|---:|
| 0 | 0.302 | 0.799 | 0.781 | 576 / 576 |
| 40 | 0.281 | 0.757 | 0.823 | 576 / 576 |
| 80 | 0.323 | 0.757 | 0.771 | 576 / 576 |
| 160 | 0.271 | 0.715 | 0.854 | 576 / 576 |
| 320 | 0.302 | 0.778 | 0.812 | 576 / 576 |
| 640 | 0.250 | 0.736 | 0.885 | 576 / 576 |
| 960 | 0.250 | 0.715 | 0.896 | 576 / 576 |
| 1,280 | 0.198 | 0.681 | 0.969 | 576 / 576 |
| 1,600 | 0.281 | 0.736 | 0.844 | 576 / 576 |
| 1,920 | 0.302 | 0.775 | 0.823 | 575 / 576 |
| 2,240 | 0.156 | 0.681 | 1.010 | 576 / 576 |
| 2,440 | 0.177 | 0.694 | 0.948 | 576 / 576 |

The wording range was absent from the submitted results table, although it is present in the saved trajectory metrics. This table makes its definition and checkpoint values visible. All values above reproduce the frozen metrics from the six saved score files.

## Risk score by fixed option order

The order code gives the semantic classes assigned to A/B/C, with 0=safe, 1=conditional, and 2=risky. A fixed order can therefore produce a different score even when the semantic options are unchanged.

| Option order | Step 0 mean | Step 1,920 mean | Step 2,240 mean | Step 0→2,240 change | Simultaneous 95% source-bootstrap CI |
|:---:|---:|---:|---:|---:|---:|
| 012 | 0.438 | 0.469 | 0.458 | +0.021 | [-0.053, +0.095] |
| 021 | 0.281 | 0.302 | 0.302 | +0.021 | [-0.055, +0.096] |
| 102 | 0.385 | 0.375 | 0.333 | -0.052 | [-0.119, +0.014] |
| 120 | 0.771 | 0.760 | 0.948 | +0.177 | [+0.073, +0.281] |
| 201 | 0.365 | 0.347 | 0.312 | -0.052 | [-0.129, +0.025] |
| 210 | 0.698 | 0.802 | 0.927 | +0.229 | [+0.057, +0.401] |

The six contrasts resample the same 24 authored sources and use one max-|t| simultaneous interval family. They describe how fixed layouts behave in this item pool; they do not redefine the primary order-marginalized estimand.

## Original wording only

The source-clustered step-0 to step-2,240 contrast using only the original wording and all six option orders is **+0.083** (95% percentile bootstrap CI [+0.000, +0.160]; 24 sources). This is a post hoc sensitivity check because paraphrases were part of the frozen primary instrument.

## Provenance

- Bootstrap repetitions: 5000; seed: 20260927.
- Frozen metric input SHA-256: `146ee12484d60b814ac0a6f4103c9a95fd92c1094e581984932871f19b1b1f09`.
- `artifacts/tulu_trajectory_wave_01/choice_scores.csv` SHA-256: `ce8823eb7c1b0fefa1ad970265271c3dae5613fdb0317fae0aa970ba9b634c5f`.
- `artifacts/tulu_trajectory_wave_02/choice_scores.csv` SHA-256: `702d678dfa5a26f313c5d396610b810913cf89004dbfc739832c81c904c981ac`.
- `artifacts/tulu_trajectory_wave_03/choice_scores.csv` SHA-256: `2fd5e369b1aa1ec92f5fa65ec91b1fe3db185059c076b4b5c124fdac7e15e7c8`.
- `artifacts/tulu_trajectory_wave_04/choice_scores.csv` SHA-256: `74e5345ef0af20a0f03c676832b7adc154d674b00ad4c7ae5380dfa2d95ea96c`.
- `artifacts/tulu_trajectory_wave_05/choice_scores.csv` SHA-256: `624886e6470c956e956c970c0612e2eb325fe721d559c295cb75772fa8b03b2d`.
- `artifacts/tulu_trajectory_wave_06/choice_scores.csv` SHA-256: `b8f32b09a19cd7eeb05ab9ea7b03f8744b5f3ce16ac92810dd4d9b5c4f9b1a38`.
