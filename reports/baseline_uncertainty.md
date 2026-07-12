# Baseline Paired Uncertainty Analysis

Bootstrap intervals resample the 24 or 40 authored source items. They are diagnostic uncertainty intervals, not population-level confidence intervals.

## Adjudicated free-form comparison

| Instruct mean | RL-Zero mean | RL-Zero − Instruct [95% interval] | Higher / tied / lower |
|---:|---:|---:|---:|
| 0.400 | 0.325 | -0.075 [-0.300, +0.125] | 5 / 29 / 6 |

On the 24 pairs where the RL-Zero row was not manually marked verbosity/token-cap confounded, means were 0.333 (Instruct) and 0.292 (RL-Zero).

## Clean structured-choice stages

| Model | Mean |
|---|---:|
| allenai/Olmo-3-7B-Instruct | 0.417 |
| allenai/Olmo-3-7B-Instruct-DPO | 0.458 |
| allenai/Olmo-3-7B-Instruct-SFT | 0.500 |

All three clean models made the same scored choice on 22/24 items. The marginal model-stage spread is 0.083.

| First | Second | Second − first [95% interval] | Nonzero paired items |
|---|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-DPO | +0.042 [+0.000, +0.125] | 1/24 |
| allenai/Olmo-3-7B-Instruct | allenai/Olmo-3-7B-Instruct-SFT | +0.083 [+0.000, +0.250] | 1/24 |
| allenai/Olmo-3-7B-Instruct-DPO | allenai/Olmo-3-7B-Instruct-SFT | +0.042 [-0.125, +0.250] | 2/24 |
