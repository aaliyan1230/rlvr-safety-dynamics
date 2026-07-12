# NF4/Double-Quant versus Default 4-bit

Matched cells: 1728. Deltas are **default_4bit minus nf4_double_quant**. Intervals are paired 95% source-item-clustered bootstrap intervals.

## Cell-level agreement

Score agreement compares semantic 0-2 scores; response-letter agreement compares literal A/B/C outputs.

| Scope | Rows | Score agreement | Response-letter agreement |
|---|---:|---:|---:|
| All models | 1728 | 85.4% | 85.4% |
| allenai/Olmo-3-7B-Instruct | 576 | 84.5% | 84.5% |
| allenai/Olmo-3-7B-Instruct-DPO | 576 | 85.1% | 85.1% |
| allenai/Olmo-3-7B-Instruct-SFT | 576 | 86.6% | 86.6% |

## Overall paired protocol score difference

| Scope | nf4_double_quant mean | default_4bit mean | Difference [95% CI] |
|---|---:|---:|---:|
| All cells | 0.369 | 0.395 | +0.027 [-0.015, +0.068] |

## Model-marginal score differences

| Model | nf4_double_quant mean | default_4bit mean | Difference [95% CI] |
|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 0.366 | 0.380 | +0.014 [-0.035, +0.062] |
| allenai/Olmo-3-7B-Instruct-DPO | 0.363 | 0.392 | +0.030 [-0.024, +0.087] |
| allenai/Olmo-3-7B-Instruct-SFT | 0.377 | 0.413 | +0.036 [-0.003, +0.080] |

## Wording-marginal score differences

| Model | Wording | nf4_double_quant mean | default_4bit mean | Difference [95% CI] |
|---|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | original | 0.424 | 0.403 | -0.021 [-0.111, +0.062] |
| allenai/Olmo-3-7B-Instruct | p1 | 0.347 | 0.347 | +0.000 [-0.069, +0.069] |
| allenai/Olmo-3-7B-Instruct | p2 | 0.410 | 0.451 | +0.042 [-0.049, +0.146] |
| allenai/Olmo-3-7B-Instruct | p3 | 0.285 | 0.319 | +0.035 [-0.014, +0.084] |
| allenai/Olmo-3-7B-Instruct-DPO | original | 0.438 | 0.444 | +0.007 [-0.090, +0.097] |
| allenai/Olmo-3-7B-Instruct-DPO | p1 | 0.354 | 0.354 | +0.000 [-0.069, +0.069] |
| allenai/Olmo-3-7B-Instruct-DPO | p2 | 0.375 | 0.444 | +0.069 [+0.000, +0.153] |
| allenai/Olmo-3-7B-Instruct-DPO | p3 | 0.285 | 0.326 | +0.042 [-0.035, +0.118] |
| allenai/Olmo-3-7B-Instruct-SFT | original | 0.424 | 0.472 | +0.049 [-0.056, +0.181] |
| allenai/Olmo-3-7B-Instruct-SFT | p1 | 0.354 | 0.375 | +0.021 [-0.035, +0.076] |
| allenai/Olmo-3-7B-Instruct-SFT | p2 | 0.396 | 0.403 | +0.007 [-0.049, +0.062] |
| allenai/Olmo-3-7B-Instruct-SFT | p3 | 0.333 | 0.403 | +0.069 [+0.000, +0.146] |

## Risky-position-marginal score differences

| Model | Risky position | nf4_double_quant mean | default_4bit mean | Difference [95% CI] |
|---|---|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | A | 0.385 | 0.323 | -0.062 [-0.120, -0.005] |
| allenai/Olmo-3-7B-Instruct | B | 0.375 | 0.432 | +0.057 [-0.005, +0.115] |
| allenai/Olmo-3-7B-Instruct | C | 0.339 | 0.385 | +0.047 [-0.021, +0.120] |
| allenai/Olmo-3-7B-Instruct-DPO | A | 0.380 | 0.344 | -0.036 [-0.104, +0.031] |
| allenai/Olmo-3-7B-Instruct-DPO | B | 0.375 | 0.443 | +0.068 [+0.000, +0.141] |
| allenai/Olmo-3-7B-Instruct-DPO | C | 0.333 | 0.391 | +0.057 [-0.005, +0.120] |
| allenai/Olmo-3-7B-Instruct-SFT | A | 0.391 | 0.333 | -0.057 [-0.110, -0.005] |
| allenai/Olmo-3-7B-Instruct-SFT | B | 0.411 | 0.505 | +0.094 [+0.026, +0.161] |
| allenai/Olmo-3-7B-Instruct-SFT | C | 0.328 | 0.401 | +0.073 [+0.010, +0.141] |

## Change in risky-position effects between protocols

Each position effect is second position minus first position. The final column is the default_4bit effect minus the nf4_double_quant effect.

| Scope | First | Second | nf4_double_quant effect | default_4bit effect | Effect difference [95% CI] |
|---|---:|---:|---:|---:|---:|
| all_models | A | B | +0.002 | +0.127 | +0.125 [+0.066, +0.186] |
| all_models | A | C | -0.052 | +0.059 | +0.111 [+0.061, +0.161] |
| all_models | B | C | -0.054 | -0.068 | -0.014 [-0.061, +0.033] |
| allenai/Olmo-3-7B-Instruct | A | B | -0.010 | +0.109 | +0.120 [+0.057, +0.188] |
| allenai/Olmo-3-7B-Instruct | A | C | -0.047 | +0.062 | +0.109 [+0.047, +0.177] |
| allenai/Olmo-3-7B-Instruct | B | C | -0.036 | -0.047 | -0.010 [-0.083, +0.068] |
| allenai/Olmo-3-7B-Instruct-DPO | A | B | -0.005 | +0.099 | +0.104 [+0.036, +0.177] |
| allenai/Olmo-3-7B-Instruct-DPO | A | C | -0.047 | +0.047 | +0.094 [+0.036, +0.151] |
| allenai/Olmo-3-7B-Instruct-DPO | B | C | -0.042 | -0.052 | -0.010 [-0.078, +0.057] |
| allenai/Olmo-3-7B-Instruct-SFT | A | B | +0.021 | +0.172 | +0.151 [+0.078, +0.229] |
| allenai/Olmo-3-7B-Instruct-SFT | A | C | -0.062 | +0.068 | +0.130 [+0.047, +0.224] |
| allenai/Olmo-3-7B-Instruct-SFT | B | C | -0.083 | -0.104 | -0.021 [-0.099, +0.062] |
