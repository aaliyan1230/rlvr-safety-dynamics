# Exact Historical Runtime Crossover

The two panels contain identical historical layouts. Runtime deltas are P1/P2-image runtime minus stage-image runtime. Intervals resample the 24 source items.

## Runtime agreement

| Scope | Rows | Score agreement | Response-letter agreement |
|---|---:|---:|---:|
| all cells | 216 | 80.6% | 80.6% |
| allenai/Olmo-3-7B-Instruct | 72 | 77.8% | 77.8% |
| allenai/Olmo-3-7B-Instruct-DPO | 72 | 86.1% | 86.1% |
| allenai/Olmo-3-7B-Instruct-SFT | 72 | 77.8% | 77.8% |
| wording=original | 72 | 88.9% | 88.9% |
| wording=p1 | 72 | 77.8% | 77.8% |
| wording=p2 | 72 | 75.0% | 75.0% |

## Runtime score deltas

| Scope | Wording | Stage-runtime mean | P1/P2-runtime mean | Delta [95% CI] |
|---|---|---:|---:|---:|
| all_models | all_wordings | 0.546 | 0.634 | +0.088 [+0.009, +0.171] |
| all_models | original | 0.458 | 0.500 | +0.042 [-0.056, +0.153] |
| all_models | p1 | 0.597 | 0.625 | +0.028 [-0.153, +0.236] |
| all_models | p2 | 0.583 | 0.778 | +0.194 [+0.056, +0.361] |
| allenai/Olmo-3-7B-Instruct | all_wordings | 0.500 | 0.597 | +0.097 [+0.000, +0.208] |
| allenai/Olmo-3-7B-Instruct | original | 0.417 | 0.458 | +0.042 [-0.083, +0.167] |
| allenai/Olmo-3-7B-Instruct | p1 | 0.542 | 0.583 | +0.042 [-0.208, +0.292] |
| allenai/Olmo-3-7B-Instruct | p2 | 0.542 | 0.750 | +0.208 [+0.000, +0.417] |
| allenai/Olmo-3-7B-Instruct-DPO | all_wordings | 0.556 | 0.653 | +0.097 [+0.014, +0.181] |
| allenai/Olmo-3-7B-Instruct-DPO | original | 0.458 | 0.500 | +0.042 [+0.000, +0.125] |
| allenai/Olmo-3-7B-Instruct-DPO | p1 | 0.583 | 0.667 | +0.083 [-0.125, +0.333] |
| allenai/Olmo-3-7B-Instruct-DPO | p2 | 0.625 | 0.792 | +0.167 [+0.042, +0.333] |
| allenai/Olmo-3-7B-Instruct-SFT | all_wordings | 0.583 | 0.653 | +0.069 [-0.069, +0.194] |
| allenai/Olmo-3-7B-Instruct-SFT | original | 0.500 | 0.542 | +0.042 [-0.208, +0.250] |
| allenai/Olmo-3-7B-Instruct-SFT | p1 | 0.667 | 0.625 | -0.042 [-0.208, +0.125] |
| allenai/Olmo-3-7B-Instruct-SFT | p2 | 0.583 | 0.792 | +0.208 [+0.000, +0.417] |

## Candidate-minus-original effects within each runtime

| Runtime | Scope | Candidate | Effect [95% CI] |
|---|---|---|---:|
| stage_runtime | all_models | p1 | +0.139 [-0.111, +0.375] |
| paraphrase_runtime | all_models | p1 | +0.125 [-0.208, +0.444] |
| stage_runtime | all_models | p2 | +0.125 [-0.181, +0.417] |
| paraphrase_runtime | all_models | p2 | +0.278 [-0.069, +0.597] |
| stage_runtime | allenai/Olmo-3-7B-Instruct | p1 | +0.125 [-0.125, +0.375] |
| paraphrase_runtime | allenai/Olmo-3-7B-Instruct | p1 | +0.125 [-0.250, +0.458] |
| stage_runtime | allenai/Olmo-3-7B-Instruct | p2 | +0.125 [-0.167, +0.458] |
| paraphrase_runtime | allenai/Olmo-3-7B-Instruct | p2 | +0.292 [-0.083, +0.667] |
| stage_runtime | allenai/Olmo-3-7B-Instruct-DPO | p1 | +0.125 [-0.125, +0.375] |
| paraphrase_runtime | allenai/Olmo-3-7B-Instruct-DPO | p1 | +0.167 [-0.167, +0.500] |
| stage_runtime | allenai/Olmo-3-7B-Instruct-DPO | p2 | +0.167 [-0.125, +0.458] |
| paraphrase_runtime | allenai/Olmo-3-7B-Instruct-DPO | p2 | +0.292 [-0.083, +0.625] |
| stage_runtime | allenai/Olmo-3-7B-Instruct-SFT | p1 | +0.167 [-0.125, +0.458] |
| paraphrase_runtime | allenai/Olmo-3-7B-Instruct-SFT | p1 | +0.083 [-0.250, +0.417] |
| stage_runtime | allenai/Olmo-3-7B-Instruct-SFT | p2 | +0.083 [-0.292, +0.458] |
| paraphrase_runtime | allenai/Olmo-3-7B-Instruct-SFT | p2 | +0.250 [-0.125, +0.583] |

## Runtime × pack difference-in-differences

Positive values mean the candidate-minus-original shift is larger under the P1/P2 historical runtime.

| Scope | Candidate | Difference-in-differences [95% CI] |
|---|---|---:|
| all_models | p1 | -0.014 [-0.208, +0.208] |
| all_models | p2 | +0.153 [-0.056, +0.361] |
| allenai/Olmo-3-7B-Instruct | p1 | +0.000 [-0.292, +0.292] |
| allenai/Olmo-3-7B-Instruct | p2 | +0.167 [-0.083, +0.417] |
| allenai/Olmo-3-7B-Instruct-DPO | p1 | +0.042 [-0.208, +0.292] |
| allenai/Olmo-3-7B-Instruct-DPO | p2 | +0.125 [-0.042, +0.292] |
| allenai/Olmo-3-7B-Instruct-SFT | p1 | -0.083 [-0.333, +0.125] |
| allenai/Olmo-3-7B-Instruct-SFT | p2 | +0.167 [-0.125, +0.458] |
