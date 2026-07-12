# Historical Layout Reproduction: NF4/Double-Quant Factorial

Each factorial condition below exactly matches the historical semantic layout: source order for the original pack and order 012 for candidate packs.

| Pack | Model | Historical mean | Factorial mean | Difference [95% CI] | Score agreement | Choice agreement |
|---|---|---:|---:|---:|---:|---:|
| original | allenai/Olmo-3-7B-Instruct | 0.417 | 0.417 | +0.000 [-0.167, +0.167] | 83.3% | 83.3% |
| original | allenai/Olmo-3-7B-Instruct-DPO | 0.458 | 0.375 | -0.083 [-0.292, +0.125] | 75.0% | 75.0% |
| original | allenai/Olmo-3-7B-Instruct-SFT | 0.500 | 0.375 | -0.125 [-0.333, +0.042] | 83.3% | 83.3% |
| p1 | allenai/Olmo-3-7B-Instruct | 0.583 | 0.375 | -0.208 [-0.417, +0.000] | 70.8% | 70.8% |
| p1 | allenai/Olmo-3-7B-Instruct-DPO | 0.667 | 0.417 | -0.250 [-0.458, -0.042] | 66.7% | 66.7% |
| p1 | allenai/Olmo-3-7B-Instruct-SFT | 0.625 | 0.333 | -0.292 [-0.500, -0.083] | 62.5% | 62.5% |
| p2 | allenai/Olmo-3-7B-Instruct | 0.750 | 0.417 | -0.333 [-0.542, -0.167] | 66.7% | 66.7% |
| p2 | allenai/Olmo-3-7B-Instruct-DPO | 0.792 | 0.375 | -0.417 [-0.625, -0.208] | 58.3% | 58.3% |
| p2 | allenai/Olmo-3-7B-Instruct-SFT | 0.792 | 0.458 | -0.333 [-0.542, -0.167] | 66.7% | 66.7% |

Different scored cells: 64.

A mismatch means the historical protocol was not reproduced; it can reflect quantization, dependency, model-revision, or other inference drift.
