# Default-4-bit Standardized Run: Historical Layout Reproduction

Each factorial condition below exactly matches the historical semantic layout: source order for the original pack and order 012 for candidate packs.

| Pack | Model | Historical mean | Factorial mean | Difference [95% CI] | Score agreement | Choice agreement |
|---|---|---:|---:|---:|---:|---:|
| original | allenai/Olmo-3-7B-Instruct | 0.417 | 0.375 | -0.042 [-0.208, +0.125] | 79.2% | 79.2% |
| original | allenai/Olmo-3-7B-Instruct-DPO | 0.458 | 0.458 | +0.000 [-0.167, +0.167] | 83.3% | 83.3% |
| original | allenai/Olmo-3-7B-Instruct-SFT | 0.500 | 0.458 | -0.042 [-0.292, +0.167] | 83.3% | 83.3% |
| p1 | allenai/Olmo-3-7B-Instruct | 0.583 | 0.458 | -0.125 [-0.292, +0.042] | 79.2% | 79.2% |
| p1 | allenai/Olmo-3-7B-Instruct-DPO | 0.667 | 0.542 | -0.125 [-0.333, +0.083] | 70.8% | 70.8% |
| p1 | allenai/Olmo-3-7B-Instruct-SFT | 0.625 | 0.542 | -0.083 [-0.250, +0.083] | 83.3% | 83.3% |
| p2 | allenai/Olmo-3-7B-Instruct | 0.750 | 0.625 | -0.125 [-0.292, +0.042] | 79.2% | 79.2% |
| p2 | allenai/Olmo-3-7B-Instruct-DPO | 0.792 | 0.583 | -0.208 [-0.375, -0.042] | 79.2% | 79.2% |
| p2 | allenai/Olmo-3-7B-Instruct-SFT | 0.792 | 0.583 | -0.208 [-0.417, +0.000] | 70.8% | 70.8% |

Different scored cells: 46.

A mismatch means the historical protocol was not reproduced; it can reflect quantization, dependency, model-revision, or other inference drift.
