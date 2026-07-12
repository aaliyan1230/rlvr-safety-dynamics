# Exact P1/P2-Runtime Candidate-Cell Reproduction

Each factorial condition below exactly matches the historical semantic layout: source order for the original pack and order 012 for candidate packs.

| Pack | Model | Historical mean | Factorial mean | Difference [95% CI] | Score agreement | Choice agreement |
|---|---|---:|---:|---:|---:|---:|
| p1 | allenai/Olmo-3-7B-Instruct | 0.583 | 0.583 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |
| p1 | allenai/Olmo-3-7B-Instruct-DPO | 0.667 | 0.667 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |
| p1 | allenai/Olmo-3-7B-Instruct-SFT | 0.625 | 0.625 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |
| p2 | allenai/Olmo-3-7B-Instruct | 0.750 | 0.750 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |
| p2 | allenai/Olmo-3-7B-Instruct-DPO | 0.792 | 0.792 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |
| p2 | allenai/Olmo-3-7B-Instruct-SFT | 0.792 | 0.792 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |

Different scored cells: 0.

A mismatch means the historical protocol was not reproduced; it can reflect quantization, dependency, model-revision, or other inference drift.
