# Exact Stage-Runtime Original-Cell Reproduction

Each factorial condition below exactly matches the historical semantic layout: source order for the original pack and order 012 for candidate packs.

| Pack | Model | Historical mean | Factorial mean | Difference [95% CI] | Score agreement | Choice agreement |
|---|---|---:|---:|---:|---:|---:|
| original | allenai/Olmo-3-7B-Instruct | 0.417 | 0.417 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |
| original | allenai/Olmo-3-7B-Instruct-DPO | 0.458 | 0.458 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |
| original | allenai/Olmo-3-7B-Instruct-SFT | 0.500 | 0.500 | +0.000 [+0.000, +0.000] | 100.0% | 100.0% |

Different scored cells: 0.

A mismatch means the historical protocol was not reproduced; it can reflect quantization, dependency, model-revision, or other inference drift.
