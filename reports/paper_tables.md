# Paper Tables

## Table 1: Free-Form Manual Scores

| Model | Mean risk | Counts | Confounded risk rows |
|---|---:|---|---:|
| allenai/Olmo-3-7B-Instruct | 0.38 | 27x0, 11x1, 2x2 | 0/40 |
| allenai/Olmo-3-7B-RL-Zero-General | 0.35 | 26x0, 14x1, 0x2 | 16/40 |

## Table 2: Structured Choice Scores

| Model | Mean parsed risk | Counts | Malformed | Token-cap rows |
|---|---:|---|---:|---:|
| allenai/Olmo-3-7B-Instruct | 0.42 | 14x0, 10x1, 0x2 | 0/24 | 0/24 |
| allenai/Olmo-3-7B-Instruct-DPO | 0.46 | 13x0, 11x1, 0x2 | 0/24 | 0/24 |
| allenai/Olmo-3-7B-Instruct-SFT | 0.50 | 13x0, 10x1, 1x2 | 0/24 | 0/24 |
| allenai/Olmo-3-7B-RL-Zero-General | 0.44 | 9x0, 7x1, 0x2 | 8/24 | 24/24 |

## Table 3: Gemini Judgeability

| Source | Model | Mean Gemini risk | Mean judgeability problem | Needs review | Failure modes |
|---|---|---:|---:|---:|---|
| gemini_choice_stage_ablation_v2 | allenai/Olmo-3-7B-Instruct | 0.42 | 0.00 | 0/24 | clean:24 |
| gemini_choice_stage_ablation_v2 | allenai/Olmo-3-7B-Instruct-DPO | 0.46 | 0.00 | 0/24 | clean:24 |
| gemini_choice_stage_ablation_v2 | allenai/Olmo-3-7B-Instruct-SFT | 0.50 | 0.00 | 0/24 | clean:24 |
| gemini_choice_stage_ablation_v2 | allenai/Olmo-3-7B-RL-Zero-General | 0.75 | 1.38 | 4/24 | malformed:5, other:4, token_capped:5, verbose_preamble:10 |
| gemini_olmo_rlzero_full_v1 | allenai/Olmo-3-7B-Instruct | 0.25 | 0.00 | 0/48 | clean:48 |
| gemini_olmo_rlzero_full_v1 | allenai/Olmo-3-7B-RL-Zero-General | 0.21 | 0.56 | 9/48 | clean:34, token_capped:14 |

## Agreement: gemini_olmo_rlzero_full_v1

* Comparable rows: 96
* Exact agreement: 79/96 (82.29%)
* Binary nonzero-risk agreement: 85/96 (88.54%)

| Manual | Gemini | Count |
|---:|---:|---:|
| 0 | 0 | 69 |
| 1 | 0 | 11 |
| 1 | 1 | 9 |
| 1 | 2 | 5 |
| 2 | 1 | 1 |
| 2 | 2 | 1 |

## Agreement: gemini_choice_stage_ablation_v2

* Comparable rows: 88
* Exact agreement: 88/88 (100.00%)
* Binary nonzero-risk agreement: 88/88 (100.00%)

| Manual | Gemini | Count |
|---:|---:|---:|
| 0 | 0 | 49 |
| 1 | 1 | 38 |
| 2 | 2 | 1 |

## Table 4: Paraphrase Robustness (Original vs Paraphrase)

| Model | Original mean | Paraphrase mean | Delta | Mean abs delta | Items changed |
|---|---:|---:|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 0.42 | 0.58 | +0.17 | 0.58 | 13/24 |
| allenai/Olmo-3-7B-Instruct-DPO | 0.46 | 0.67 | +0.21 | 0.54 | 12/24 |
| allenai/Olmo-3-7B-Instruct-SFT | 0.50 | 0.62 | +0.12 | 0.62 | 14/24 |
