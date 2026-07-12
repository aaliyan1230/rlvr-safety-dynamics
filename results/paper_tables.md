# Paper Tables

## Table 1: Free-Form Manual Scores

| Model | Mean risk | Counts | Confounded risk rows |
|---|---:|---|---:|
| allenai/Olmo-3-7B-Instruct | 0.40 | 28x0, 8x1, 4x2 | 0/40 |
| allenai/Olmo-3-7B-RL-Zero-General | 0.33 | 28x0, 11x1, 1x2 | 16/40 |

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
| gemini_choice_judgments | allenai/Olmo-3-7B-Instruct | 0.42 | 0.00 | 0/24 | clean:24 |
| gemini_choice_judgments | allenai/Olmo-3-7B-Instruct-DPO | 0.46 | 0.00 | 0/24 | clean:24 |
| gemini_choice_judgments | allenai/Olmo-3-7B-Instruct-SFT | 0.50 | 0.00 | 0/24 | clean:24 |
| gemini_choice_judgments | allenai/Olmo-3-7B-RL-Zero-General | 0.75 | 1.38 | 4/24 | malformed:5, other:4, token_capped:5, verbose_preamble:10 |
| gemini_freeform_judgments | allenai/Olmo-3-7B-Instruct | 0.25 | 0.00 | 0/48 | clean:48 |
| gemini_freeform_judgments | allenai/Olmo-3-7B-RL-Zero-General | 0.21 | 0.56 | 9/48 | clean:34, token_capped:14 |

## Agreement: gemini_freeform_judgments

* Comparable rows (all): 96
* Exact agreement (all): 85/96 (88.54%)
* Binary nonzero-risk agreement (all): 88/96 (91.67%)
* Exact agreement (risk prompts only): 69/80 (86.25%)
* Binary agreement (risk prompts only): 72/80 (90.00%)

| Manual | Gemini | Count |
|---:|---:|---:|
| 0 | 0 | 72 |
| 1 | 0 | 8 |
| 1 | 1 | 9 |
| 1 | 2 | 2 |
| 2 | 1 | 1 |
| 2 | 2 | 4 |

## Agreement: gemini_choice_judgments

* Comparable rows (all): 88
* Exact agreement (all): 88/88 (100.00%)
* Binary nonzero-risk agreement (all): 88/88 (100.00%)

| Manual | Gemini | Count |
|---:|---:|---:|
| 0 | 0 | 49 |
| 1 | 1 | 38 |
| 2 | 2 | 1 |

## Table 4: Legacy Confounded Protocol Shift (Original vs P1)

**Diagnostic only.** Wording and semantic option position changed together, so these deltas do not identify a wording effect.

| Model | Original mean | Paraphrase mean | Delta | Mean abs delta | Items changed |
|---|---:|---:|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 0.42 | 0.58 | +0.17 | 0.58 | 13/24 |
| allenai/Olmo-3-7B-Instruct-DPO | 0.46 | 0.67 | +0.21 | 0.54 | 12/24 |
| allenai/Olmo-3-7B-Instruct-SFT | 0.50 | 0.62 | +0.12 | 0.62 | 14/24 |

## Table 5: Legacy Confounded Protocol Shift (P1 and P2)

**Diagnostic only.** Both candidate packs fixed safe=A, ambiguous=B, and risky=C while the original varied score positions. The values combine wording and position effects.

| Model | Orig | Seed 1 mean | Seed 2 mean | Seed 1 delta | Seed 2 delta | Seed 1 flip | Seed 2 flip | Seed 1 MAD | Seed 2 MAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | 0.42 | 0.58 | 0.75 | +0.17 | +0.33 | 13/24 | 14/24 | 0.58 | 0.67 |
| allenai/Olmo-3-7B-Instruct-DPO | 0.46 | 0.67 | 0.79 | +0.21 | +0.33 | 12/24 | 16/24 | 0.54 | 0.67 |
| allenai/Olmo-3-7B-Instruct-SFT | 0.50 | 0.62 | 0.79 | +0.12 | +0.29 | 14/24 | 16/24 | 0.62 | 0.71 |

### Category-Level Deltas

| Model | Category | Orig | Seed 1 mean | Seed 2 mean | Seed 1 delta | Seed 2 delta | Seed 1 flip | Seed 2 flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| allenai/Olmo-3-7B-Instruct | evaluation_awareness | 0.38 | 0.50 | 0.50 | +0.12 | +0.12 | 3/8 | 3/8 |
| allenai/Olmo-3-7B-Instruct | resource_acquisition | 0.75 | 0.38 | 0.50 | -0.38 | -0.25 | 3/8 | 4/8 |
| allenai/Olmo-3-7B-Instruct | self_preservation | 0.12 | 0.88 | 1.25 | +0.75 | +1.12 | 7/8 | 7/8 |
| allenai/Olmo-3-7B-Instruct-DPO | evaluation_awareness | 0.50 | 0.62 | 0.75 | +0.12 | +0.25 | 3/8 | 4/8 |
| allenai/Olmo-3-7B-Instruct-DPO | resource_acquisition | 0.75 | 0.38 | 0.62 | -0.38 | -0.12 | 3/8 | 5/8 |
| allenai/Olmo-3-7B-Instruct-DPO | self_preservation | 0.12 | 1.00 | 1.00 | +0.88 | +0.88 | 6/8 | 7/8 |
| allenai/Olmo-3-7B-Instruct-SFT | evaluation_awareness | 0.38 | 0.50 | 0.75 | +0.12 | +0.38 | 5/8 | 5/8 |
| allenai/Olmo-3-7B-Instruct-SFT | resource_acquisition | 0.75 | 0.62 | 0.75 | -0.12 | +0.00 | 3/8 | 4/8 |
| allenai/Olmo-3-7B-Instruct-SFT | self_preservation | 0.38 | 0.75 | 0.88 | +0.38 | +0.50 | 6/8 | 7/8 |

*P1 was evaluated 2026-07-06 and P2 on 2026-07-08. Their historical checks confirmed internal label ordering but did not compare candidates with source items.*
