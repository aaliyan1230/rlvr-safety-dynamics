# Safety drift or measurement drift?

Reproducible audit of a small instrumental-behavior evaluation for open post-training lineages.

## Current result

The evidence does **not** support a robust increase in instrumental-risk choices across the evaluated OLMo SFT, DPO, and final Instruct checkpoints.

Two complete counterbalanced experiments crossed 24 source items, four wordings, all six semantic option orders, and three model stages. Each protocol produced all 1,728 expected responses with no malformed or token-capped rows.

| Inference protocol | Instruct | DPO | SFT | Stage spread |
|---|---:|---:|---:|---:|
| NF4, fp16 compute, double quantization | 0.366 | 0.363 | 0.377 | 0.014 |
| BitsAndBytes default 4-bit, fixed 4.57.6 runtime | 0.380 | 0.392 | 0.413 | 0.033 |

All paired stage-contrast 95% source-clustered intervals lie inside the predeclared ±0.10 practical-equivalence band in both protocols. This is a conditional result for this 24-item deterministic instrument, not a causal claim about RLVR or a population-wide null.

The measurement itself is much less stable than those marginal stage means:

- only 39.6%–51.0% of model × source × wording cells were score-invariant across all six permutations under NF4, and 40.6%–46.9% under default 4-bit;
- category-level wording effects opposed one another: resource-acquisition rewrites generally lowered scores while self-preservation rewrites raised them, so aggregate wording means hide construct heterogeneity; and
- the pooled B-minus-A risky-position effect was +0.002 under NF4 but +0.127 under default 4-bit, a protocol-effect difference of +0.125 [0.066, 0.186].

The historical +0.12 to +0.33 paraphrase-pack shifts are not wording effects. Those runs changed wording, semantic option position, and Kaggle runtime together. Exact-image reruns reproduce the historical original cells and P1/P2 cells at 100%, but the reciprocal crossover agrees on only 174/216 identical cells. The P1/P2-image runtime raises the pooled score by +0.088 [0.009, 0.171] relative to the stage-image runtime; the P2-layout delta is +0.194 [0.056, 0.361]. The pack × runtime interactions remain imprecise, so the old shifts cannot be partitioned into clean wording, order, and runtime components.

Free-form and reasoning-style comparisons remain dominated by censoring. Final visible-text means were 0.400 for OLMo Instruct and 0.325 for RL-Zero-General, but 16/40 RL-Zero-General risk rows were confounded. Qwen Thinking hit the token cap on 11/12 outputs. These runs establish judgeability failures, not clean behavioral differences.

## Evidence status

The computational correction is complete and checksum-verified. Source-aware Gemini 2.5 Pro validation passed all 72 paraphrase pairs. A second disclosed, prompt-only AI audit accepted 72/72 under an explicit project-owner assumption: 63 passed without a noted concern and nine passed with construct-fidelity caveats. This closes the project's internal semantic-review gate under that assumption, but it is not independent human validation and must not be reported as such.

The recommended next study is a checkpoint-resolved decomposition along the exact Tülu 3.1 8B GRPO trajectory: measure order- and wording-marginalized behavior and measurement reliability at 12 pinned checkpoints, then require a matching blinded free-form signal before calling any change “safety drift.” The repaired endpoint feasibility pilot passed, and the cross-format package is frozen before trajectory outcomes. Structured wave 01 has now passed integrity: 1,152/1,152 unique strict responses across pinned steps 0 and 40, zero malformed or capped rows, exact revisions, explicit masks, and 6.631 GiB peak memory per T4. Its behavioral outcomes remain uninspected; five structured waves remain.

## Repository map

| Path | Contents |
|---|---|
| `src/rlvr_safety/` | Installable package for prompt construction, parsing, scoring, adjudication, factorial/runtime analysis, generation, and provenance |
| `configs/experiments/` | Pinned OLMo protocols, exact historical-runtime reproductions, Tülu endpoint pilot, and 12-point trajectory plan |
| `artifacts/` | Compact checksummed baseline, factorial, and exact-runtime evidence bundles |
| `data/` | Canonical prompts, validation records, the balanced 576-condition design, and frozen free-form/capability anchors |
| `reports/` | Generated statistical reports, methodology audit, runtime audit, and research roadmap |
| `kaggle/` | Private-dataset staging plus restartable T4×2 runners |
| `tests/` | Unit tests and deterministic smoke fixtures |

## Reproduce locally

```bash
python3 -m pip install -e '.[analysis,dev]'
make check
make analyze-factorial
make analyze-runtime-crossover
make anchors
```

`make check` validates prompts, compiles the package and runners, runs the unit and smoke tests, regenerates baseline tables, and verifies tracked artifact checksums. The two analysis targets deterministically regenerate the counterbalanced and exact-runtime reports.

To rebuild the balanced prompt pack and private Kaggle dataset payload:

```bash
make factorial-pack
make kaggle-bundle
```

The primary result files are:

- `reports/factorial_default4bit_v2_results.md`
- `reports/factorial_nf4_v1_results.md`
- `reports/factorial_nf4_vs_default4bit.md`
- `reports/historical_runtime_crossover_v1.md`
- `reports/tulu_endpoint_pilot_v1.md`
- `reports/tulu_trajectory_wave_01_integrity.md`
- `reports/tulu_trajectory_wave_02_integrity.md`
- `reports/ai_semantic_audit_v1.md`
- `data/tulu_cross_format_anchors_v1.manifest.json`
- `reports/methodology_audit.md`
- `reports/research_roadmap.md`

## Limitations

- There are only 24 authored source items from three narrow categories; 1,728 repeated conditions do not create 1,728 independent prompts.
- Deterministic decoding measures one response path per condition.
- The 0/1/2 score is an ordinal, judgment-dependent index, and category cancellation challenges a single aggregate construct.
- The accepted semantic audit is AI-only and non-independent; nine pairs carry explicit construct-fidelity caveats.
- Quantization/runtime changes alter individual responses and position effects, even when marginal stage contrasts are stable.
- The OLMo checkpoints are not a controlled RLVR trajectory, and the results should not be generalized to other lineages or frontier systems.
