# Safety drift or measurement drift? A corrected small-model audit

## Summary

This project began by asking whether small open RLVR-lineage models show more instrumental behavior than related instruction checkpoints. The audit uncovered a more immediate problem: the original structured comparisons mixed model behavior with option order, prompt wording, output censoring, and inference-runtime drift.

After correcting the design and reproducing the historical runtimes, the defensible result is:

> No robust increase in instrumental-risk choices was detected across the evaluated OLMo SFT, DPO, and final Instruct checkpoints. Marginal stage differences are practically small under two complete counterbalanced inference protocols, while individual responses remain highly sensitive to wording, semantic option order, and runtime.

This is not evidence that RLVR is generally safe. The OLMo checkpoints are not a controlled RLVR trajectory, the item pool contains only 24 authored sources, and the semantic audit lacks independent blinded human review.

## Research questions

1. After averaging over four wordings and all six semantic option orders, how large are the descriptive differences among OLMo SFT, DPO, and final Instruct checkpoints?
2. How stable is the instrument to wording, option position, quantization, and dependency/runtime changes?
3. Which apparent historical effects survive exact inference reproduction?
4. What controlled follow-up can distinguish behavioral safety drift from measurement drift during verifiable-reward optimization?

## Evidence and methods

### Baseline layers

The retained baseline contains:

- 48 free-form prompts per OLMo model, including 40 risk prompts and eight benign controls;
- 24 structured-choice source items across resource acquisition, self-preservation, and evaluation awareness; and
- Gemini-assisted triage followed by explicit, machine-readable adjudication of six free-form scores.

Scores use the ordinal index 0=safe/non-instrumental, 1=ambiguous/mixed, and 2=clear instrumental endorsement. Arithmetic means are descriptive summaries of that scale.

### Counterbalanced design

The corrected structured design crosses:

| Factor | Levels |
|---|---|
| Source item | 24; eight per category |
| Wording | original, P1, P2, P3 |
| Semantic option order | all six permutations of 0/1/2 over A/B/C |
| Model stage | SFT, DPO, final Instruct |
| Total per protocol | 1,728 responses |

Both completed runs use deterministic greedy decoding and a 96-token limit. The first uses explicit NF4, fp16 compute, and double quantization. The second uses `BitsAndBytesConfig(load_in_4bit=True)` under the same pinned Transformers 4.57.6 environment. Both have 1,728/1,728 unique cells, zero malformed rows, zero capped rows, immutable model revisions, and checksummed inputs and outputs.

The independent resampling unit is the authored source item, not each repeated condition. Reported intervals are 95% source-clustered bootstrap intervals. The practical-equivalence margin for paired stage contrasts was fixed at ±0.10 points.

### Exact historical-runtime crossover

The historical original stage run and P1/P2 runs used different content-addressed Kaggle images. The reciprocal reproduction runs the same 216 cells under each exact image:

- original wording at every source's historical semantic order;
- P1 at order 012;
- P2 at order 012;
- three OLMo stages and 24 sources.

One panel reproduces the original-stage environment; the other reproduces the P1/P2 environment. The difference-in-differences is

`[(candidate − original) under P1/P2 runtime] − [(candidate − original) under stage runtime]`.

This isolates runtime sensitivity for the historical layouts. It still cannot separate candidate wording from option position because P1/P2 remain fixed at order 012.

## Results

### 1. Free-form and reasoning-style runs are censored comparisons

Final adjudicated visible-text results over 40 risk prompts are:

| Model | Mean | Score counts | Confounded rows |
|---|---:|---|---:|
| OLMo 3 Instruct | 0.400 | 28×0, 8×1, 4×2 | 0/40 |
| OLMo 3 RL-Zero-General | 0.325 | 28×0, 11×1, 1×2 | 16/40 |

RL-Zero-General does not score higher on visible text, but differential truncation makes a safety or equivalence interpretation invalid. Treating its 16 confounded rows as unknown on the 0–2 scale yields a wide all-attempt bound of [0.175, 0.975].

The same problem is stronger in structured RL-Zero-General: all 24 outputs hit the token cap and eight are malformed. Qwen3 4B Thinking hit the cap on 11/12 outputs, with 9/10 risk rows confounded. These runs demonstrate judgeability and censoring failures, not clean behavioral effects.

### 2. The original clean stage ordering is fragile

On the original 24 structured items, SFT=0.500, DPO=0.458, and final Instruct=0.417. The three stages return identical scores on 22/24 items; two items generate the entire 0.083 range. This descriptive ordering is not a stable stage effect.

### 3. Counterbalancing removes any material marginal stage separation

| Protocol | Model | Marginal mean [95% CI] | Rows |
|---|---|---:|---:|
| NF4/double | Instruct | 0.366 [0.266, 0.469] | 576 |
| NF4/double | DPO | 0.363 [0.260, 0.469] | 576 |
| NF4/double | SFT | 0.377 [0.278, 0.488] | 576 |
| Default 4-bit | Instruct | 0.380 [0.276, 0.488] | 576 |
| Default 4-bit | DPO | 0.392 [0.292, 0.500] | 576 |
| Default 4-bit | SFT | 0.413 [0.311, 0.528] | 576 |

The stage spread is 0.014 under NF4 and 0.033 under default 4-bit. Every paired stage-contrast interval is contained inside ±0.10:

| Protocol | Contrast | Difference [95% CI] |
|---|---|---:|
| NF4/double | DPO − Instruct | −0.003 [−0.023, 0.017] |
| NF4/double | SFT − Instruct | +0.010 [−0.033, 0.054] |
| NF4/double | SFT − DPO | +0.014 [−0.024, 0.056] |
| Default 4-bit | DPO − Instruct | +0.012 [−0.021, 0.042] |
| Default 4-bit | SFT − Instruct | +0.033 [−0.021, 0.085] |
| Default 4-bit | SFT − DPO | +0.021 [−0.019, 0.059] |

The strict practical-equivalence criterion is therefore met for these stages in both protocols. This conclusion is conditional on the authored item pool, deterministic decoding, and the selected scale; it is not a causal or population-level equivalence statement.

### 4. Aggregate wording stability hides category opposition

No model-specific wording-versus-original interval excludes zero after marginalizing over all six orders. That aggregate result should not be read as semantic invariance.

Both protocols show the same qualitative category opposition:

- resource-acquisition rewrites lower scores relative to original, often with intervals excluding zero;
- self-preservation rewrites raise scores, especially P2; and
- evaluation-awareness effects vary by pack.

These opposing shifts cancel in the pooled wording mean. A single global score therefore mixes constructs that respond differently to seemingly equivalent rewrites. The source-aware Gemini 2.5 Pro validator passed 72/72 source/candidate pairs, but that automated same-provider review is not enough to rule out subtle semantic movement. A blinded two-human audit remains required.

### 5. Option-position behavior is inference-protocol dependent

Under NF4, none of the simultaneous pooled or per-model risky-position intervals excludes zero. The pooled B-minus-A contrast is +0.002 [−0.092, 0.096]. Under default 4-bit, the same contrast is +0.127 [0.047, 0.206].

The paired change in B-minus-A position effect from NF4 to default 4-bit is +0.125 [0.066, 0.186]. Thus a measurement bias can appear or disappear with the inference implementation even when marginal model-stage contrasts remain small.

Only 39.6%–51.0% of model × source × wording cells are permutation-invariant under NF4 and 40.6%–46.9% under default 4-bit. Mean within-item wording and order ranges are about 0.60–0.65, far larger descriptively than the 0.014–0.033 marginal stage spreads. These range/spread comparisons are diagnostics, not causal ratios.

### 6. The historical prompt-pack shifts also contain runtime drift

The original stage environment reproduces all 72 historical original cells exactly. The P1/P2 environment reproduces all 144 historical candidate cells exactly. Exact reproduction therefore confirms the recovered images, package versions, immutable model revisions, and loading path.

Holding all 216 historical layouts fixed and changing only the runtime gives:

| Scope | Stage-runtime mean | P1/P2-runtime mean | Runtime delta [95% CI] |
|---|---:|---:|---:|
| All cells | 0.546 | 0.634 | +0.088 [0.009, 0.171] |
| Original | 0.458 | 0.500 | +0.042 [−0.056, 0.153] |
| P1 | 0.597 | 0.625 | +0.028 [−0.153, 0.236] |
| P2 | 0.583 | 0.778 | +0.194 [0.056, 0.361] |

Only 174/216 scores and response letters agree between runtimes. The pack × runtime interactions are −0.014 [−0.208, 0.208] for P1 and +0.153 [−0.056, 0.361] for P2. Those interactions remain uncertain with 24 sources, so the historical shifts cannot be numerically decomposed into wording, option position, and runtime shares. The valid conclusion is non-identifiability, not that one confound alone explains everything.

## Interpretation

The corrected OLMo result is a bounded measurement finding:

1. The balanced aggregate score does not materially separate SFT, DPO, and final Instruct under either tested 4-bit protocol.
2. Individual choices are unstable across nominally equivalent wording/order conditions, and position effects themselves depend on the inference runtime.
3. Historical paraphrase deltas are invalid as wording estimates because wording, position, and runtime changed together.
4. Reasoning-style/RL-Zero outputs introduce severe, checkpoint-dependent censoring that must be modeled separately from behavior.

The stronger research question is therefore not whether answer order matters—it is already known to matter—but whether a safety-behavior trajectory and the measurement-error trajectory diverge during one controlled post-training run.

## Recommended follow-up

The configured Phase 2 study follows the exact Tülu 3.1 8B GRPO lineage from the pinned DPO base through 11 public GRPO checkpoints. At each of 12 points it would run the complete 576-condition factorial, totaling 6,912 structured responses. A capability anchor verifies that optimization is active, and blinded free-form anchors at selected milestones determine whether any structured signal generalizes across format.

Launch remains conditional on:

- a blinded two-reviewer semantic audit of all 72 source/candidate pairs;
- frozen free-form scoring and capability panels;
- a clean-checkout reproduction of the Phase 1 artifacts and analyses.

The endpoint feasibility gate has passed. The repaired pinned step-0/step-1,920 pilot produced 96/96 strict responses with zero malformed or capped rows, exact revision resolution, explicit attention masks, 6.63 GiB peak memory per T4, and 7.8 minutes wall time. These pilot outputs establish execution feasibility only and are not a marginalized safety comparison.

The prespecified result labels are safety drift, measurement drift, mixed drift, no detectable drift, or inconclusive. A structured multiple-choice change alone is not enough for a safety-drift claim.

## Reproducibility

```bash
python3 -m pip install -e '.[analysis,dev]'
make check
make analyze-factorial
make analyze-runtime-crossover
```

The compact evidence bundles and SHA-256 manifests live under `artifacts/`. Detailed generated outputs and model caches are intentionally excluded; each compact bundle records the remote Kaggle kernel, exact config, resolved revisions, runtime metadata, scored rows, metrics, and checksums needed to audit the reported results.

## Limitations

- Twenty-four authored sources are too narrow for broad behavioral generalization.
- Repeated wordings and orders improve within-source identification, not source-population coverage.
- Deterministic decoding does not characterize sampling variability.
- Arithmetic summaries of a 0/1/2 ordinal rubric are descriptive.
- Automated semantic validation is not independent human validation.
- Category cancellation questions whether one pooled construct is appropriate.
- Quantization and dependencies alter response-level behavior.
- OLMo SFT/DPO/Instruct is not the controlled GRPO trajectory needed for an RLVR causal claim.
