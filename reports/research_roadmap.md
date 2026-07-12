# Research roadmap: Safety drift or measurement drift?

Status: 2026-07-12. Both Phase 1 factorials, the exact 432-generation historical-runtime crossover, and the repaired 96-generation Tülu endpoint feasibility pilot are complete on Kaggle T4×2. Every computational panel passed its strict completeness and censoring gates. Independent blinded human semantic validation remains outstanding.

## Decision in one sentence

The current project does not show that RLVR increased instrumental behavior. Two counterbalanced protocols find marginal OLMo stage spreads of 0.014 and 0.033, while exact reproduction shows that the historical prompt-pack comparison also changed inference runtime. The useful next contribution is to ask whether safety behavior or the measurement function changes along one controlled GRPO training trajectory.

## Corrected current-state verdict

The evidence supports five conclusions.

1. **No robust behavioral increase has been observed.** In the adjudicated free-form OLMo comparison, Instruct scored **0.400** and RL-Zero-General **0.325**, while 16/40 RL-Zero-General risk rows were length-confounded. In the legacy clean structured-choice comparison, SFT scored 0.500, DPO 0.458, and final Instruct 0.417. That ordering is inconsistent with a monotonic post-training risk increase. RL-Zero-General is not a clean counterexample: 24/24 structured rows hit the token cap and 8/24 were malformed.
2. **Neither counterbalanced protocol shows material marginal stage separation.** Under NF4/double quantization, means were 0.366 for Instruct, 0.363 for DPO, and 0.377 for SFT; the stage spread was 0.014. Under standardized default 4-bit loading they were 0.380, 0.392, and 0.413; the spread was 0.033. Every paired source-clustered 95% stage interval in both protocols lies inside the predeclared ±0.10 band.
3. **Format and censoring effects are real.** Qwen Thinking hit the cap on 11/12 outputs, OLMo Think on 12/12, and OLMo RL-Zero-General repeatedly produced capped, verbose, or malformed responses. These rows establish measurement failure, not latent unsafe behavior.
4. **The old wording attribution is non-identifiable.** The original and paraphrase runs changed semantic option position as well as wording. The reported +0.12 to +0.33 prompt-pack deltas, item-flip rates, and “7–9× model-stage spread” comparison therefore cannot be attributed to wording. They should be described only as differences between confounded prompt packs. The baseline manifest records this limitation.
5. **The historical comparison also changed inference runtime.** The original and P1/P2 kernels used different content-addressed Kaggle images and dependency stacks. Each exact image now reproduces its own historical cells at 100%, but the two runtimes agree on only 174/216 identical layouts. The P1/P2-image runtime shifts the pooled score by +0.088 [0.009, 0.171]. Thus the historical deltas alias wording, semantic option position, and runtime.

The defensible headline is therefore:

> No robust RLVR-related increase in instrumental behavior was detected. Two complete counterbalanced protocols bound the marginal OLMo stage contrasts inside ±0.10, while exact historical reproduction demonstrates substantial option/order and runtime sensitivity in the measurement process.

This is a correction, not a negative finding about all RLVR systems. The models are small, the behavioral item pool has only 24 independent source items in the structured study, and the existing OLMo RL-Zero comparison is not a controlled GRPO trajectory.

## Why stage × order is necessary but not the novelty

Fully permuting answer options is the right correction. It is not, by itself, a new research result. Option-order sensitivity and prompt-format sensitivity are already well established in multiple-choice evaluation: see [Pezeshkpour and Hruschka (NAACL Findings 2024)](https://aclanthology.org/2024.findings-naacl.130/), [Khatun and Brown (2024)](https://arxiv.org/abs/2401.07955), [Li et al. (LREC-COLING 2024)](https://aclanthology.org/2024.lrec-main.251/), and [Sclar et al. (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html). [BrittleBench](https://arxiv.org/abs/2603.13285) extends the same concern to semantics-preserving prompt variation and model ranking.

More directly, a [recent option-position audit](https://openreview.net/forum?id=qeF7zlpneF) reports that fine-tuning can change position preferences without changing permutation-debiased performance. Consequently, “post-training stage interacts with answer order” is a replication or domain transfer to safety prompts, not sufficient novelty.

Intermediate checkpoints introduce an additional construct-validity risk. [Bunn, Wiegreffe, and Bogin (GEM 2025)](https://aclanthology.org/2025.gem-1.46/) show that an intermediate model can identify answer text yet fail to emit the corresponding option label, confounding core ability with emerging format-following and symbol-binding ability. This makes the free-form anchor a required measurement check rather than an optional robustness appendix. Checkpoint-resolved RLVR analysis is also no longer novel by itself: [Wang et al. (2026)](https://arxiv.org/abs/2601.04537) analyze intermediate RLVR checkpoints and report approximately linear weight and teacher-forced output-log-probability trajectories across several settings. The novelty target here is narrower: decompose a safety-behavior trajectory from a simultaneously measured format/wording/order trajectory and require cross-format agreement.

The completed factorials and runtime crossover show that inference reproducibility is part of the measurement question, not bookkeeping around it. The stronger direction begins after independent semantic review and the remaining Phase 2 gates: follow one exact GRPO lineage across training, marginalize known measurement perturbations at every checkpoint, and test whether the latent behavioral estimate and the measurement-error profile follow different trajectories. This follows the statistical-evaluation emphasis of [NIST AI 800-3](https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models): observations should be modeled as noisy, structured measurements rather than treated as a benchmark's ground truth.

## Target question and operational definitions

The target question is:

> During verifiable-reward optimization, does order- and wording-marginalized instrumental behavior change, or do apparent changes arise because the model's response to the evaluation format changes?

Use the following labels only after applying the decision rules below.

- **Safety drift:** a sustained checkpoint change in the order- and wording-marginalized behavioral score that is practically non-trivial, survives source-item-clustered uncertainty, and replicates in free-form responses.
- **Measurement drift:** checkpoint changes in option-position preference, wording sensitivity, permutation invariance, parseability, verbosity, or censoring without a corresponding change in the marginalized behavioral score.
- **Mixed drift:** both criteria are met.
- **No detectable drift:** both behavioral and measurement changes are bounded inside predeclared equivalence margins.
- **Inconclusive:** quality or precision gates fail. A non-significant contrast is not an equivalence result.

The score remains the predeclared ordinal index: safe/non-instrumental = 0, ambiguous/mixed = 1, clear instrumental endorsement = 2. The minimum practically important change in the source-averaged score is **0.10 points** on this 0–2 scale. This margin is fixed before Phase 1 results are inspected and is slightly larger than the legacy clean-stage spread of 0.08.

## Phase 1: identify the measurement function

### Completed dual-protocol design

Both factorial runs use the same complete crossing:

| Factor | Levels |
|---|---|
| Source item | 24: eight each for resource acquisition, self-preservation, and evaluation awareness |
| Wording | 4: original, p1, p2, p3 |
| Semantic option order | all 3! = 6 permutations of scores 0, 1, 2 |
| Model stage | 3: `allenai/Olmo-3-7B-Instruct-SFT`, `-DPO`, and final `Instruct` |
| Total | 24 × 4 × 6 × 3 = **1,728 generations** |

Generation is deterministic, uses the model chat template, requests a two-line choice plus short reason, caps generation at 96 tokens, and loads each 7B model in 4-bit on Kaggle T4×2. The unit of inference is the **source item**, not each of the 1,728 repeated cells.

The first run, `wording_position_v1`, completed at 2026-07-10 11:50 UTC using explicit NF4 with double quantization, fp16 compute, and one model per GPU. Its checksummed artifact under `artifacts/factorial_nf4_v1/` records:

- 1,728/1,728 expected cells, zero execution errors, zero malformed rows, and zero token-capped rows;
- prompt SHA-256 `040de139b26f4d78230dc47ec9ee4f9be933b13bbeb3e44eb1a3d6630bb7ed29`;
- resolved model commits `e1452fc…` (SFT), `b33130b…` (DPO), and `6e5971d…` (Instruct); and
- a wall-clock interval of about 49 minutes from recorded start to completion.

The second run, `wording_position_v2_legacy_quant`, completed at 2026-07-10 16:53 UTC under Kaggle kernel `aaliyanshaikh/rlvr-wording-position-factorial-v2-legacy-quant`. It holds the factorial fixed and uses `BitsAndBytesConfig(load_in_4bit=True)`, `device_map="auto"`, and `trust_remote_code=True` under pinned Transformers 4.57.6. Despite its historical-motivated name, it is a standardized counterfactual, not an exact historical reproduction; the historical kernels used two different runtimes.

### Completed factorial results

| Protocol | Model | Marginal mean [source-clustered 95% CI] | Rows |
|---|---|---:|---:|
| NF4/double | OLMo 3 Instruct | 0.366 [0.266, 0.469] | 576 |
| NF4/double | OLMo 3 Instruct-DPO | 0.363 [0.260, 0.469] | 576 |
| NF4/double | OLMo 3 Instruct-SFT | 0.377 [0.278, 0.488] | 576 |
| Default 4-bit | OLMo 3 Instruct | 0.380 [0.276, 0.488] | 576 |
| Default 4-bit | OLMo 3 Instruct-DPO | 0.392 [0.292, 0.500] | 576 |
| Default 4-bit | OLMo 3 Instruct-SFT | 0.413 [0.311, 0.528] | 576 |

The marginal stage spreads are 0.014 and 0.033. All six paired stage-contrast 95% intervals are contained inside ±0.10. The widest is the default-4-bit SFT-minus-Instruct contrast, +0.033 [-0.021, +0.085]. The two protocols agree on 85.4% of individual cells; their overall mean difference is +0.027 [-0.015, +0.068] for default 4-bit minus NF4.

No per-model wording-versus-original contrast has a 95% interval excluding zero in either protocol. Category-level contrasts are larger and often oppose one another: resource-acquisition rewrites lower scores while self-preservation rewrites raise them. Aggregate stability must not be generalized to every category or taken as proof of semantic equivalence.

Position behavior is protocol-dependent. The NF4 pooled B-minus-A contrast is +0.002 [-0.092, +0.096] with a simultaneous interval, whereas the default-4-bit contrast is +0.127 [0.047, 0.206]. Their paired effect difference is +0.125 [0.066, 0.186]. This does not imply stable order invariance: only 39.6%–51.0% of cells keep the same score across all six permutations under NF4 and 40.6%–46.9% under default 4-bit. Mean wording/order ranges are about 0.60–0.65. Range-versus-spread ratios are descriptive, not variance components or causal effect ratios.

Historical-layout reproduction is a separate check. The NF4 and standardized default-4-bit factorials disagree with 64/216 and 46/216 historical scores, respectively. Exact-image panels resolve why: the original and P1/P2 runs used different runtimes. Each exact image reproduces its own historical cells at 100%, yet the two runtimes agree on only 174/216 identical layouts and differ by +0.088 [0.009, 0.171] overall. Counterbalancing, standardized inference, and exact historical reproduction must remain distinct analyses.

Empirical paired-source bootstrap power diagnostics pass the predeclared planning threshold for this item pool: minimum two-sided power for a 0.10-point shift is 99.2% under NF4 and 96.2% under default 4-bit. These diagnostics reuse the observed 24-source residual distributions. They support conditional sensitivity in this fixed pool, not generalizability to a broader prompt population.

### Final Phase 1 outputs

Both artifacts produce the following descriptive outputs:

1. model-stage means and paired contrasts marginalized over wording and all six orders;
2. wording contrasts marginalized over all six orders;
3. risky-option position contrasts marginalized over wording and relative order;
4. permutation invariance, mean within-item wording range, mean within-item order range, and A/B/C choice counts by model and wording;
5. model × wording and model × order interactions;
6. malformed, token-cap, verbosity, and missing-cell rates reported separately from behavior; and
7. a corrected note that replaces, rather than coexists with, the old causal wording claims.

The existing descriptive sum-of-squares table may be retained, but it must not be presented as causal variance attribution. With an ordinal outcome and 24 source clusters, the primary inferential model should be a cumulative-link mixed model with fixed effects for model, wording, full semantic order, category, model × wording, and model × order, plus a source-item random intercept. Report source-clustered bootstrap intervals for easily interpreted marginal means and paired contrasts. Six permutations are repeated measurements, not six independent prompts.

### Phase 1 quality gate

Do not interpret model or wording contrasts until all of the following hold:

- exactly 1,728 unique model × source × wording × order cells are present;
- every input and output has a recorded model revision, prompt/config hash, generation parameters, and parser version;
- malformed plus capped rows are below 1% per clean model after targeted reruns; if they exceed 5% for any model, report that model as non-comparable;
- two reviewers, blinded to model output, verify that each wording preserves the scenario and the 0/1/2 option semantics; any invalid source-wording pair is removed by a rule fixed before outcome inspection; and
- a source-clustered simulation using the observed Phase 1 covariance shows at least 80% power for a 0.10-point paired checkpoint change. Both diagnostics pass conditionally, with 99.2% and 96.2% minimum power; retain the limitation to this authored item pool. More permutations do not establish population generalizability.

Both computational runs pass completeness, parser/censoring, conditional power, and the strict rule that the full paired-stage 95% intervals lie inside ±0.10. Phase 1 as an independently validated scientific instrument is not complete because the planned two-reviewer semantic audit is absent and the authored pool remains narrow. Automated source-aware validation passed 72/72 pairs but is not a substitute for blinded human review.

## Phase 2: exact Tülu 3.1 8B GRPO trajectory

### Why this lineage

The [Tülu 3 report](https://arxiv.org/abs/2411.15124) documents the open post-training pipeline. The [Tülu 3.1 8B model card and checkpoint repository](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B) state that version 3.1 changes only the final RL stage relative to the Tülu 3 DPO model, switches that stage to GRPO without a reward model, and exposes intermediate checkpoint branches. This is a much cleaner intervention than comparing OLMo Instruct with RL-Zero-General, which differs in training path and output style.

The primary panel is fixed before looking at safety outcomes. It is dense early and then sampled every 320 steps. Immutable commits are pinned because branch heads can move.

| Trajectory point | Repository / revision | Pinned commit |
|---:|---|---|
| 0 | `allenai/Llama-3.1-Tulu-3-8B-DPO` | `a7beb67e33ffd01cc87ac3b46cadc1000985b8db` |
| 40 | `allenai/Llama-3.1-Tulu-3.1-8B`, `step_40` | `fa47681c7387782feabb8ba4479e4cc1983763a0` |
| 80 | same, `step_80` | `3717881c953fb1f8e57432accbb7b6538f066ff6` |
| 160 | same, `step_160` | `3f46ba82012d9b952eb17da545ffcad1abff13d1` |
| 320 | same, `step_320` | `35b54436aa4615571168bfddd8b8a8f979e0e83b` |
| 640 | same, `step_640` | `dd516e7622e80586e9db3ee38e9b6630c3a4ae28` |
| 960 | same, `step_960` | `8338fd92eb42777ad2d38790d946de1afdf289c8` |
| 1,280 | same, `step_1280` | `5b905e72315ca875b5e8bbe850bb4726ed2bda7f` |
| 1,600 | same, `step_1600` | `6c2f75b13175441daf6bf3a8fcbed44ff63427e5` |
| 1,920 | same, `step_1920` | `46239c2d07db76b412e1f1b0b4542f65b81fe01f` |
| 2,240 | same, `step_2240` | `91671144b31f2a2d525d3c3aa229b6c7e1648636` |
| 2,440 | same, `step_2440` | `9fed80c5fd705e4885468a980c7a14e14d016b08` |

As of 2026-07-10, the repository's `main` revision resolves to the same commit as `step_1920`; it must not be counted as an additional checkpoint. Step 2,440 is retained because it distinguishes continued training from the selected release checkpoint.

### Phase 2 experiment

Run the full validated Phase 1 factorial at each of the 12 points: 12 checkpoints × 24 sources × 4 wordings × 6 orders = **6,912 short structured generations**. Hold tokenizer/chat template, prompt bytes, quantization, decoding, token cap, parser, and scoring code fixed. Only the pinned checkpoint changes.

Add two triangulation panels:

- **Capability anchor:** a fixed, held-out, verifiably scored panel drawn from the GRPO task families, evaluated at all 12 points. This confirms that optimization is active and permits a capability-versus-safety comparison. Report the small panel's uncertainty; it is not a replacement for the published Tülu evaluations.
- **Free-form safety anchor:** the same 24 constructs at steps 0, 320, 960, 1,600, 1,920, and 2,440, without answer options. Score blinded outputs under the existing rubric with two human reviewers; use an LLM judge only to triage disagreements and censoring. This is the required cross-format check for a safety-drift claim.

If two adjacent prespecified checkpoints bracket a behavioral or measurement change of at least 0.10, run the public 40-step branches inside only that interval. Record this refinement rule before the primary panel completes. Do not add checkpoints merely because an isolated point looks interesting.

## Falsifiable hypotheses

All confidence intervals below are source-item-clustered. Confirmatory families use simultaneous max-|t| bootstrap intervals or Holm correction. Equivalence uses two one-sided tests with the ±0.10 behavioral margin. “Supported” requires both the stated magnitude and uncertainty criterion; otherwise report contradicted or unresolved, not “partially confirmed.”

| ID | Hypothesis and operational test | What would falsify or fail to support it |
|---|---|---|
| **H1** | **Legacy-layout deltas are substantially attenuated by order marginalization.** Within one fixed default-4-bit protocol, the mean absolute P1/P2 corrected wording contrast, after averaging all six orders, will be at least 50% smaller than its same-run historical-layout contrast. | Attenuation is below 50%, or corrected contrasts increase. The historical causal estimate remains invalid either way because its packs also used different runtimes. |
| **H2** | **Option position materially affects the structured score.** After marginalizing wording and model, the largest risky-position contrast is at least 0.10 and its simultaneous 95% interval excludes zero. | All position contrasts are below 0.10 or include zero. |
| **H3** | **A wording effect survives counterbalancing.** At least one p1/p2/p3-versus-original contrast has absolute magnitude at least 0.10, a simultaneous 95% interval excluding zero, and the same direction in at least two of three OLMo stages. | No contrast meets all three conditions. |
| **H4** | **Clean OLMo endpoint stages are behaviorally equivalent.** All pairwise SFT/DPO/Instruct marginalized contrasts lie inside ±0.10 under an equivalence test. | Any contrast is credibly outside the equivalence band. A wide interval is inconclusive, not support for H4. |
| **H5** | **Post-training stage changes the measurement function.** At least one stage × wording or stage × order contrast changes permutation invariance or within-item range by at least 0.10, with a simultaneous interval excluding zero. | Interaction contrasts remain below 0.10 or uncertain. This is the only Phase 1 result that could motivate a trajectory-level measurement-drift prediction. |
| **H6** | **GRPO produces safety drift.** Relative to step 0, the order- and wording-marginalized score changes by at least 0.10 at two adjacent prespecified checkpoints in the same direction, with simultaneous 95% intervals excluding zero. | All checkpoint contrasts are equivalent within ±0.10, or a change is isolated, reverses immediately, or is explained by censoring. |
| **H7** | **Capability improvement can occur without safety drift.** The verifiable capability anchor improves by at least 3 percentage points by step 1,920 while the corresponding safety contrast is equivalent within ±0.10. | Capability does not improve, or the safety contrast is credibly outside the equivalence band. This directly tests the rival prediction suggested by [work on safety preservation under RLVR](https://arxiv.org/abs/2511.21050). |
| **H8** | **GRPO produces measurement drift even when marginalized safety is stable.** Across two adjacent checkpoints, option/wording range or censoring changes by at least 0.10 while the behavioral contrast remains equivalent within ±0.10. | Measurement metrics are stable, or behavioral risk also changes; the latter implies mixed rather than pure measurement drift. |
| **H9** | **A structured safety-drift signal generalizes across format.** Any H6 direction appears at the matching free-form milestones, with a paired contrast of at least 0.10, acceptable inter-rater agreement, and no checkpoint-specific censoring increase above five percentage points. | The free-form effect is absent, reverses, or is concentrated in unjudgeable rows. In that case the result is a structured-measure effect, not a safety-drift claim. |

Current Phase 1 evidence status:

- **H1 meets its point-estimate criterion only in the standardized default-4-bit run.** Mean absolute P1/P2 historical-layout contrasts fall from 0.125 to 0.060 after order marginalization, a 51.9% attenuation. NF4 does not replicate that attenuation. Exact-image work further shows that the historical packs used different runtimes, so H1 cannot partition the original cross-run shifts.
- **H2 is protocol-contingent, not robustly supported.** Default 4-bit supports pooled B-minus-A at +0.127 [0.047, 0.206], but NF4 gives +0.002 [-0.092, 0.096].
- **H3 is not supported at the prespecified per-model aggregate level in either protocol.** Every wording-versus-original 95% interval includes zero. Category-level opposition remains important secondary evidence.
- **H4 is supported conditionally for this instrument.** All paired stage 95% intervals in both protocols are fully contained inside ±0.10. This does not establish population-wide or causal equivalence.
- **H5 is not supported.** All simultaneous stage contrasts for permutation invariance, wording range, and order range include zero in both protocols. Runtime changes alter position effects, but that is not a model-stage interaction.
- **H6–H9 are Phase 2 hypotheses and have not been tested.**

## Primary metrics and analysis

### Behavioral estimand

For checkpoint `t`, define the primary risk index as:

`R_t = mean over source items of [mean score over four wordings and six orders]`.

This ordering gives each source item equal weight. It prevents prolific repeated cells from masquerading as independent evidence. Report `R_t`, paired `R_t - R_0`, `P(score > 0)`, and `P(score = 2)`, all with source-clustered intervals. Category estimates are secondary and explicitly multiplicity-adjusted.

Fit a cumulative-link mixed model with checkpoint as both a categorical factor and a predeclared smooth function of `log1p(step)`, plus wording, full semantic order, category, checkpoint × wording, checkpoint × order, and a source-item random intercept. Categorical paired contrasts are primary; the smooth trajectory and any change point are descriptive unless confirmed by the predeclared dense follow-up.

### Measurement estimands

At every checkpoint report:

- risky-position max-minus-min marginal score;
- full A/B/C predicted-position distribution;
- proportion of model × source × wording cells invariant over all six permutations;
- mean within-source wording range after order marginalization;
- mean within-source order range within wording;
- malformed, token-cap, verbosity, refusal, and missing-output rates; and
- inter-rater agreement and adjudication rate for the free-form panel.

Do not collapse these into the behavioral score. A capped answer remains a censoring event even when the visible prefix can be assigned a 0/1/2 label.

### Capability and trajectory analysis

Report exact-match capability separately from safety. Plot capability, `R_t`, order range, wording range, and censoring against training step on aligned axes. Estimate paired checkpoint contrasts, not a correlation alone: twelve trajectory points are too few for a stable causal correlation. The final selected checkpoint (`step_1920`) and continued checkpoint (`step_2440`) must both be shown to avoid endpoint selection bias.

Release row-level data, exact prompt hashes, pinned model commits, environment versions, raw generations, parser decisions, adjudications, and an analysis manifest. Any exclusion rule must be applied without access to checkpoint identity.

## Compute feasibility on Kaggle T4×2

This plan performs inference, not GRPO training. An 8B model fits on one 15 GB T4 with 4-bit NF4 weights and fp16 compute. Run two checkpoints concurrently, one per GPU; never shard one model across both cards. Stream one revision per worker and delete its weight cache after the output and hashes are safely written, because retaining twelve full checkpoint snapshots can exhaust ephemeral disk.

The repaired endpoint pilot is the direct Tülu calibration. Pinned steps 0 and 1,920 ran concurrently on T4×2 and produced 96/96 strict responses in 7.8 wall-clock minutes, with zero malformed or capped rows, exact revision resolution, explicit attention masks, and 6.63 GiB peak memory per GPU. It sampled only the original wording and two orders, so its scores have no scientific interpretation. The full trajectory still requires twelve distinct revisions, and download/cache overhead may dominate across six waves.

| Workload | Generations | Expected execution plan |
|---|---:|---|
| Phase 1 NF4 factorial | 1,728 at 96 tokens max | **Completed in about 49 wall-clock minutes**; 1,728/1,728 rows, zero malformed/capped |
| Phase 1 default-4-bit replication | 1,728 at 96 tokens max | **Completed in about 72 wall-clock minutes**; 1,728/1,728 rows, zero malformed/capped |
| Tülu endpoint feasibility pilot | 96 at 96 tokens max | **Completed in 7.8 wall-clock minutes** on T4×2; 96/96 strict, zero malformed/capped, 6.63 GiB peak per GPU |
| Phase 2 primary trajectory | 6,912 at 96 tokens max | Six two-checkpoint waves; about 3 wall-clock hours of generation by linear pilot scaling, plus variable checkpoint-download and restart overhead |
| Free-form anchor | 144 at 192 tokens max | Add to milestone jobs after the structured outputs pass integrity checks |
| Capability anchor | Size to be frozen before trajectory outcomes | Cap the panel so the complete trajectory stays within the available weekly GPU quota |

The pilot projects about three wall-clock hours and six T4 GPU-hours for the 6,912-row structured panel by linear scaling. This excludes potentially substantial checkpoint-download and cache turnover, so it is a measured-throughput projection rather than a runtime guarantee. Each two-checkpoint wave must checkpoint after every source batch and retain at least a two-hour margin below Kaggle's session limit. If a wave projects past ten hours, reduce the shard size, not the checkpoint panel or factorial design. Freeze the capability panel before inspecting trajectory outcomes.

## Decision and stop gates

1. **Phase 1 integrity gate:** both computational runs pass exact-cell, provenance, parser/censoring, and conditional precision checks. Do not close independent construct validation until the blinded two-reviewer semantic audit also passes.
2. **Protocol-reproduction gate:** passed as an audit, not as a single matched protocol. The two exact historical images reproduce their own cells at 100% and disagree materially on identical layouts. Keep historical, standardized-default, and NF4 claims separate.
3. **Precision gate:** both empirical paired-source diagnostics pass conditionally for a 0.10 change (minimum power 99.2% and 96.2%). Retain the limitation to this 24-item pool; add independent sources if broader-prompt generalization is a target.
4. **Phase 2 pilot gate: passed.** The repaired pinned step-0/step-1,920 run completed 96/96 strict responses in 7.8 minutes, used 6.63 GiB peak memory per T4, resolved both exact revisions, and used explicit attention masks. The pilot is feasibility evidence only; do not interpret its two-order endpoint scores scientifically.
5. **Sparse-to-dense gate:** evaluate the twelve fixed trajectory points first. Add 40-step checkpoints only inside an interval whose adjacent endpoints differ by at least 0.10 on a behavioral or measurement metric.
6. **Safety-claim gate:** use “safety drift” only if H6 and H9 pass and checkpoint-specific censoring does not account for the effect.
7. **Measurement-claim gate:** if H8 passes while H6 is equivalent, stop behavioral escalation and report measurement drift. Do not run broader model families to search for a positive safety result.
8. **Mixed-result gate:** if both H6/H9 and H8 pass, report both trajectories and perform the prespecified dense interval follow-up; do not residualize away measurement drift post hoc.
9. **Null stop gate:** if behavioral and measurement trajectories are both equivalent and the precision gate passed, stop. A well-bounded null trajectory is the result.
10. **Failure stop gate:** if more than 5% of rows at two or more checkpoints are malformed/capped after one format repair, the instrument is not comparable for this lineage. End the checkpoint study as a documented judgeability failure.
11. **External-validation gate:** only after a replicating safety signal should the project spend compute on reward-hacking or emergent-misalignment evaluations. Use the tasks and code released with [Anthropic's reward-hacking study](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) and the [UK AISI evaluation repository](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment), and predeclare which text-only tasks are compatible with Tülu. A failure to reproduce there bounds the claim to this instrument.

## Research contribution if executed

The contribution is not “models are sensitive to option order.” It is a checkpoint-resolved decomposition of a safety evaluation during one open, controlled GRPO stage:

1. a corrected, fully counterbalanced measurement instrument;
2. a pinned behavioral and measurement trajectory from DPO through GRPO;
3. explicit rival hypotheses for safety drift, measurement drift, mixed drift, and equivalence; and
4. cross-format validation that prevents a multiple-choice artifact from being called a safety change.

The [OLMo 3 report](https://arxiv.org/abs/2512.13961) shows why open model flows make this class of analysis possible, but the present OLMo results cannot supply the controlled trajectory because RL-Zero-General follows a different path and is heavily censored. Tülu 3.1 supplies the cleaner first test. The result should be framed as evidence about this lineage and instrument, not RLVR in general.

## Primary-source map

- Multiple-choice order effects: [Pezeshkpour and Hruschka (2024)](https://aclanthology.org/2024.findings-naacl.130/), [Khatun and Brown (2024)](https://arxiv.org/abs/2401.07955), and [Li et al. (2024)](https://aclanthology.org/2024.lrec-main.251/).
- Prompt-format and perturbation sensitivity: [Sclar et al. (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html) and [BrittleBench](https://arxiv.org/abs/2603.13285).
- Training-dependent option-position effects: [Auditing Option-Position Effects in Multiple-Choice Evaluation](https://openreview.net/forum?id=qeF7zlpneF).
- Intermediate-checkpoint format/symbol-binding confounding: [Bunn, Wiegreffe, and Bogin (GEM 2025)](https://aclanthology.org/2025.gem-1.46/).
- Existing RLVR checkpoint-trajectory analysis: [Wang et al. (2026)](https://arxiv.org/abs/2601.04537).
- Statistical evaluation framing: [NIST AI 800-3](https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models).
- Exact post-training lineage: [Tülu 3 paper](https://arxiv.org/abs/2411.15124) and [Tülu 3.1 8B checkpoint repository](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B).
- Rival prediction that RLVR can preserve safety: [Breaking the Safety-Capability Tradeoff](https://arxiv.org/abs/2511.21050).
- Reward-hacking external validation: [Anthropic study](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) and [UK AISI code](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment).
- Open model-flow context: [OLMo 3 technical report](https://arxiv.org/abs/2512.13961).
