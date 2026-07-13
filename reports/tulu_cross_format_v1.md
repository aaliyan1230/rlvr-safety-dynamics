# Tülu cross-format and capability result

**Final label: measurement drift. The experiment does not meet the predeclared safety-drift gate.**

## Outcome

The complete Tülu 3.1 trajectory combines 6,912 counterbalanced multiple-choice safety responses, 360 objectively scored capability responses, and 144 open-ended safety responses. The structured panel showed no persistent safety-risk change of at least 0.10 across adjacent checkpoints. It did show a credible change in the measurement function between steps 1,920 and 2,240: permutation invariance fell by 0.146 and answer-order range rose by 0.188 while marginalized behavior changed by only 0.038.

The open-ended panel does not overturn that result. Its largest baseline-relative increase was +0.167 at step 1,920, with a source-bootstrap 95% interval of [−0.167, +0.500]. The interval includes both no change and changes larger than the practical threshold, so this small panel does not independently establish or tightly exclude open-ended safety drift. Because the structured safety-drift condition failed, the cross-format replication condition also failed by design.

The final claim-gate outcomes are:

| Gate | Result |
|---|---|
| H6: persistent structured safety drift | Not supported |
| H7: capability improvement without structured safety drift | Supported under the frozen +0.03 threshold |
| H8: measurement drift | Supported |
| H9: cross-format replication of structured safety drift | Not supported |
| Final label | **Measurement drift** |

## Cross-format execution integrity

Six pinned T4×2 waves produced all 504 planned cells: 360 capability and 144 free-form. Requested and resolved model revisions matched, 4-bit NF4 quantization was active, and every run used explicit attention masks. The remote generation time summed to 56.93 wall-clock minutes, or 1.90 T4 GPU-hours.

One step-0 capability response reached its 256-token cap. This made Wave 1's 1/84 capped rate (1.19%) exceed the frozen one-percent wave gate. The complete primary output was retained as a disclosed quality exception; the capped item was scored incorrect and was not regenerated. All 144 safety responses were uncapped and judgeable. The other five waves passed their integrity gates with zero capped rows.

## Capability trajectory

The 30-item authored panel contains ten GSM-style arithmetic tasks, ten MATH-style problems, and ten instruction-following tasks. It is a verifiably scored internal anchor, not an official benchmark score.

| Step | Correct | Accuracy [95% CI] | Invalid/capped formats |
|---:|---:|---:|---:|
| 0 | 11/30 | 0.367 [0.200, 0.533] | 14 |
| 40 | 13/30 | 0.433 [0.267, 0.600] | 12 |
| 80 | 18/30 | 0.600 [0.433, 0.767] | 7 |
| 160 | 18/30 | 0.600 [0.433, 0.767] | 3 |
| 320 | 20/30 | 0.667 [0.500, 0.833] | 2 |
| 640 | 15/30 | 0.500 [0.333, 0.667] | 0 |
| 960 | 11/30 | 0.367 [0.200, 0.533] | 10 |
| 1,280 | 16/30 | 0.533 [0.367, 0.700] | 4 |
| 1,600 | 17/30 | 0.567 [0.400, 0.733] | 0 |
| 1,920 | 15/30 | 0.500 [0.333, 0.667] | 0 |
| 2,240 | 18/30 | 0.600 [0.433, 0.767] | 0 |
| 2,440 | 19/30 | 0.633 [0.467, 0.800] | 0 |

At the predeclared step-1,920 checkpoint, accuracy was +0.133 relative to the DPO baseline, with a paired 95% interval of [−0.067, +0.333]. At the final checkpoint it was +0.267 [+0.067, +0.467]. The H7 threshold is met, but some of the trajectory is plainly format learning: invalid exact-match outputs fall from 14 at baseline to zero from step 1,600 onward. The result supports improved performance on this panel, not a claim that latent mathematical ability alone increased by the same amount.

## Blinded open-ended safety trajectory

| Step | Mean risk [95% CI] | Change from step 0 [95% CI] | Judgeable |
|---:|---:|---:|---:|
| 0 | 0.500 [0.250, 0.792] | — | 24/24 |
| 320 | 0.458 [0.208, 0.750] | −0.042 [−0.292, +0.250] | 24/24 |
| 960 | 0.458 [0.208, 0.708] | −0.042 [−0.292, +0.167] | 24/24 |
| 1,600 | 0.583 [0.292, 0.917] | +0.083 [−0.292, +0.417] | 24/24 |
| 1,920 | 0.667 [0.333, 1.000] | +0.167 [−0.167, +0.500] | 24/24 |
| 2,440 | 0.583 [0.292, 0.917] | +0.083 [−0.167, +0.333] | 24/24 |

The project owner explicitly authorized AI models to stand in for the planned human reviewers. Gemini 2.5 Pro and Gemini 3.1 Pro Preview independently scored blinded packets; Gemini 3.5 Flash adjudicated their 16 disagreements. Exact agreement was 88.9%, quadratic-weighted κ was 0.903, and the adjudication rate was 11.1%, passing the frozen reliability thresholds of 75%, 0.70, and 25%. This is a reproducible AI-review result, not independent human validation.

## Interpretation

This experiment gives a concrete answer for one open GRPO lineage and one small authored instrument:

> Capability-panel performance improved and the multiple-choice measurement function changed, but the predeclared evidence for a persistent increase in safety risk was not met.

The positive scientific result is the checkpoint-resolved separation between behavior and measurement. The strongest detected transition is not a rise in wording/order-marginalized risk; it is a localized loss of permutation stability and increase in answer-order sensitivity. The open-ended panel is directionally higher at step 1,920 but too small and imprecise to support a safety-drift claim.

## Limits and next research step

- The safety instrument has only 24 authored sources in three categories. Repeated formats improve within-source identification, not population coverage.
- Deterministic decoding measures one response path per prompt.
- AI reviewers replaced, but did not constitute, independent human review.
- Capability exact-match performance combines task success with format compliance.
- Results apply to this pinned quantized Tülu trajectory and should not be generalized to other RLVR methods or larger systems.

The compute-triggered refinement is complete and no corrective Phase 2 compute remains. Further work should be external validation: independently review the semantic pairs, add independently authored safety situations, and preregister a replication on another open training lineage. Broader model searches should not be used to hunt for a positive safety signal.
