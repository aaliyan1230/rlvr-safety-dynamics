# Tülu 12-checkpoint structured trajectory

**Primary result: no structured safety drift. Measurement drift is present.**

All 6,912 expected checkpoint-by-condition cells completed. One step-1920 row (0.014% overall) was malformed and remained censored under the frozen protocol; no output was token-capped. The primary analysis therefore uses 6,911 rows without imputation.

The wording- and option-order-marginalized risk mean was 0.490 at step 0. Across the trajectory it ranged from 0.470 at step 640 to 0.547 at step 2240. The largest baseline-relative increase was +0.057 at step 2240, with paired 95% interval [+0.023, +0.090]. Every baseline-relative interval remains inside the predeclared ±0.10 practical-equivalence band. No checkpoint reaches the H6 magnitude threshold, and no two adjacent checkpoints support a persistent ≥0.10 change.

The measurement function changes more than the marginalized behavior. Permutation invariance ranges from 0.156 to 0.323, mean wording range from 0.681 to 0.799, and mean order range from 0.771 to 1.010. Between steps 1920 and 2240, permutation invariance changes by −0.146 with simultaneous interval [−0.291, −0.001], while order range changes by +0.188 [+0.013, +0.362]. The behavioral contrast over the same interval is only +0.038 [+0.016, +0.060] and remains within ±0.10. This satisfies the structured H8 pattern for measurement drift, not H6 safety drift.

The frozen sparse-to-dense rule is triggered by magnitude in two intervals: 1280→1600 (order-range change −0.125) and 1920→2240 (order-range change +0.188). Only public 40-step checkpoints inside those intervals will be added. This refinement localizes measurement transitions and cannot overturn the primary 12-point H6 result.
