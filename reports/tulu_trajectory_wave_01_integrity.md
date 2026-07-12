# Tülu trajectory wave 01 integrity report

**Gate result: PASS. Scientific interpretation: withheld until the complete trajectory panel is available.**

Wave 01 evaluated the pinned DPO base at step 0 and the pinned GRPO checkpoint at step 40 across the full 576-condition structured panel per checkpoint.

| Checkpoint | Rows | Strict parses | Malformed | Capped | Peak GPU memory | Exact revision |
|---|---:|---:|---:|---:|---:|---|
| step 0 DPO | 576 | 576 | 0 | 0 | 6.631 GiB | yes |
| step 40 GRPO | 576 | 576 | 0 | 0 | 6.631 GiB | yes |

The accepted run produced 1,152/1,152 unique model-by-condition cells in 35.39 wall-clock minutes on Tesla T4×2. It used deterministic decoding, NF4/fp16/double quantization, explicit all-ones attention masks, and prompt SHA-256 `040de139b26f4d78230dc47ec9ee4f9be933b13bbeb3e44eb1a3d6630bb7ed29`.

Kernel version 1 failed before producing any generation because two Xet caches under `/kaggle/working` exhausted disk during concurrent checkpoint reconstruction. Version 2 moved caching outside the persisted output directory, disabled Xet reconstruction, and deleted the cache after both workers exited. The failure and repair did not expose or condition on behavioral outcomes.

No score mean, checkpoint contrast, wording effect, position effect, or permutation-invariance result was inspected for this integrity report. Wave 01 remains uninterpretable in isolation under the frozen protocol.
