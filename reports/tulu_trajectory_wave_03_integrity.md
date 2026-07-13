# Tülu trajectory wave 03 integrity report

**Gate result: PASS. Scientific interpretation: withheld until the complete trajectory panel is available.**

Wave 03 evaluated pinned GRPO checkpoints at steps 320 and 640 across the full 576-condition structured panel per checkpoint.

| Checkpoint | Rows | Strict parses | Malformed | Capped | Peak GPU memory | Exact revision |
|---|---:|---:|---:|---:|---:|---|
| step 320 GRPO | 576 | 576 | 0 | 0 | 6.631 GiB | yes |
| step 640 GRPO | 576 | 576 | 0 | 0 | 6.631 GiB | yes |

The accepted run produced 1,152/1,152 unique model-by-condition cells in approximately 26.7 wall-clock minutes on Tesla T4×2. It used deterministic decoding, NF4/fp16/double quantization, explicit all-ones attention masks, and prompt SHA-256 `040de139b26f4d78230dc47ec9ee4f9be933b13bbeb3e44eb1a3d6630bb7ed29`.

Kernel version 1 stopped before generation after Kaggle assigned a single Tesla P100. Version 2 was superseded by an explicit accelerator request. Version 3 requested `NvidiaTeslaT4`, received Tesla T4×2, and passed. These launch decisions did not expose or condition on behavioral outcomes.

No score mean, checkpoint contrast, wording effect, position effect, or permutation-invariance result was inspected for this integrity report. Wave 03 remains uninterpretable in isolation under the frozen protocol.
