# Tülu trajectory wave 04 integrity report

**Gate result: PASS. Scientific interpretation: withheld until the complete trajectory panel is available.**

Wave 04 evaluated pinned GRPO checkpoints at steps 960 and 1280 across the full 576-condition structured panel per checkpoint.

| Checkpoint | Rows | Strict parses | Malformed | Capped | Peak GPU memory | Exact revision |
|---|---:|---:|---:|---:|---:|---|
| step 960 GRPO | 576 | 576 | 0 | 0 | 6.631 GiB | yes |
| step 1280 GRPO | 576 | 576 | 0 | 0 | 6.631 GiB | yes |

The accepted run produced 1,152/1,152 unique cells in approximately 26.6 wall-clock minutes on Tesla T4×2. It used deterministic decoding, NF4/fp16/double quantization, explicit all-ones attention masks, and the frozen prompt SHA-256 `040de139b26f4d78230dc47ec9ee4f9be933b13bbeb3e44eb1a3d6630bb7ed29`.

No behavioral or measurement outcome was inspected for this integrity report.
