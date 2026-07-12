# Tülu endpoint feasibility pilot

**Gate result: PASS. Scientific interpretation: not allowed.**

This pilot validates checkpoint access, format compliance, memory, and throughput only. Its two-order endpoint scores are not a marginalized safety estimate.

Wall-clock runtime: 7.8 minutes on Tesla T4, Tesla T4.

| Endpoint | Rows | Strict parses | Malformed | Capped | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| tulu_step_0000_dpo | 48 | 48 | 0 | 0 | 6.63 GiB |
| tulu_step_1920_grpo | 48 | 48 | 0 | 0 | 6.63 GiB |

Both immutable revisions resolved exactly. Generation used `explicit_all_ones` attention masking and the pinned NF4/fp16/double-quantization protocol.

The pilot includes only the original wording and orders 012/210. Endpoint score differences are deliberately not reported as evidence about GRPO or safety.
