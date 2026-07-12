# Historical inference-runtime audit

Audit date: 2026-07-10.

## Finding

There was no single historical inference protocol behind the legacy original-versus-paraphrase comparison. The original stage scores and the P1/P2 scores were produced in different content-addressed Kaggle images with materially different Transformers and Accelerate versions. The old prompt-pack deltas therefore alias at least three changes:

1. candidate wording;
2. semantic option position; and
3. inference runtime.

The option-position confound alone already makes the legacy deltas non-identifiable as wording effects. The runtime difference is an additional reason not to use those cross-run deltas as a behavioral estimate.

## Reconstructed runtimes

| Historical cells | Kaggle image digest | Runtime after the historical install command |
|---|---|---|
| Original stage baseline | `sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c` | Python 3.12.13; torch 2.10.0+cu128; Transformers 5.12.1; Accelerate 1.14.0; bitsandbytes 0.49.2 |
| P1 and P2 | `sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` | Python 3.12.13; torch 2.10.0+cu128; Transformers 5.0.0; Accelerate 1.13.0; tokenizers 0.22.2; bitsandbytes 0.49.2; sentencepiece 0.2.1; protobuf 5.29.5 |

For the original stage environment, tokenizers 0.22.2, sentencepiece 0.2.1, and protobuf 5.29.5 are best-supported image defaults rather than versions directly printed by that historical kernel. The reproduction config treats them as strict checks: if the pinned image resolves differently, the run fails and preserves the discrepancy rather than silently continuing.

The image digests come from Kaggle's retained kernel metadata. The surrounding image history is available in the official [Kaggle Docker releases](https://github.com/Kaggle/docker-python/releases).

## Evidence trail

The retained historical sources show different dependency operations:

- `rlvr-choice-stage-ablation-v2` ran `pip install -q -U transformers>=4.57.0 accelerate bitsandbytes`. With no upper bound, that upgraded the stage image to Transformers 5.12.1 and Accelerate 1.14.0.
- P1's source at `local/kaggle_paraphrase_choice_v1/script.py` requested `transformers>=4.53.0`, Accelerate, bitsandbytes, and sentencepiece without `-U`. On the retained P1/P2 image, only bitsandbytes needed installation; the image's Transformers 5.0.0 and Accelerate 1.13.0 remained in place.
- P2's retained kernel source installed only `bitsandbytes>=0.46.1` and otherwise used the same P1/P2 image defaults.
- A verbose diagnostic on the identical P1/P2 image printed the complete package stack in the table above.
- The baseline wheel sizes and the adjacent Qwen run metadata under `results/kaggle_qwen_cheap_v2/run_metadata.json` identify Transformers 5.12.1, Accelerate 1.14.0, bitsandbytes 0.49.2, and torch 2.10.0+cu128 for the upgraded stage image.

The first exact-image stage attempt intentionally replayed the historical floating upgrade command. On 2026-07-10 it resolved Transformers 5.13.0, so the strict runtime gate stopped before generation. The failed metadata and log are preserved under `results/kaggle_historical_stage_reproduction_v1_failed_1/`. Subsequent runners record the historical command but pin the versions that originally resolved; replaying a floating dependency operation is not an exact reproduction once the package index changes.

The original kernels did not record resolved Hugging Face revisions. Their requested `main` tips can nevertheless be reconstructed because the following repository commits predate the historical runs and remained the resolved tips in the completed NF4 run:

| Model | Immutable revision |
|---|---|
| OLMo 3 Instruct-SFT | `e1452fc572d51966ff4aaeb25118b891eb93e549` |
| OLMo 3 Instruct-DPO | `b33130b7de49f0c2553b5c2b3bc8409ff3e627d1` |
| OLMo 3 Instruct | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |
| OLMo 3 RL-Zero-General | `24e5814cb36944d3eb36a49679d02c2b3bb46195` |

The first three revisions are also recorded directly in `artifacts/factorial_nf4_v1/model_*_metadata.json`. The reproduction study excludes RL-Zero-General because its historical structured outputs were all token-capped and do not pass the clean-model gate.

## Shared generation settings

The historical kernels otherwise used the same core loading and generation path:

- tokenizer and causal LM loaded with `trust_remote_code=True`;
- `device_map="auto"`;
- `BitsAndBytesConfig(load_in_4bit=True)` with no explicit compute dtype, quantization type, or double-quantization setting;
- the model repository's chat template with the same system and user messages;
- greedy decoding, a 96-token maximum, `temperature=None`, `top_p=None`, and EOS as the padding token; and
- no explicit attention mask.

Under the two retained Transformers versions, the default bitsandbytes configuration is FP4 with float32 compute, no double quantization, and uint8 storage. The exact-image runners also record the model's resolved quantization configuration and fail if the loaded model does not report itself as quantized.

P2 has one irrecoverable historical caveat: its source caught any 4-bit load exception and silently retried with fp16, without recording the selected branch. Four-bit loading is strongly supported because P1 succeeded on the same image and package set, but it was not proven by P2's own metadata. The new reproduction has no fallback.

## Correct reproduction design

Merely rerunning each pack in its original environment would reproduce 216 cells but retain perfect aliasing between pack and runtime. The implemented correction is the smallest reciprocal crossover:

| Factor | Levels |
|---|---|
| Runtime | original-stage image; P1/P2 image |
| Pack/layout | original at each source's historical order; P1 at order 012; P2 at order 012 |
| Model | SFT; DPO; final Instruct |
| Source | 24 |
| Total | 2 × 3 × 3 × 24 = **432 generations** |

`configs/experiments/historical_stage_reproduction_v1.json` runs all three layouts under the stage runtime. Its original cells are fidelity checks; its P1/P2 cells are reciprocal runtime controls. `configs/experiments/historical_paraphrase_reproduction_v1.json` runs the same layouts under the P1/P2 runtime. Its P1/P2 cells are fidelity checks; its original cells are reciprocal controls.

Both kernels pin the retained Docker digest, record the historical dependency command, install its exact historical resolution, enforce the expected runtime versions, use immutable model commits, record actual quantization state, and require complete uncensored output. The primary runtime estimand is a source-paired difference-in-differences:

`[(candidate - original) under P1/P2 runtime] - [(candidate - original) under stage runtime]`.

This estimates how much the apparent pack delta changes solely with the runtime, while holding the pack layout fixed. It still does not isolate wording from option position. Only the 3! counterbalanced factorial identifies order-marginalized wording contrasts.

## Status

Completed 2026-07-10 UTC. Both strict exact-image panels produced all 216 expected rows with zero malformed or capped outputs.

- The stage-image panel reproduces all 72 historical original cells at 100% score and response-letter agreement.
- The P1/P2-image panel reproduces all 144 historical candidate cells at 100% score and response-letter agreement.
- Across the 216 identical layouts, the two runtimes agree on 174 cells (80.6%). The P1/P2-image runtime raises the pooled score by +0.088 [0.009, 0.171].
- By layout, runtime deltas are +0.042 [-0.056, 0.153] for original, +0.028 [-0.153, 0.236] for P1, and +0.194 [0.056, 0.361] for P2.
- The runtime × pack interactions are -0.014 [-0.208, 0.208] for P1 and +0.153 [-0.056, 0.361] for P2. Neither interval excludes zero.

The runtime effect is therefore material, but the 24-source crossover does not precisely partition the historical candidate shifts. The old deltas remain jointly confounded by wording, position, and runtime. The full result is in `reports/historical_runtime_crossover_v1.md`; checksummed inputs and metrics are under `artifacts/historical_stage_runtime_v1/`, `artifacts/historical_paraphrase_runtime_v1/`, and `artifacts/runtime_crossover_v1/`.

The Transformers 4.57.6 default-4-bit factorial remains a standardized fixed-runtime counterfactual, not an exact historical replication.
