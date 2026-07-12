# Kaggle execution

GPU workflows use two isolated T4 workers, write restartable outputs under
`/kaggle/working/`, and fail closed on missing, malformed, or token-capped cells.

`factorial_v1/` produced the two complete 1,728-cell OLMo factorials:

- explicit NF4/fp16/double quantization; and
- standardized BitsAndBytes default 4-bit loading under Transformers 4.57.6.

The historical original and P1/P2 runs did not share a runtime. The reciprocal
reproduction is therefore split across `historical_stage_v1/` and
`historical_paraphrase_v1/`. Each pins the retained Kaggle image and exact
historical package resolution, then runs the same 216 historical-layout cells.
Together they form a 432-cell runtime crossover.

`tulu_endpoint_v1/` is a 96-generation feasibility-only pilot for the pinned
Tülu DPO and selected GRPO endpoints. Its output is not the full marginalized
Phase 2 estimate and cannot be used as a safety result.

```bash
make kaggle-bundle
kaggle datasets version \
  -p kaggle/datasets/factorial-v1 \
  -m "chore: refresh counterbalanced experiment bundle" \
  --dir-mode zip
kaggle kernels push \
  -p kaggle/factorial_v1 \
  --accelerator NvidiaTeslaT4
```

The same pattern applies to each kernel directory. Always wait for the dataset
version to reach `ready` before pushing a kernel that depends on a new config or
package snapshot.

The private dataset bundle includes the prompt pack, source-aware validation
artifacts, all experiment configs, and an exact copy of `src/rlvr_safety/`.
Generated bundle contents are ignored because `scripts/prepare_kaggle_bundle.py`
rebuilds them. Dataset metadata and kernel sources remain tracked.

Every completed run must be pulled, checked against its `run_summary.json`,
analyzed locally, reduced to a compact checksummed bundle under `artifacts/`,
and referenced by remote kernel slug. Preserve failed versions when a strict
gate catches dependency or runtime drift.

The earlier `seed3/` runner is superseded. Its prompt pack repeated the fixed
A=safe/B=ambiguous/C=risky confound, and its first remote run used P100 hardware
and failed before producing scorable results.
