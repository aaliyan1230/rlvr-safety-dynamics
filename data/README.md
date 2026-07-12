# Data and prompt packs

## Canonical inputs

- `prompts_seed.jsonl`: 48 free-form prompts.
- `choice_eval_targeted.jsonl`: 24 structured-choice source items.
- `gemini_choice_paraphrases_v1.jsonl` through `v3.jsonl`: candidate
  scenario/option rewrites, one per source item.
- `gemini_choice_paraphrase_validations_source_aware_v*.jsonl`: source-aware
  reviews by `gemini-2.5-pro`. These compare each candidate with the original
  scenario and score-0/1/2 option meanings.
- `choice_factorial_v1.jsonl`: 24 sources × 4 wordings × all six semantic
  option permutations (576 conditions).
- `choice_factorial_v1.manifest.json`: input hashes and balance counts.
- `freeform_adjudications_v1.jsonl`: the six final-label changes applied to
  the baseline manual scoring sheet.
- `ai_semantic_audit_v1.jsonl`: 72 prompt-only semantic judgments from one
  disclosed AI reviewer, accepted by explicit project-owner assumption in
  place of the planned human audit. This is not independent human validation.
- `tulu_freeform_anchor_v1.jsonl`: the frozen option-free form of the 24
  structured safety constructs, including source-specific 0/1/2 scoring anchors.
- `tulu_freeform_scoring_protocol_v1.md`: blinded two-pass AI scoring,
  adjudication, agreement, and claim rules accepted under the same disclosed
  substitution assumption.
- `tulu_capability_anchor_v1.jsonl`: 30 authored held-out, objectively scored
  items spanning the Tülu GRPO mixture's GSM, MATH, and instruction-following
  task families.
- `tulu_cross_format_anchors_v1.manifest.json`: frozen hashes and workload
  counts for both cross-format panels.

## Legacy validation artifacts

`gemini_choice_paraphrase_validations_v2.jsonl` is retained for historical
reproducibility. That earlier check saw only the candidate item, used the same
model family as generation, and established internal score ordering—not
semantic equivalence to the source. It is not the authority for the new
factorial experiment.
