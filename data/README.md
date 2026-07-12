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

## Legacy validation artifacts

`gemini_choice_paraphrase_validations_v2.jsonl` is retained for historical
reproducibility. That earlier check saw only the candidate item, used the same
model family as generation, and established internal score ordering—not
semantic equivalence to the source. It is not the authority for the new
factorial experiment.
