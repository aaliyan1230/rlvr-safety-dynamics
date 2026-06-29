.PHONY: validate subset compile smoke-score

validate:
	python3 scripts/validate_prompts.py --prompts data/prompts_seed.jsonl

subset:
	python3 scripts/select_prompt_subset.py --prompts data/prompts_seed.jsonl --out results/prompts_cheap_test.jsonl --per-category 2 --seed 7

compile:
	python3 -m py_compile scripts/validate_prompts.py scripts/select_prompt_subset.py scripts/score_manual_template.py scripts/run_behavioral_eval.py

smoke-score:
	python3 scripts/score_manual_template.py --generations tests/fixtures/generations_sample.jsonl --out results/manual_scores_template.sample.csv
