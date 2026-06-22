.PHONY: validate compile smoke-score

validate:
	python3 scripts/validate_prompts.py --prompts data/prompts_seed.jsonl

compile:
	python3 -m py_compile scripts/validate_prompts.py scripts/score_manual_template.py scripts/run_behavioral_eval.py

smoke-score:
	python3 scripts/score_manual_template.py --generations tests/fixtures/generations_sample.jsonl --out results/manual_scores_template.sample.csv
