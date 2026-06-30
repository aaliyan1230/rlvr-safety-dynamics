.PHONY: validate validate-choice subset choice-eval compile smoke-score smoke-choice-score smoke-gemini-judge smoke-gemini-paraphrase smoke-judge-analysis

validate:
	python3 scripts/validate_prompts.py --prompts data/prompts_seed.jsonl

validate-choice:
	python3 scripts/validate_choice_prompts.py --prompts data/choice_eval_targeted.jsonl

subset:
	python3 scripts/select_prompt_subset.py --prompts data/prompts_seed.jsonl --out results/prompts_cheap_test.jsonl --per-category 2 --seed 7

choice-eval:
	python3 scripts/build_choice_eval.py --prompts data/prompts_seed.jsonl --out data/choice_eval_targeted.jsonl --seed 17

compile:
	python3 -m py_compile scripts/validate_prompts.py scripts/validate_choice_prompts.py scripts/select_prompt_subset.py scripts/build_choice_eval.py scripts/score_choice_eval.py scripts/score_manual_template.py scripts/run_behavioral_eval.py scripts/gemini_common.py scripts/gemini_judge_outputs.py scripts/gemini_generate_paraphrases.py scripts/analyze_judge_results.py

smoke-score:
	python3 scripts/score_manual_template.py --generations tests/fixtures/generations_sample.jsonl --out results/manual_scores_template.sample.csv

smoke-choice-score:
	python3 scripts/score_choice_eval.py --generations tests/fixtures/choice_generations_sample.jsonl --out results/choice_scores_template.sample.csv --summary-out results/choice_score_summary.sample.md

smoke-gemini-judge:
	python3 scripts/gemini_judge_outputs.py --generations tests/fixtures/gemini_judge_sample.jsonl --out results/gemini_judge_sample.mock.jsonl --max-new-tokens 96 --mock

smoke-gemini-paraphrase:
	python3 scripts/gemini_generate_paraphrases.py --choice-prompts data/choice_eval_targeted.jsonl --out results/gemini_paraphrases_sample.mock.jsonl --validation-out results/gemini_paraphrases_validation_sample.mock.jsonl --per-source 1 --limit 2 --mock
	python3 scripts/validate_choice_prompts.py --prompts results/gemini_paraphrases_sample.mock.jsonl --expected-count 2 --allow-subset-categories

smoke-judge-analysis: smoke-gemini-judge
	python3 scripts/analyze_judge_results.py --judge-results results/gemini_judge_sample.mock.jsonl --out-md results/gemini_judge_analysis_sample.mock.md
