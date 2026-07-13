PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: validate validate-choice audit-ai anchors trajectory-wave-configs cross-format-wave-configs subset choice-eval factorial-pack kaggle-bundle compile test verify-artifacts check smoke-score smoke-choice-score smoke-gemini-judge smoke-gemini-paraphrase smoke-judge-analysis paper-tables baseline-analysis analyze-sensitivity analyze-factorial analyze-factorial-nf4 analyze-factorial-default4bit compare-factorial-protocols compare-historical-default4bit compare-historical-exact-stage compare-historical-exact-paraphrase analyze-runtime-crossover analyze-tulu-pilot

validate:
	$(PYTHON) scripts/validate_prompts.py --prompts data/prompts_seed.jsonl

validate-choice:
	$(PYTHON) scripts/validate_choice_prompts.py --prompts data/choice_eval_targeted.jsonl

audit-ai:
	$(PYTHON) scripts/build_ai_semantic_audit.py

anchors:
	$(PYTHON) scripts/build_cross_format_anchors.py

trajectory-wave-configs:
	$(PYTHON) scripts/build_tulu_wave_configs.py

cross-format-wave-configs: trajectory-wave-configs
	$(PYTHON) scripts/build_tulu_cross_format_configs.py

subset:
	$(PYTHON) scripts/select_prompt_subset.py --prompts data/prompts_seed.jsonl --out results/prompts_cheap_test.jsonl --per-category 2 --seed 7

choice-eval:
	$(PYTHON) scripts/build_choice_eval.py --prompts data/prompts_seed.jsonl --out data/choice_eval_targeted.jsonl --seed 17

factorial-pack:
	$(PYTHON) -m rlvr_safety.cli.build_factorial \
		--original data/choice_eval_targeted.jsonl \
		--wording p1=data/gemini_choice_paraphrases_v1.jsonl \
		--wording p2=data/gemini_choice_paraphrases_v2.jsonl \
		--wording p3=data/gemini_choice_paraphrases_v3.jsonl \
		--out data/choice_factorial_v1.jsonl \
		--manifest-out data/choice_factorial_v1.manifest.json
	$(PYTHON) scripts/validate_choice_prompts.py --prompts data/choice_factorial_v1.jsonl --expected-count 576

kaggle-bundle: factorial-pack cross-format-wave-configs
	$(PYTHON) scripts/prepare_kaggle_bundle.py

compile:
	$(PYTHON) -m compileall -q src scripts tests kaggle/factorial_v1 \
		kaggle/historical_stage_v1 kaggle/historical_paraphrase_v1 \
		kaggle/tulu_endpoint_v1 kaggle/tulu_trajectory_wave_*
	$(PYTHON) -m compileall -q kaggle/tulu_cross_format_wave_*

test:
	$(PYTHON) -m unittest discover -s tests -v

verify-artifacts:
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts data/ai_semantic_audit_v1.manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts data/tulu_cross_format_anchors_v1.manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/baseline/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/factorial_nf4_v1/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/factorial_default4bit_v2/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/historical_stage_runtime_v1/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/historical_paraphrase_runtime_v1/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/runtime_crossover_v1/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_endpoint_pilot_v1/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_trajectory_wave_01/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_trajectory_wave_02/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_trajectory_wave_03/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_trajectory_wave_04/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_trajectory_wave_05/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_trajectory_wave_06/manifest.json
	$(PYTHON) -m rlvr_safety.cli.verify_artifacts artifacts/tulu_structured_trajectory_v1/manifest.json

check: validate validate-choice audit-ai anchors kaggle-bundle compile test smoke-score smoke-choice-score smoke-gemini-judge smoke-gemini-paraphrase smoke-judge-analysis paper-tables baseline-analysis analyze-factorial analyze-runtime-crossover analyze-tulu-pilot verify-artifacts

smoke-score:
	$(PYTHON) scripts/score_manual_template.py --generations tests/fixtures/generations_sample.jsonl --out results/manual_scores_template.sample.csv

smoke-choice-score:
	$(PYTHON) scripts/score_choice_eval.py --generations tests/fixtures/choice_generations_sample.jsonl --out results/choice_scores_template.sample.csv --summary-out results/choice_score_summary.sample.md

smoke-gemini-judge:
	$(PYTHON) scripts/gemini_judge_outputs.py --generations tests/fixtures/gemini_judge_sample.jsonl --out results/gemini_judge_sample.mock.jsonl --max-new-tokens 96 --mock

smoke-gemini-paraphrase:
	$(PYTHON) scripts/gemini_generate_paraphrases.py --choice-prompts data/choice_eval_targeted.jsonl --out results/gemini_paraphrases_sample.mock.jsonl --validation-out results/gemini_paraphrases_validation_sample.mock.jsonl --per-source 1 --limit 2 --pack-id smoke --mock
	$(PYTHON) scripts/validate_choice_prompts.py --prompts results/gemini_paraphrases_sample.mock.jsonl --expected-count 2 --allow-subset-categories

smoke-judge-analysis: smoke-gemini-judge
	$(PYTHON) scripts/analyze_judge_results.py --judge-results results/gemini_judge_sample.mock.jsonl --out-md results/gemini_judge_analysis_sample.mock.md

paper-tables:
	$(PYTHON) scripts/build_paper_tables.py --out results/paper_tables.md
	$(PYTHON) scripts/build_paper_tables.py --out reports/paper_tables.md

baseline-analysis:
	$(PYTHON) -m rlvr_safety.cli.analyze_baseline \
		--freeform artifacts/baseline/freeform_manual_scores.csv \
		--freeform-key artifacts/baseline/freeform_model_key.csv \
		--choice artifacts/baseline/choice_stage_scores.csv \
		--choice-key artifacts/baseline/choice_stage_model_key.csv \
		--out-md reports/baseline_uncertainty.md \
		--out-json results/baseline_metrics.json

analyze-sensitivity:
	$(PYTHON) scripts/analyze_prompt_sensitivity.py \
		--original-scores artifacts/baseline/choice_stage_scores.csv \
		--original-model-key artifacts/baseline/choice_stage_model_key.csv \
		--paraphrase-scores artifacts/baseline/paraphrase_p1_scores_confounded.csv \
		--out-md reports/prompt_sensitivity.md

analyze-factorial: analyze-factorial-nf4 analyze-factorial-default4bit compare-factorial-protocols compare-historical-default4bit

analyze-factorial-nf4:
	$(PYTHON) -m rlvr_safety.cli.analyze_factorial \
		--scores artifacts/factorial_nf4_v1/choice_scores.csv \
		--out-md reports/factorial_nf4_v1_results.md \
		--out-json artifacts/factorial_nf4_v1/factorial_metrics.json \
		--protocol-note "NF4 with fp16 compute and double quantization under Transformers 4.57.6; robustness protocol."

analyze-factorial-default4bit:
	$(PYTHON) -m rlvr_safety.cli.analyze_factorial \
		--scores artifacts/factorial_default4bit_v2/choice_scores.csv \
		--out-md reports/factorial_default4bit_v2_results.md \
		--out-json artifacts/factorial_default4bit_v2/factorial_metrics.json \
		--protocol-note "BitsAndBytes default 4-bit loading under Transformers 4.57.6; standardized fixed-runtime primary protocol, not an exact historical reproduction."

compare-factorial-protocols:
	$(PYTHON) -m rlvr_safety.cli.compare_protocols \
		--reference-dir artifacts/factorial_nf4_v1 \
		--comparison-dir artifacts/factorial_default4bit_v2 \
		--reference-label nf4_double_quant \
		--comparison-label default_4bit \
		--out-json artifacts/factorial_default4bit_v2/protocol_comparison_nf4.json \
		--out-md reports/factorial_nf4_vs_default4bit.md \
		--title "NF4/Double-Quant versus Default 4-bit"

compare-historical-default4bit:
	$(PYTHON) -m rlvr_safety.cli.compare_historical \
		--factorial-scores artifacts/factorial_default4bit_v2/choice_scores.csv \
		--historical original=artifacts/baseline/choice_stage_scores.csv \
		--historical p1=artifacts/baseline/paraphrase_p1_scores_confounded.csv \
		--historical p2=artifacts/baseline/paraphrase_p2_scores_confounded.csv \
		--original-model-key artifacts/baseline/choice_stage_model_key.csv \
		--title "Default-4-bit Standardized Run: Historical Layout Reproduction" \
		--out-md reports/factorial_default4bit_v2_reproduction.md \
		--out-json artifacts/factorial_default4bit_v2/reproduction_metrics.json

compare-historical-exact-stage:
	$(PYTHON) -m rlvr_safety.cli.compare_historical \
		--factorial-scores artifacts/historical_stage_runtime_v1/choice_scores.csv \
		--historical original=artifacts/baseline/choice_stage_scores.csv \
		--original-model-key artifacts/baseline/choice_stage_model_key.csv \
		--title "Exact Stage-Runtime Original-Cell Reproduction" \
		--out-md reports/historical_stage_runtime_reproduction.md \
		--out-json artifacts/historical_stage_runtime_v1/reproduction_metrics.json

compare-historical-exact-paraphrase:
	$(PYTHON) -m rlvr_safety.cli.compare_historical \
		--factorial-scores artifacts/historical_paraphrase_runtime_v1/choice_scores.csv \
		--historical p1=artifacts/baseline/paraphrase_p1_scores_confounded.csv \
		--historical p2=artifacts/baseline/paraphrase_p2_scores_confounded.csv \
		--original-model-key artifacts/baseline/choice_stage_model_key.csv \
		--title "Exact P1/P2-Runtime Candidate-Cell Reproduction" \
		--out-md reports/historical_paraphrase_runtime_reproduction.md \
		--out-json artifacts/historical_paraphrase_runtime_v1/reproduction_metrics.json

analyze-runtime-crossover: compare-historical-exact-stage compare-historical-exact-paraphrase
	$(PYTHON) -m rlvr_safety.cli.analyze_runtime_crossover \
		--stage-dir artifacts/historical_stage_runtime_v1 \
		--paraphrase-dir artifacts/historical_paraphrase_runtime_v1 \
		--out-json artifacts/runtime_crossover_v1/metrics.json \
		--out-md reports/historical_runtime_crossover_v1.md \
		--title "Exact Historical Runtime Crossover"

analyze-tulu-pilot:
	$(PYTHON) -m rlvr_safety.cli.analyze_tulu_pilot \
		--result-dir artifacts/tulu_endpoint_pilot_v1 \
		--config artifacts/tulu_endpoint_pilot_v1/config.json \
		--out-json artifacts/tulu_endpoint_pilot_v1/feasibility_metrics.json \
		--out-md reports/tulu_endpoint_pilot_v1.md
