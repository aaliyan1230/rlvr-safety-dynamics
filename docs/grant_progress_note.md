# Grant Progress Note

As of the initial scaffold, the project has concrete artifacts rather than only a plan:

* Created a public-ready local project structure for `rlvr-safety-dynamics`.
* Drafted a seed prompt set with 48 prompts across goal persistence, resource acquisition, self-preservation, deception, evaluation awareness, and benign controls.
* Wrote a manual scoring rubric with a 0-2 instrumental endorsement scale and safety/oversight tags.
* Added model-pair configuration for Qwen3 4B and OLMo 3 follow-up comparisons.
* Added a prompt validation script.
* Added a Hugging Face generation runner intended for Kaggle/free-tier GPU use.
* Added a manual scoring CSV-template generator.
* Added local gates: prompt validation, script compilation, and a non-GPU scoring-template smoke test.

Remaining pre-submission gates:

1. Run `make validate`, `make compile`, and `make smoke-score`.
2. Create the GitHub repository or another public work link.
3. Optionally run a tiny CPU smoke test with a toy model, or run the first real generation pass on Kaggle.
4. Update the grant application with the public repo link and this concrete progress.
