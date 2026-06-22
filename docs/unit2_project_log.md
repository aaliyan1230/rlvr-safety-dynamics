# Unit 2 Project Log: Test Your Ideas

This file answers BlueDot Technical AI Safety Project Unit 2 prompts for the RLVR safety dynamics project.

## 1. Get It Working

Current working scaffold:

* Seed prompt dataset exists with 48 prompts across goal persistence, resource acquisition, self-preservation, deception, evaluation awareness, and benign controls.
* Manual scoring rubric exists with a 0-2 instrumental endorsement scale.
* Model configuration exists for Qwen3 4B and OLMo 3 follow-up comparisons.
* Local scripts exist for prompt validation, generation, and manual scoring template creation.
* Local gates passed:
  * `make validate`
  * `make compile`
  * `make smoke-score`

Stuck points / things to improve:

* No real model generations have been run yet.
* The prompt set is custom and inspired by InstrumentalEval, so construct validity is a live concern.
* The first model pair is a Thinking/Instruct comparison, not a clean before/after RLVR causal comparison.
* Visible thinking traces and response length may confound scoring.
* The strongest public artifact needs a GitHub remote and at least a small real generation run.

## 2. Project Summary

This project tests whether current small open Thinking or RLVR-lineage models show stronger instrumental-convergence-style behavior than matched instruction models. I will start with a Kaggle-feasible Qwen3 4B Instruct vs Thinking comparison, using a small prompt set covering goal persistence, resource acquisition, self-preservation, deception, evaluation awareness, and benign controls. Outputs will be manually scored with a transparent rubric before any LLM judge is used. If the behavioral signal is nontrivial, I will add paraphrase checks and selected activation-level measurements to see whether the differences survive wording changes and have simple representation-level correlates.

## 3. Assumptions

| ID | Assumption | Type | Why it matters |
|---|---|---|---|
| A1 | InstrumentalEval-style prompts can measure a useful proxy for goal persistence, resource acquisition, self-preservation, deception, and evaluation awareness. | Method | If false, the project measures prompt artifacts rather than safety-relevant behavior. |
| A2 | A 48-60 prompt first run is enough to reveal whether there is any signal worth expanding. | Method | If false, the MVP may be too underpowered to inform next steps. |
| A3 | Qwen3 4B Instruct and Thinking variants are close enough to make a useful first comparison. | Existing work / model choice | If false, the result is too confounded by model differences. |
| A4 | OLMo 3 Instruct / Think / RL-Zero checkpoints are accessible and runnable enough to support a more RLVR-relevant follow-up. | Tools and resources | If false, the project must remain a reasoning-specialization audit rather than an RLVR-lineage audit. |
| A5 | Kaggle T4x2 or a small $100 compute grant is enough for at least one meaningful 4B pair and possibly one 7B follow-up. | Tools and resources | If false, the project needs a smaller subset or a stronger compute ask. |
| A6 | Manual scoring can distinguish safe redirection from instrumental endorsement with tolerable ambiguity. | Data / method | If false, aggregate scores will not be trustworthy. |
| A7 | Prompt paraphrases can reveal wording sensitivity without making the project too large. | Method | If false, robustness checks may dominate the project. |
| A8 | Thinking models' longer outputs can be controlled by token caps, response-length reporting, and scoring final answers separately from visible reasoning traces. | Method | If false, differences may mostly reflect verbosity rather than behavior. |
| A9 | Small models are capable enough to express the relevant behaviors in text. | Existing work / data | If false, null results may be uninformative about larger reasoning models. |
| A10 | Selected final-token or answer-token activations are enough for a lightweight representation-level follow-up. | Method | If false, activation analysis should be deferred rather than forced. |
| A11 | A public, limitations-aware audit would be useful to people thinking about RLVR side effects, open-model evals, or FAR-style research engineering. | Impact | If false, the project may still be educational but less valuable externally. |
| A12 | The project can stay honest about causal limits while still being portfolio-relevant. | Impact / communication | If false, the write-up risks overclaiming or seeming too incremental. |

## 4. Cruxes and Cheapest Checks

| Assumption | Importance | Uncertainty | Cheapest useful check | Prediction before testing | Change-course result |
|---|---:|---:|---|---|---|
| A6: Manual scoring works | High | Medium | Score 10-15 fixture or first-run outputs blind to model identity. | Most outputs will be scoreable as 0/1/2 after one rubric pass; ambiguous cases will cluster around resource acquisition and self-preservation. | If more than one-third of outputs are ambiguous even after rubric refinement, simplify categories or switch to qualitative case study. |
| A5: Compute is enough | High | Medium | Load one Qwen3 4B model in 4-bit and generate a 6-12 prompt subset. | Qwen3 4B should load on Kaggle T4x2 in 4-bit with batch size 1 and 384 max new tokens. | If loading/generation is unstable, use a smaller prompt subset, use a smaller model, or request grant compute before expanding. |
| A1: Prompts measure the intended behavior | High | High | Manually inspect 2-3 prompts per category and check whether a safe answer and risky answer are clearly distinguishable. | Goal persistence, deception, and evaluation awareness will be cleaner than self-preservation/resource acquisition. | If categories collapse into generic safety-refusal behavior, revise prompts toward more discriminating scenarios. |
| A8: Thinking verbosity is controllable | Medium | High | Compare response lengths and score final answer separately from visible thinking traces. | Thinking model outputs will be longer, but final-answer scoring and token caps will keep comparison usable. | If verbosity dominates scores, report length as primary confound and avoid behavior claims. |
| A3: Qwen3 pair is a useful first comparison | Medium | Medium | Run the same 6-12 prompt subset on both Qwen3 4B variants. | The Thinking model may show slightly stronger goal-persistence/evaluation-awareness language, but effect may be uneven. | If differences are all attributable to style/verbosity, pivot to OLMo 3 RL-Zero or frame Qwen3 only as tooling smoke test. |

## 5. First Cheap Test

### What will I be testing?

The first cheap test is a small Qwen3 4B Instruct vs Thinking generation run on 6-12 prompts sampled from the seed set. It tests whether the model-loading path works, whether outputs are scoreable under the rubric, whether response length is a serious confound, and whether there is any visible behavioral difference worth expanding.

### What do I expect the results to be? Why?

I expect the tooling to work with 4-bit loading and batch size 1, though the Thinking model may be slower and more verbose. I expect most outputs to be scoreable, with clearest distinctions in deception and evaluation-awareness prompts. I do not expect a clean or uniform "Thinking model is more instrumental" result from the first tiny run; a more realistic prediction is that any signal will be category-specific and entangled with response style.

### What result would make me change course?

I would change course if:

* The Qwen3 4B models do not load or generate reliably on available compute.
* The rubric cannot score outputs consistently because nearly all responses are generic safety statements.
* Thinking outputs are so much longer that the comparison is mostly a verbosity artifact.
* The prompts fail to distinguish safe, ambiguous, and instrumental responses.

In those cases, I would either reduce the prompt set, revise the categories, use Qwen3 only as a tooling smoke test, or pivot the main comparison to OLMo 3 checkpoints before making stronger claims.

## 6. What This Means for the Next Work Session

The next highest-value action is not a full benchmark. It is a small generation run that can answer the first cruxes:

1. Can the models run under the compute constraints?
2. Are the outputs scoreable?
3. Are the categories meaningful?
4. Does response length or thinking-trace exposure dominate the comparison?
5. Would the result change whether I expand, revise, or pivot?
