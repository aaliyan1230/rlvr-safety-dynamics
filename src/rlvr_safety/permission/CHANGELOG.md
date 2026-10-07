# Permission benchmark changelog

Releases are identified by date-stamped IDs stored inside every record, never by file names.
Records written by the frozen 5 October scaffold use the legacy ID `permission-development-v1`
and are read only by `permission/v1/`.

## permission-schema-2026-10-07

First canonical schema. Differences from the legacy scaffold:

- **Action order is a design factor.** The menu order of the three actions rotates (all six orders
  for multiple choice; a rotating single order or all six for executable prompts). Allowed and
  forbidden twins always share order, wording and IDs.
- **Executable decision is exactly one next action**, answered as `{"action": "<id>"}`. The legacy
  prompt allowed zero to two actions.
- **Action IDs have two arms**: the scenario's own IDs (`semantic`) and deterministic neutral IDs
  (`neutral`, e.g. `op_k`), so we can test whether names steer the model.
- **The option-free prompt no longer shows an operation list.**
- **Multiple choice shows one lettered menu** (the legacy prompt showed a bulleted list and then a
  second lettered one) and asks for "one letter only".
- **Controls are first-class records** (`control.requested_action`): allowed-only, executable-only,
  with an exact final-state goal that only the requested action reaches.
- Records can carry `authoring` provenance and `supersedes` (the legacy `source_id` they revise).
- `checks.py` adds mechanical checks (effects execute, alternative and boundary reach the goal,
  clarification leaves state unchanged, twins differ only in the permission sentence, unique
  prompts, balanced rotation). Passing them does not replace two-person human review.

Planned, not yet released: `mcq-rule-2026-10-08` (answer-reading rule), `mcq-judge-2026-10-09`
(judge rubric).
