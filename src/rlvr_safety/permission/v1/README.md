# Permission benchmark v1 (frozen)

These modules are the 5 October 2026 development scaffold, the code that produced the first
real-model smoke run (26 responses, protocol gate failed). They are kept so those results can be
replayed. **Do not extend them.** The canonical, current implementation lives one level up in
`src/rlvr_safety/permission/` (and `infra/runpod/` for the runner).

Files were moved with `git mv` and only these lines changed: relative import paths, the code-hash
list in `benchmark._code_hashes`, and the bundle layout in `infra/runpod/v1/permission_smoke.py`.
First 16 hex characters of each original file's SHA-256 (as of commit `e096741`):

| Original | Now | SHA-256 (original) |
|---|---|---|
| `src/rlvr_safety/permission_schema.py` | `src/rlvr_safety/permission/v1/schema.py` | `ea639fb53701271d` |
| `src/rlvr_safety/permission_prompts.py` | `src/rlvr_safety/permission/v1/prompts.py` | `f4ab47effd31eb2d` |
| `src/rlvr_safety/permission_environment.py` | `src/rlvr_safety/permission/v1/environment.py` | `ae6a9ae77ba6c4cd` |
| `src/rlvr_safety/permission_benchmark.py` | `src/rlvr_safety/permission/v1/benchmark.py` | `3bc1bcc990893045` |
| `src/rlvr_safety/permission_generation.py` | `src/rlvr_safety/permission/v1/generation.py` | `5482a19d80de1985` |
| `src/rlvr_safety/cli/permission_dev.py` | `src/rlvr_safety/permission/v1/cli.py` | `dfb3e7cb43af8146` |
| `infra/runpod/permission_smoke.py` | `infra/runpod/v1/permission_smoke.py` | `219e7e2b87911e22` |
| `infra/runpod/permission_workload.py` | `infra/runpod/v1/permission_workload.py` | `d755439c29cadde2` |

`tests/v1/test_replay_oct5_smoke.py` replays all 26 saved records (private docs checkout) and
checks every score is reproduced exactly.
