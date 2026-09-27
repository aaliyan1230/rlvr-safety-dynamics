# research-compute — TEMPLATE, not a shared dependency

API-driven GPU pods for research experiments, on Lambda Cloud. Built to
replace manually clicking through RunPod's console: agents (Claude, Codex,
you) launch a pod, run something, and the pod stops itself when done, via
one command.

**This folder is a template, never a live dependency.** No project's coding
agent should reach into `~/Projects/research-compute` at run time — that
would mean every project secretly depends on a folder that happens to exist
on one machine. Instead, onboarding a project **copies** `lambda/`,
`pod-env/`, and `bin/` straight into that project's own repo (e.g.
`<project>/infra/lambda/`), plus a project-scoped skill file into that
project's own `.claude/skills/`. After that copy, the project is fully
self-contained: its coding agent never touches this folder again. See
"Onboarding a new project" below — do this via the `lambda-gpu-pods` global
skill, which exists specifically to perform that copy.

The only thing shared *across* projects afterward is the Lambda account
itself and (by choice) one persistent filesystem name/region — a config
value each project's own copy of `config.yaml` carries independently, not a
code dependency. Projects can drift from this template after copying and
that's fine; fixes made here don't propagate automatically, you'd re-copy
or hand-patch as needed.

## Design

- **One shared persistent filesystem** holds everything that shouldn't be
  re-downloaded or reinstalled: the Hugging Face model cache, uv's package
  cache, and a per-project Python venv for each research repo that uses
  this. A pod attaches, does its thing, and disappears; the filesystem
  outlives every pod.
- **One venv per project**, all on that shared filesystem. Different repos
  can pin conflicting package versions without stepping on each other, but
  they all draw from the same shared package-download cache, so bootstrapping
  a *second* project's venv is still a local operation, not a re-download.
- **Lock-file-first**, same pattern as crosslingual-rule-following's
  `infra/runpod/`: each project repo commits its own
  `requirements.lock.txt`; the first bootstrap of a project generates one if
  it doesn't exist yet, everyone/every agent after that installs from it.
- **The pod never sits there billing while idle.** `bin/run.sh` stops the
  pod itself the moment its command finishes, errors, or gets interrupted
  (via a shell `trap`, so this happens even if the command Ctrl-C's mid-run).
  There's deliberately no idle-timeout watchdog or hard runtime cap layered
  on top of that by default — that was a specific choice (see
  "Safety model" below) — but `lambda/config.yaml` has an opt-in
  `max_runtime_hours` backstop if you ever want one.
- **Launching costs real money, so it always requires an explicit `--yes`**,
  both in `lambda_cli.py` and `bin/run.sh`. Nothing in this toolkit launches
  a pod on its own initiative. See "Safety model."
- **No GitHub credentials ever touch the pod's disk.** Code sync uses SSH
  agent forwarding (`ssh -A`) — the pod authenticates to GitHub through your
  laptop's already-loaded key for the duration of the clone, nothing is
  written to the shared filesystem. See `docs/LAMBDA_NOTES.md`.

Full API research/gotchas: [`docs/LAMBDA_NOTES.md`](docs/LAMBDA_NOTES.md).

## Safety model

Launching a pod is a billed action, same category as a purchase. This
toolkit's policy, matching what was decided when it was built:

1. **Always confirm before launching.** If you're an agent reading this to
   decide whether to run something: get the user's explicit go-ahead in
   chat first — state the instance type and its $/hr price — *then* pass
   `--yes`. Never pass `--yes` on your own initiative.
2. **The pod stops itself when the task ends**, success or failure. This is
   the only automatic safety net by design (no separate idle-timeout
   watchdog). If you use `--keep-alive` for an interactive debugging
   session, `bin/run.sh` prints the exact `stop` command at the end —
   actually run it when you're done.
3. Check `python3 lambda/lambda_cli.py list` if you're ever unsure what's
   running, and `python3 lambda/lambda_cli.py stop-all --yes` as an
   emergency kill-everything switch.

## First-time setup (once per Lambda account, not per project)

### Project credentials

The CLI also reads Lambda settings from the project's ignored `.env` file.
Copy `.env.example`, fill `LAMBDA_API_KEY` locally, and run `chmod 600 .env`.
An exported environment variable overrides the file. Only Lambda variables
are loaded; other project credentials are not forwarded. Never commit real keys.

To make the Lambda key available to a GPU workload, set
`LAMBDA_FORWARD_API_KEY=1` in the local `.env`, or use
`bin/run.sh --forward-lambda-api-key` with the usual run arguments. The wrapper
sends the key over encrypted SSH stdin and exports it in the workload process.
It does not save it to the pod's filesystem, repo, or persistent activation file.
Use `--no-forward-lambda-api-key` to disable this for one run. A workload that
prints its environment can still expose the key; share keys only through a
private secret channel. Lambda API keys grant full API access and can launch
paid resources. Verify the effective workspace; selecting a workspace in the
console does not by itself prove the key targets it.

No GPU launch is authorized by setting a key or enabling forwarding. Configure
the filesystem and registered SSH key, verify the key's workspace in Lambda,
and obtain the per-launch approval described above before running a workload.

1. **Get an API key**: https://cloud.lambda.ai/api-keys → save it to a
   local file rather than pasting it anywhere it'd end up in git or a chat
   transcript:
   ```bash
   mkdir -p ~/.config/lambda-cloud
   echo 'YOUR_KEY_HERE' > ~/.config/lambda-cloud/api_key
   chmod 600 ~/.config/lambda-cloud/api_key
   ```
   Every command below reads it via
   The CLI reads `LAMBDA_API_KEY` first, then this file. Set
   `LAMBDA_API_KEY_FILE` to use a different local key file. The key determines
   the Lambda workspace; `config.yaml` contains resource names, not an
   organization selector.

2. **Register an SSH key with Lambda**, if you haven't already (reuses
   `~/.ssh/lambda_id_ed25519` if you generated one for this; otherwise
   `ssh-keygen -t ed25519 -f ~/.ssh/lambda_id_ed25519`):
   ```bash
   python3 lambda/lambda_cli.py ssh-keys
   ```
   Check the output for a key whose `public_key` matches your local
   `~/.ssh/lambda_id_ed25519.pub` byte for byte — if one's already there
   (common if you've used Lambda's console before), just set
   `ssh_key_name` in `lambda/config.yaml` to *its* registered name.
   Otherwise register it:
   ```bash
   python3 lambda/lambda_cli.py add-ssh-key --name <a-name> \
       --pubkey-file ~/.ssh/lambda_id_ed25519.pub
   ```
   and use that name in `ssh_key_name`.

3. **Find a region with A100 capacity** (`preferred_instance_types` in
   `lambda/config.yaml` already tries a real single-GPU 80GB name first,
   falling back to 40GB — but as of writing Lambda doesn't actually sell a
   single-GPU 80GB A100 at all, only an 8x-GPU 80GB bundle at ~11x the
   price, so in practice this resolves to 40GB):
   ```bash
   python3 lambda/lambda_cli.py find-capacity --prefer a100_80gb_sxm4 a100_sxm4 a100
   ```
   If nothing matches (Lambda's lineup/names can change), run
   `python3 lambda/lambda_cli.py types --filter a100` to see the exact
   current names and fix `preferred_instance_types` accordingly.

4. **Create the persistent filesystem in that region**, via the
   [Lambda console](https://cloud.lambda.ai/filesystems) → New Filesystem
   → same region printed above. (The public API only exposes *listing*
   filesystems, not creating them, as of this writing — see
   `docs/LAMBDA_NOTES.md`. This is a two-minute one-time step.) Then
   confirm it:
   ```bash
   python3 lambda/lambda_cli.py filesystems
   ```

5. **Fill in `lambda/config.yaml`**: set `filesystem:` and `region:` to
   what you just created. Commit this file — it's not a secret, and every
   agent/teammate using this toolkit needs the same values.

## Onboarding a new project (copies files — no dependency afterward)

This is what the `lambda-gpu-pods` global skill does automatically when you
ask it to set up a project for GPU pods; the steps below are what it's
actually doing, spelled out in case you want to do it by hand.

1. **Copy the toolkit into the target repo:**
   ```bash
   TARGET=~/Projects/<your-other-project>
   mkdir -p "$TARGET/infra/lambda"
   cp -r ~/Projects/research-compute/lambda ~/Projects/research-compute/pod-env \
         ~/Projects/research-compute/bin ~/Projects/research-compute/docs \
         "$TARGET/infra/lambda/"
   ```
   `config.yaml` comes along as-is — it already has the shared
   `filesystem`/`region`/`ssh_key_name` filled in from the account-level
   bootstrap above, so nothing else to configure for the shared-filesystem
   default.

2. **Copy a project-scoped skill** so that project's own coding agent
   discovers this without any global/external reference:
   ```bash
   mkdir -p "$TARGET/.claude/skills/lambda-gpu-pods"
   cp ~/Projects/research-compute/PROJECT_SKILL_TEMPLATE.md \
      "$TARGET/.claude/skills/lambda-gpu-pods/SKILL.md"
   ```
   That template already uses paths relative to the repo root
   (`infra/lambda/...`), not this folder's absolute path — check it if
   you're doing this by hand, nothing to edit.

3. **Add the project's own `requirements.txt`** (floor-pinned deps; torch is
   handled separately by `bootstrap_pod.sh`) somewhere in its tree — root by
   default, or pass `--req-file`/`--lock-file` (relative to the repo root)
   to `infra/lambda/bin/run.sh` if it lives elsewhere. Leave
   `requirements.lock.txt` **absent**; the first run generates and prints
   it for you to commit.

4. **Commit `infra/lambda/` and `.claude/skills/lambda-gpu-pods/` into that
   project's own repo.** From here on, that project is self-contained —
   its coding agent runs `infra/lambda/bin/run.sh` locally, never reaches
   into `~/Projects/research-compute`.

## Running an experiment (from inside a project that's been onboarded)

Run from that project's own repo root, using its own copy of the toolkit:

```bash
infra/lambda/bin/run.sh \
  --repo git@github.com:your-org/your-project.git \
  --branch main \
  --req-file requirements.txt \
  --lock-file requirements.lock.txt \
  --cmd "python experimental/analysis/some_script.py" \
  --yes
```

What this does, end to end: finds a live A100 with capacity in the
filesystem's region → launches it with the filesystem attached → waits for
SSH → clones/pulls the repo on the pod over forwarded-agent git auth →
bootstraps (or reuses) that project's venv → sets `HF_HOME` to the shared
model cache → runs your command → **stops the pod**, always, even if the
command fails.

For gated HF models, export `HF_TOKEN` before running and reference it in
your `--cmd`, e.g. `--cmd "HF_TOKEN=$HF_TOKEN python train.py"` — like
`HF_TOKEN`, it's never written to the shared filesystem.

Useful flags:
- `--project-name NAME` — override the venv/repo directory name (default:
  repo basename)
- `--instance-type NAME` — skip auto-selection, force a specific GPU
- `--pod-id ID` — reuse an already-running pod (e.g. one you kept alive)
  instead of launching a new one; doesn't require `--yes`
- `--keep-alive` — leave the pod running after the command finishes, for
  interactive follow-up; **prints the stop command, run it when you're
  actually done**

## Everyday commands

Run from wherever the current copy of `lambda/` lives — this folder for
account-level bootstrap, or `<project>/infra/lambda/` from inside an
onboarded project:

```bash
python3 lambda/lambda_cli.py list                     # what's running right now
python3 lambda/lambda_cli.py status --id <id>          # detail on one instance
python3 lambda/lambda_cli.py stop --id <id>             # stop one
python3 lambda/lambda_cli.py stop-all --yes              # emergency: stop everything
python3 lambda/lambda_cli.py types --filter a100          # instance types + live capacity + price
```

SSH in directly for interactive debugging (e.g. after `--keep-alive`):
```bash
ssh -A -i ~/.ssh/lambda_id_ed25519 ubuntu@<ip>
source /lambda/nfs/<filesystem>/envs/<project-name>.env
```

## Files

| Path | Purpose |
|---|---|
| `lambda/lambda_cli.py` | Lambda Cloud API client + CLI (stdlib only, no deps) |
| `lambda/config.yaml` | shared defaults: filesystem, region, ssh key, GPU preference order |
| `lambda/config_to_env.py` | tiny config.yaml → shell-env parser used by `run.sh` |
| `pod-env/bootstrap_pod.sh` | runs on the pod: per-project venv + torch + shared HF/uv cache |
| `bin/run.sh` | the orchestrator: launch → sync repo → bootstrap → run → stop |
| `docs/LAMBDA_NOTES.md` | API research notes and gotchas |

## Troubleshooting

- **`find-capacity` fails for every preferred type**: Lambda sold out of
  that tier in your filesystem's region right now. Retry later, or widen
  `preferred_instance_types` in `config.yaml`, or pass `--instance-type`
  once you've checked `lambda_cli.py types --filter <gpu>` for what's
  actually free.
- **`ssh-add` fails / agent forwarding doesn't authenticate to GitHub**:
  make sure an `ssh-agent` is running locally and
  `ssh-add ~/.ssh/lambda_id_ed25519` succeeds before calling `run.sh` — it
  tries this automatically but a passphrase-protected key may need an
  interactive prompt the first time.
  Also confirm the *same* key (or another key tied to your GitHub account)
  is what GitHub expects — the pod authenticates as *you*, through the
  forwarded agent, not as a separate deploy key.
- **A pod is running and you don't know why**: `lambda_cli.py list`, then
  `lambda_cli.py status --id <id>` for detail, then `stop --id <id>` or
  `stop-all --yes`.
- **Different behavior between runs of the same project**: check whether
  the repo's `requirements.lock.txt` changed; `bootstrap_pod.sh` always
  installs from the lock file when one exists, so a stale local venv is
  usually just an out-of-date lock file that needs a fresh
  `rm requirements.lock.txt && bin/run.sh ...` to regenerate.
