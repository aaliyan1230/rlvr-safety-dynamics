# Lambda Cloud API — operational notes

Findings from building this toolkit (Aug 2026), things that aren't obvious
from a first skim of the docs and matter for not wasting money.

## Auth

`https://cloud.lambdalabs.com/api/v1`, HTTP Basic auth, API key as username,
blank password:
```
curl -H 'User-Agent: OpenAI File Downloader, XaiImageApiFetch/1.0' \
  -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances
```
Keys are created/revoked at https://cloud.lambda.ai/api-keys. `lambda_cli.py`
reads `LAMBDA_API_KEY` from the environment and never writes it to disk.

**Use curl, not Python's urllib/requests.** Confirmed live 2026-08-15:
Lambda's Cloudflare front door 403s Python's default TLS ClientHello
fingerprint with error 1010 (a JA3 bot-fingerprint block), regardless of
headers set. The exact same API key against the exact same endpoint works
fine via `curl` and fails via `urllib.request`. `lambda_cli.py` shells out
to `curl` for this reason — don't "simplify" it back to a pure-Python HTTP
client without retesting against the live API first.

**API keys are scoped to a workspace, and the console can be misleading
about which one is active.** Hit this live: created a filesystem while the
console's team switcher showed team "X", generated an API key while it also
showed team "X", and `GET /file-systems` still returned `{"data": []}` for
that key — even a freshly-generated key made no difference. Root cause: the
key was scoped to the account's **default workspace**, not the named team,
regardless of what the console UI displayed as "active." A filesystem
created directly in the default workspace was immediately visible. If
`file-systems`/`instances` mysteriously don't show something you can see in
the console: **don't trust the team switcher** — try recreating the
resource without selecting any team at all, or explicitly check which
workspace a `ssh-keys` list entry's `workspace_id` corresponds to before
concluding it's an API bug.

## Billing

- **Per-minute granularity**, quoted as a per-GPU-hour rate. Billing starts
  the instant `POST /instance-operations/launch` succeeds and stops the
  instant `POST /instance-operations/terminate` succeeds — nothing in
  between (idle, booting, crashed) pauses it.
- **Never run `sudo shutdown -h now` / `systemctl poweroff` inside the pod
  to "turn it off."** Lambda's own docs warn this puts the instance into an
  Alert state and **billing continues**. The only way to actually stop
  billing is the `terminate` API call — which is exactly what
  `lambda_cli.py stop` / `bin/run.sh`'s exit trap do. Don't add any
  in-pod shutdown shortcuts to this toolkit.

## Filesystems (persistent storage)

- A filesystem is created in **one region** and **cannot move or be
  recreated in another region later** — the region choice made during
  first-time bootstrap (see README) is effectively permanent.
- A filesystem **can only be attached at instance launch time** — you
  cannot attach one to an already-running instance. If you forget
  `--filesystem` at launch, terminate and relaunch.
- As of writing, only **one filesystem per launched instance** is
  supported (`file_system_names` accepts at most one entry).
- Mounts at `/lambda/nfs/<filesystem-name>` on the instance.
- Any instance type launched in the filesystem's region can attach it —
  the constraint is region match, not instance-type match. So the GPU tier
  used for any given run can vary; only the region is fixed.
- Files moved to a `.Trash-*` directory on the filesystem still count
  toward billed storage. Actually `rm` things you don't want kept.

## Capacity is volatile

Lambda sells out of popular GPU tiers (A100, H100) in specific regions
often. `GET /instance-types` reports live `regions_with_capacity_available`
per type — always check this right before launching rather than assuming a
type/region combo that worked yesterday still has room today.
`lambda_cli.py find-capacity --region <fixed-region> --prefer <tiers...>`
automates trying a priority list of GPU tiers within the filesystem's fixed
region and picks the first one with room.

## SSH keys

- Exactly one `ssh_key_names` entry is required per launch (not zero, not
  many).
- Register a key with `POST /ssh-keys` (`lambda_cli.py add-ssh-key`) or via
  the console. Default OS login user on Lambda's stock Ubuntu image is
  `ubuntu`.

## Code sync: SSH agent forwarding, not a persisted token

`bin/run.sh` SSHes in with `-A` (agent forwarding) and clones over
`git@github.com:...` from inside the pod. This means GitHub auth flows
through your laptop's already-loaded SSH key for the duration of that one
SSH session and **nothing GitHub-credential-shaped ever touches the pod's
disk** — consistent with how `HF_TOKEN` is handled (exported fresh each
session, never persisted). This matters more here than it did for the
single-project RunPod setup because this filesystem is meant to be shared
across many research repos, some possibly private to different orgs.

## Endpoints used by `lambda_cli.py`

| Method | Path | Used for |
|---|---|---|
| GET | `/instance-types` | capacity check, pricing lookup |
| GET | `/ssh-keys` | list registered keys |
| POST | `/ssh-keys` | register a new key |
| GET | `/file-systems` | list persistent filesystems + their region |
| POST | `/instance-operations/launch` | launch |
| GET | `/instances` | list running/booting instances |
| GET | `/instances/{id}` | poll status/IP |
| POST | `/instance-operations/terminate` | stop (the only thing that stops billing) |

Full reference: https://docs.lambda.ai/api/cloud

## Confirmed working end-to-end (2026-08-15)

Full pipeline validated live on a real account, real pod: capacity check →
launch (`gpu_1x_a100_sxm4`, us-east-1, filesystem attached) → SSH →
`bootstrap_pod.sh` (fresh venv, torch+CUDA, package install, lock file
generation) → loaded `Qwen/Qwen2.5-7B-Instruct` in bf16 → correct generation
→ terminate. 15.2GB peak GPU memory (comfortable on the 40GB card), ~230s
model load (mostly weight-loading into GPU, not download - HF fetch itself
was ~38s for the sharded safetensors files). No single-GPU 80GB A100 exists
in Lambda's lineup (see `config.yaml` comments) - 40GB is the realistic
single-GPU ceiling for now.
