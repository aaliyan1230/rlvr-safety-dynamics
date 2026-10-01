# RunPod REST v2 migration foundation

This project's stdlib-only CLI uses the RunPod REST v2 control plane and the
required User-Agent. It currently provides **read-only** inventory, capacity,
connection inspection, and billing reconciliation. It does not launch, fund,
stop, or delete resources. Lambda remains available in `infra/lambda/`.

Store `RUNPOD_API_KEY` in the ignored project `.env` with mode `600`, or export
it for the CLI process. The CLI reads only that variable and does not execute
environment-file contents. Credentials are sent to curl through stdin, never
command arguments. Pod environment variables are omitted from CLI output.

```bash
python3 infra/runpod/runpod_cli.py pods
python3 infra/runpod/runpod_cli.py volumes
python3 infra/runpod/runpod_cli.py gpus --filter A100 --min-cuda-version 12.8
python3 infra/runpod/runpod_cli.py status POD_ID
python3 infra/runpod/runpod_cli.py billing \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-10-01T23:59:59Z
```

The GPU catalog is constrained to one GPU on Secure Cloud. Its prices are
catalog rates, not an accepted launch quote. Availability is advisory; probe
actual GPU access after launch. Inventory follows pagination and rejects
malformed responses rather than interpreting them as an empty account.

## Remaining acceptance gates

Before implementing or using paid provisioning, reconcile usage across both
providers and reserve the maximum run cost in the private ledger. Keep the $950
allocation and $600 review checkpoint. Do not count a prepaid-credit purchase
again as consumed GPU usage; record cash purchases separately from utilization.

The next stage must provide a reviewed launch plan with explicit GPU ID/count,
price ceiling, runtime deadline, image, and volume/location. Obtain launch
approval after quoting the concrete GPU and hourly rate. Preserve known resource
IDs before waiting; do not retry an ambiguous create request blindly. Verify
teardown independently. A local process timeout does not survive laptop failure
and is not a provider-enforced cap.

Use direct SSH with port 22 mapped, registered public keys, and an image running
sshd. Keep repos, per-project environments, model caches, progress records, and
outputs on a network volume. Verify retrieval after Pod deletion and off-provider
backups. Rebuild environments from pinned dependencies; do not copy a Lambda venv.

Run a benign development pilot covering CUDA access, inference, progress polling,
failure cleanup, and persistence before changing confirmatory experiments. Keep
runtime, model, tokenizer, generation, and scoring provenance exact.

## API transition

RunPod documents retirement of REST v1 on November 15, 2026. The inspected
`runpodctl` v2.14.0 release still uses v1 for CRUD operations. Do not replace this
client with an older CLI/tutorial without verifying its current API behavior.

- [REST v2 migration](https://docs.runpod.io/api-reference-v2/migrate-from-v1)
- [REST v2 schema](https://docs.runpod.io/api-reference-v2/openapi.json)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [Billing](https://docs.runpod.io/accounts-billing/billing)

Account/payment observations and migration decisions belong only in `local/`.

## Bounded migration pilot

`pilot.py` is a specific acceptance probe, separate from the read-only inspection
CLI. It creates up to two sequential one-GPU A100 SXM 80 GB Secure Cloud Pods
and a temporary 50 GB STANDARD network volume. Its catalog ceiling is $1.59/hour;
it terminates immediately if the accepted Pod quote exceeds $1.69/hour.
Obtain authorization and reserve the combined compute/storage exposure in the
private ledger before running it:

```bash
python3 infra/runpod/pilot.py run --yes --datacenter EUR-IS-1 \
  --output /absolute/path/to/ignored-private-pilot-record
```

The probe registers the local SSH **public** key while preserving existing keys,
clones this public repo, builds a volume-backed environment, loads a fixed
Qwen2.5-7B-Instruct revision in bf16, and checks a benign deterministic answer.
Dependencies, prompt, generation settings, GPU, timing, and peak GPU memory are
recorded. Each progress poll opens a fresh direct SSH connection to a detached
job. The second Pod verifies that the result's SHA-256 survived the first Pod's
deletion. An intentional exit-7 command verifies failure detection. Results and
logs are copied locally before final resource deletion. No cloud API credential
is forwarded to either Pod. The temporary volume is deleted at the end; this
probe does not establish a permanent model cache.

The overall deadline is one hour, including boot and download overhead, with a
30-minute workload timeout. `finally` cleanup runs on ordinary failures. A
separate detached **local** watchdog also recovers uniquely named resources,
including an ambiguous create response. It survives the controlling process
exiting but requires the Mac and network to remain available. It is not a cloud
deadline or a guaranteed dollar cap. Verify deletion before ending a session:

```bash
python3 infra/runpod/pilot.py cleanup --state /absolute/path/to/pilot/state.json
python3 infra/runpod/runpod_cli.py pods
python3 infra/runpod/runpod_cli.py volumes
```

Only resources recorded by that pilot or matching its unique run name are
removed. Do not use this as a general experiment runner: dataset execution,
lock-file enforcement, external watchdogs, and scientific runtime equivalence
remain separate acceptance work. A successful arithmetic probe is infrastructure
validation, not a study result.
