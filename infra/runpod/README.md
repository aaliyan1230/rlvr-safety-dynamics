# RunPod GPU workflow

RunPod is this project's sole configured GPU provider. The REST v2 inspection
client and bounded acceptance pilot are maintained here. The October 1 live
pilot passed provisioning, 7B inference, detached progress polling, replacement
Pod persistence, failure detection, and verified resource deletion. It does not
validate arbitrary scientific workloads or unattended recovery after Mac failure.

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

## Before a paid workload

Inspect RunPod-native account-wide billing and credit balance. Estimate maximum
compute, storage, and tax exposure in the launch plan, including concurrent runs.
Use native billing/invoices as spending authority; no local transaction ledger
or reservation database is required. Obtain explicit approval after quoting
the GPU/count, hourly rate, maximum runtime, image, and volume/location. Catalog
availability and prices are advisory; verify the accepted quote and CUDA access.
Credit purchases and consumed usage are separate records. Disabled auto-pay and
an hourly spend limit are not cumulative project-dollar caps.

The pilot below is the supported automated acceptance probe. A general experiment
runner, dependency-lock enforcement, off-provider backups, and a deadline
controller independent of the laptop remain acceptance work before unattended
research jobs. Preserve exact model/checkpoint, prompt, runtime, generation, and
scoring provenance when implementing those jobs. Use direct SSH with port 22,
registered public keys, and sshd in the selected image. Network volumes are tied
to a data center and continue billing after Pod deletion; retrieve outputs and
explicitly verify volume cleanup. A remote workload timeout alone does not stop
cloud billing. Never retry an ambiguous create request blindly.

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
Obtain authorization after comparing estimated compute/storage exposure with
native spending, available credit and the project allocation before running it:

```bash
python3 infra/runpod/pilot.py run --yes --datacenter EUR-IS-1 \
  --private-key /absolute/path/to/your/ssh-private-key \
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
