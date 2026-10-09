#!/usr/bin/env python3
"""Prepare, preflight and run a permission experiment bundle on one supervised RunPod GPU.

- ``prepare``   build a deterministic bundle from public model and image metadata;
                no paid resources.
- ``preflight`` check bundle, spending envelope, empty inventory, billing, price and capacity.
                Creates nothing.
- ``run``       preflight, launch one Pod (container disk only, no volumes), run the bundle,
                retrieve and verify results, delete the Pod, verify the account is empty.

Credentials never reach the Pod. Supervised use only: the local watchdog needs this laptop.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pilot
from runpod_cli import USER_AGENT, ApiError, list_pods, public_pod, request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from rlvr_safety.io import sha256_file  # noqa: E402
from rlvr_safety.permission.experiment import (  # noqa: E402
    CONTROLLER_FILES,
    check_envelope,
    immutable_image,
    write_bundle,
)
from rlvr_safety.provenance import verify_manifest  # noqa: E402

AVAILABILITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
SETUP_AND_RETRIEVAL_SECONDS = 900


def controller_hashes() -> dict:
    """SHA-256 of each local controller file so a record names the exact controller used."""
    return {name: sha256_file(HERE / name) for name in CONTROLLER_FILES}


def ssh(connection, args, command, *, data=None, timeout=45, check=True):
    argv = pilot.ssh_command(connection, args.private_key, args.output / "known_hosts")
    argv[1:1] = ["-F", "/dev/null"]
    result = subprocess.run(
        argv + [command],
        input=data,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={
            k: v
            for k, v in os.environ.items()
            if k not in {"RUNPOD_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
        },
    )
    if check and result.returncode:
        diagnostic = args.output / "ssh-failure.json"
        diagnostic.write_text(
            json.dumps(
                {
                    "command": command,
                    "exit": result.returncode,
                    "stderr": result.stderr.decode(errors="replace")[-8000:],
                },
                indent=2,
            )
            + "\n"
        )
        diagnostic.chmod(0o600)
        raise RuntimeError(f"SSH failed with exit {result.returncode}")
    return result


def extract_results(data: bytes, output: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("unsafe result archive member")
        for member in members:
            target = output / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())


def verify_archive(archive_path: Path, bundle: Path) -> None:
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    expected = {*manifest["files"], "bundle_manifest.json"}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != expected:
            raise ValueError("archive contents do not match prepared bundle")
        for member in members:
            if (
                not member.isfile()
                or archive.extractfile(member).read() != (bundle / member.name).read_bytes()
            ):
                raise ValueError("archive contents do not match prepared bundle")


def bootstrap(root: str) -> str:
    q = shlex.quote(root)
    return f"""set -eu
cd {q}
export PYTHONPATH={q}/src HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
python3 -m venv --system-site-packages venv
venv/bin/python - <<'INSTALL'
from pip._vendor import requests
original = requests.Session.send
def send(self, request, **kwargs):
    request.headers['User-Agent'] = '{USER_AGENT}'
    return original(self, request, **kwargs)
requests.Session.send = send
from pip._internal.cli.main import main
raise SystemExit(main(['install', '--disable-pip-version-check',
                       '-c', 'runtime-lock.txt', '-r', 'runtime-pins.txt']))
INSTALL
venv/bin/python permission_workload.py --bundle {q}
"""


def extract_bundle_command(root: str) -> str:
    return f"""python3 - <<'EXTRACT'
import tarfile
from pathlib import Path
root = Path({root!r})
with tarfile.open(root / 'bundle.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or '..' in path.parts or not member.isfile():
            raise ValueError('unsafe prepared archive member')
    for member in members:
        target = root / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
EXTRACT
"""


def choose_datacenter(launch: dict) -> dict:
    """Pick the allowed data center with the best availability for the approved GPU and price."""
    catalog = request(
        "/catalog/gpus",
        {
            "include": "AVAILABILITY",
            "product": "POD",
            "cloud": launch["cloud"],
            "count": launch["gpu_count"],
            "minCudaVersion": launch["min_cuda_version"],
        },
    )
    gpu = next((g for g in catalog["gpus"] if g["id"] == launch["gpu"]), None)
    if gpu is None:
        raise RuntimeError(f"GPU {launch['gpu']} is not in the catalog")
    price = gpu["price"][launch["cloud"].lower()]
    if price > launch["hourly_ceiling"]:
        raise RuntimeError(f"catalog price ${price}/h exceeds the approved ceiling")
    options = [
        d
        for d in gpu.get("dataCenters", [])
        if d["id"] in launch["datacenters"] and d.get("availability", "NONE") != "NONE"
    ]
    if not options:
        raise RuntimeError("approved GPU has no capacity in the allowed data centers")
    best = max(options, key=lambda d: AVAILABILITY_RANK.get(d["availability"], 0))
    return {"datacenter": best["id"], "price": price, "availability": best["availability"]}


def spent_under_envelope(envelope: dict, today: date) -> float:
    """Native usage since the envelope's window start, minus the usage recorded at its creation."""
    data = request(
        "/billing",
        {
            "startTime": envelope["usage_window_start"],
            "endTime": (
                datetime.combine(today, datetime.min.time(), UTC) + timedelta(days=1)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bucketSize": "day",
        },
    )
    total = data["metadata"]["totals"]["totalAmount"]
    return max(total - envelope["usage_baseline_usd"], 0.0)


def preflight(args) -> dict:
    """Everything that must hold before paid resources are created. Creates nothing."""
    bundle = args.bundle
    verify_manifest(bundle / "bundle_manifest.json")
    archive = bundle.parent / (bundle.name + ".tar.gz")
    expected = (bundle.parent / (bundle.name + ".tar.gz.sha256")).read_text().strip()
    if sha256_file(archive) != expected:
        raise ValueError("bundle archive changed")
    verify_archive(archive, bundle)
    spec = json.loads((bundle / "experiment.json").read_text())
    launch = spec["launch"]
    image_ref = immutable_image(launch)
    hashes = controller_hashes()
    if spec.get("controller_sha256") != hashes:
        raise ValueError("controller changed since bundle preparation; prepare a new bundle")
    envelope = json.loads((args.root / spec["envelope"]).read_text())
    today = date.today()
    spent = spent_under_envelope(envelope, today)
    checked = check_envelope(launch, envelope, spent, today)
    pods, volumes = list_pods(), request("/network-volumes")["networkVolumes"]
    problems = list(checked["problems"])
    if pods or volumes:
        problems.append("account has resources; review them before launching")
    if args.balance_usd < launch["max_launch_usd"] * 2:
        problems.append("recorded balance is below twice the maximum launch cost")
    placement = None
    try:
        placement = choose_datacenter(launch)
    except RuntimeError as exc:
        problems.append(str(exc))
    return {
        "experiment_id": spec["experiment_id"],
        "bundle_sha256": expected,
        "controller_sha256": hashes,
        "image": {
            "ref": launch["image"],
            "digest": launch["image_digest"],
            "immutable_ref": image_ref,
        },
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "balance_usd": args.balance_usd,
        "balance_source": args.balance_source,
        "usage_spent_under_envelope_usd": spent,
        "inventory": {"pods": [p["id"] for p in pods], "volumes": [v["id"] for v in volumes]},
        "placement": placement,
        "envelope_warnings": checked["warnings"],
        "problems": problems,
        "ok": not problems,
    }


def run(args) -> None:
    if not args.yes:
        raise ValueError("launch requires explicit approval and --yes")
    report = preflight(args)
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.chmod(0o700)
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["ok"]:
        raise RuntimeError(f"preflight failed: {report['problems']}")
    bundle = args.bundle
    spec = json.loads((bundle / "experiment.json").read_text())
    launch = spec["launch"]
    deadline = min(
        time.time() + launch["max_minutes"] * 60, getattr(args, "deadline", None) or float("inf")
    )
    if deadline - time.time() < 180:
        raise ValueError("approved overall deadline leaves insufficient launch/cleanup time")
    workload_seconds = int(launch["max_minutes"] * 60 - SETUP_AND_RETRIEVAL_SECONDS)
    archive = bundle.parent / (bundle.name + ".tar.gz")
    expected = report["bundle_sha256"]
    if not args.private_key.is_file() or not args.private_key.with_suffix(".pub").is_file():
        raise ValueError("explicit local SSH private/public key pair required")
    state_path = args.output / "state.json"
    state = {
        "name": "rlvr-perm-" + uuid.uuid4().hex[:10],
        "experiment_id": spec["experiment_id"],
        "pod_ids": [],
        "started_at": time.time(),
        "deadline": deadline,
        "bundle_sha256": expected,
        "controller_sha256": controller_hashes(),
        "launch": launch,
        "placement": report["placement"],
        "supervised_only": True,
        "passed": False,
        "outputs_retrieved": False,
        "cleanup_verified": False,
    }
    pilot.save(state_path, state)
    with (args.output / "watchdog.log").open("w") as log:
        guard = subprocess.Popen(
            [sys.executable, str(HERE / "pilot.py"), "watchdog", "--state", str(state_path)],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    state["watchdog_pid"] = guard.pid
    pilot.save(state_path, state)
    connection = None
    remote_root = "/workspace/" + state["name"]
    workload_exit = None
    retrieval_error = None
    try:
        public_key = args.private_key.with_suffix(".pub").read_text().strip()
        keys = request("/account/ssh-keys")["keys"]
        if public_key not in keys:
            updated = request("/account/ssh-keys", method="PUT", body={"keys": keys + [public_key]})
            if public_key not in updated["keys"]:
                raise RuntimeError("SSH public key registration was not confirmed")
        pod = request(
            "/pods",
            method="POST",
            body={
                "name": state["name"] + "-1",
                "cloud": launch["cloud"],
                "image": immutable_image(launch),
                "gpu": {
                    "id": launch["gpu"],
                    "count": launch["gpu_count"],
                    "minCudaVersion": launch["min_cuda_version"],
                },
                "dataCenterIds": [report["placement"]["datacenter"]],
                "disk": launch["container_disk_gb"],
                "ports": ["22/tcp"],
                "startSsh": True,
                "startJupyter": False,
            },
        )
        state["pod_ids"].append(pod["id"])
        state["pod"] = public_pod(pod)
        pilot.save(state_path, state)
        if pod.get("cost", float("inf")) > launch["hourly_ceiling"]:
            raise RuntimeError("accepted GPU quote exceeds approved ceiling")
        ready_until = min(state["deadline"], time.time() + 600)
        while time.time() < ready_until:
            pod = request("/pods/" + pod["id"])
            if pod["status"] == "ERROR":
                raise RuntimeError("Pod entered ERROR")
            connection = (pod.get("ssh") or {}).get("direct")
            if connection and ssh(connection, args, "true", check=False).returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError("SSH readiness deadline expired")
        ssh(
            connection,
            args,
            f"mkdir -p {remote_root}; cat > {remote_root}/bundle.tar.gz",
            data=archive.read_bytes(),
            timeout=180,
        )
        remote_hash = ssh(connection, args, f"sha256sum {remote_root}/bundle.tar.gz").stdout
        if remote_hash.decode().split()[0] != expected:
            raise RuntimeError("uploaded archive checksum mismatch")
        ssh(connection, args, extract_bundle_command(remote_root))
        ssh(
            connection,
            args,
            f"cat > {remote_root}/bootstrap.sh",
            data=bootstrap(remote_root).encode(),
        )
        job = (
            f"timeout {workload_seconds} bash {remote_root}/bootstrap.sh "
            f"> {remote_root}/job.log 2>&1; echo $? > {remote_root}/exit.code"
        )
        pid = ssh(
            connection,
            args,
            f"nohup bash -c {shlex.quote(job)} </dev/null >/dev/null 2>&1 & echo $!",
        )
        state["workload_pid"] = pid.stdout.decode().strip()
        pilot.save(state_path, state)
        until = min(state["deadline"] - 90, time.time() + workload_seconds + 30)
        while time.time() < until:
            result = ssh(connection, args, f"cat {remote_root}/exit.code 2>/dev/null", check=False)
            if result.returncode == 0:
                workload_exit = int(result.stdout.decode().strip())
                break
            progress = ssh(
                connection,
                args,
                f"tail -n 1 {remote_root}/out/progress.jsonl 2>/dev/null",
                check=False,
            )
            pilot.emit("permission_run_progress", output=progress.stdout.decode()[-1200:])
            time.sleep(15)
        state["workload_exit"] = workload_exit
        if workload_exit != 0:
            raise RuntimeError(f"workload incomplete or failed (exit {workload_exit})")
        state["workload_completed"] = True
    except ApiError as exc:
        state["api_error"] = {"status": exc.status, "problem": exc.problem}
        raise
    finally:
        try:
            if connection:
                exported = ssh(
                    connection,
                    args,
                    f"mkdir -p {remote_root}/out; cp {remote_root}/job.log {remote_root}/out/ "
                    f"2>/dev/null || true; tar -czf - -C {remote_root}/out .",
                    timeout=300,
                    check=False,
                )
                if exported.returncode == 0:
                    (args.output / "retrieved-output.tar.gz").write_bytes(exported.stdout)
                    extract_results(exported.stdout, args.output / "retrieved")
                    for manifest in sorted(
                        (args.output / "retrieved").glob("*/benchmark/artifact_manifest.json")
                    ):
                        verify_manifest(manifest)
                    state["outputs_retrieved"] = True
                    if workload_exit == 0:
                        summary_path = args.output / "retrieved/run_summary.json"
                        if not summary_path.is_file():
                            raise RuntimeError("completed workload has no run summary")
                        summary = json.loads(summary_path.read_text())
                        if not summary.get("completed"):
                            raise RuntimeError("retrieved run summary reports an incomplete run")
                        state["passed"] = True
                else:
                    state["outputs_retrieved"] = False
                    if workload_exit == 0:
                        raise RuntimeError("completed workload results could not be retrieved")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            state["passed"] = False
            retrieval_error = exc
        finally:
            pilot.save(state_path, state)
            pilot.cleanup(state)
            state["cleanup_verified"] = True
            state["finished_at"] = time.time()
            pilot.save(state_path, state)
            pilot.emit(
                "permission_run_cleanup_verified",
                pod_ids=state["pod_ids"],
                elapsed_seconds=state["finished_at"] - state["started_at"],
            )
            postflight(args, state)
        if retrieval_error is not None:
            raise retrieval_error


def postflight(args, state: dict) -> dict:
    """Independent check that the account is empty, plus a billing snapshot (which may lag)."""
    pods, volumes = list_pods(), request("/network-volumes")["networkVolumes"]
    record = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pods": [p["id"] for p in pods],
        "volumes": [v["id"] for v in volumes],
        "empty": not pods and not volumes,
        "elapsed_seconds": state["finished_at"] - state["started_at"],
        "billing_may_lag": True,
    }
    spec = json.loads((args.bundle / "experiment.json").read_text())
    try:
        envelope = json.loads((args.root / spec["envelope"]).read_text())
        record["usage_spent_under_envelope_usd"] = spent_under_envelope(envelope, date.today())
    except (ApiError, OSError, KeyError, ValueError) as exc:
        record["billing_error"] = str(exc)
    (args.output / "postflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["empty"]:
        raise RuntimeError(f"resources remain after cleanup: {record['pods'] + record['volumes']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    for name in ("preflight", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--bundle", type=Path, required=True)
        cmd.add_argument("--balance-usd", type=float, required=True)
        cmd.add_argument("--balance-source", required=True, help="where the balance was read")
    sub.choices["run"].add_argument("--private-key", type=Path, required=True)
    sub.choices["run"].add_argument("--output", type=Path, required=True)
    sub.choices["run"].add_argument("--deadline", type=float)
    sub.choices["run"].add_argument("--yes", action="store_true")
    for cmd in sub.choices.values():
        cmd.add_argument("--root", type=Path, default=PROJECT)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(write_bundle(args.spec, args.out_dir, root=args.root), indent=2))
    elif args.command == "preflight":
        report = preflight(args)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ok"] else 1)
    else:
        run(args)


if __name__ == "__main__":
    main()
