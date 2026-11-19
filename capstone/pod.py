#!/usr/bin/env python3
"""Pod lifecycle for the capstone: create with a deadline, run detached jobs, verify teardown.

Adapted from graft-monitors infra/runpod/graft_pod.py. Every resource is named `capstone-*`
(shared team account). State lives in local/capstone/pods/<name>.json (private repo). A
detached local watchdog deletes the Pod at its deadline or after `idle_min` idle minutes, and
`finish` deletes the Pod once every model file is checksum-verified off the Pod. All of this
needs the laptop online; it is not a cloud cap. Default artifact path: `run --hf-prefix` has the
Pod upload its outputs to the private HF dataset itself, so the only credential a Pod receives is
that HF token (over SSH stdin, mode 0600); the RunPod key never leaves the laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra" / "runpod"))
from runpod_cli import (  # noqa: E402
    USER_AGENT,
    ApiError,
    api_key,
    config_quote,
    list_pods,
    public_pod,
    request,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
STATE_DIR = PROJECT / "local" / "capstone" / "pods"
KEY = Path("~/.ssh/id_ed25519").expanduser()
KNOWN_HOSTS = STATE_DIR / "known_hosts"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
JOBS = "/workspace/jobs"
HF_REPO = "ic-org/rlvr-safety-dynamics-shared"


def state_path(name: str) -> Path:
    if not name.startswith("capstone-"):
        raise ValueError("Pod names must start with capstone-")
    return STATE_DIR / f"{name}.json"


def load(name: str) -> dict:
    return json.loads(state_path(name).read_text())


def save(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(state["name"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def emit(stage: str, **fields) -> None:
    print(
        json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, **fields}), flush=True
    )


def dotenv_value(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env = PROJECT / ".env"
    for line in env.read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.strip() == name:
            parts = shlex.split(v, comments=True)
            return parts[0] if parts else None
    return None


def create_pod(body: dict) -> dict:
    """POST /pods, surfacing the provider error body (it carries no secrets)."""
    cfg = (
        "header = " + config_quote("Authorization: Bearer " + api_key()) + "\n"
        'header = "Content-Type: application/json"\n'
        "data-binary = " + config_quote(json.dumps(body)) + "\n"
    )
    r = subprocess.run(
        [
            "curl",
            "--disable",
            "-4",
            "-s",
            "--max-time",
            "60",
            "--config",
            "-",
            "-X",
            "POST",
            "-H",
            f"User-Agent: {USER_AGENT}",
            "-w",
            "\n%{http_code}",
            "https://api.runpod.io/v2/pods",
        ],
        input=cfg,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"},
    )
    payload, _, code = r.stdout.rpartition("\n")
    if not code.isdigit() or int(code) >= 400:
        raise ApiError(
            f"create failed HTTP {code}: {payload[:400]}", int(code) if code.isdigit() else None
        )
    return json.loads(payload)


def gone(path: str) -> bool:
    try:
        request(path)
    except ApiError as exc:
        if exc.status == 404:
            return True
        raise
    return False


def terminate(pod_id: str) -> None:
    path = "/pods/" + pod_id
    try:
        request(path, method="DELETE")
    except ApiError as exc:
        if exc.status != 404:
            raise
    for _ in range(24):
        if gone(path):
            return
        time.sleep(5)
    raise RuntimeError("Pod deletion not verified: " + pod_id)


def ssh_argv(state: dict) -> list[str]:
    c = state.get("ssh")
    if not c:
        raise RuntimeError("No SSH connection recorded")
    return [
        "ssh",
        "-i",
        str(KEY),
        "-p",
        str(c["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        f"{c.get('username', 'root')}@{c['host']}",
    ]


def remote(
    state: dict,
    command: str,
    *,
    input_text: str | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess:
    r = subprocess.run(
        ssh_argv(state) + [command],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and r.returncode:
        raise RuntimeError(f"SSH exit {r.returncode}: {r.stderr[-800:]}")
    return r


def cmd_volume_create(a) -> None:
    if not a.name.startswith("capstone-"):
        raise ValueError("Volume names must start with capstone-")
    existing = [v for v in request("/network-volumes")["networkVolumes"] if v["name"] == a.name]
    if existing:
        emit("volume_exists", **existing[0])
        return
    v = request(
        "/network-volumes",
        method="POST",
        body={"name": a.name, "dataCenter": a.dc, "size": a.size, "type": "STANDARD"},
    )
    emit("volume_created", **v)


def cmd_up(a) -> None:
    path = state_path(a.name)
    if path.exists() and not load(a.name).get("deleted_at"):
        raise RuntimeError(f"{a.name} already has live state; run `down` first")
    env = {"PUBLIC_KEY": KEY.with_suffix(".pub").read_text().strip()}
    body = {
        "name": a.name,
        "cloud": "SECURE",
        "image": a.image,
        "gpu": {"id": a.gpu, "count": a.count, "minCudaVersion": a.min_cuda},
        "dataCenterIds": a.dc.split(","),
        "disk": a.disk,
        "ports": ["22/tcp"],
        "startSsh": True,
        "startJupyter": False,
        "env": env,
    }
    if a.volume:
        body["mounts"] = {"network": [{"volumeId": a.volume, "path": "/workspace"}]}
    state = {
        "name": a.name,
        "gpu": a.gpu,
        "count": a.count,
        "dc": a.dc,
        "image": a.image,
        "volume": a.volume,
        "max_rate": a.max_rate,
        "purpose": a.purpose,
        "requested_at": time.time(),
        "deadline": time.time() + a.hours * 3600,
        "idle_min": a.idle_min,
    }
    save(state)
    try:
        # Never retry a create blindly: recover by unique name on failure.
        pod = create_pod(body)
    except ApiError as exc:
        matches = [p for p in list_pods() if p.get("name") == a.name]
        if not matches:
            state["deleted_at"] = time.time()
            state["create_error"] = str(exc)
            save(state)
            raise
        pod = matches[0]
    state["pod_id"] = pod["id"]
    state["cost_per_hr"] = pod.get("cost")
    state["created_at"] = time.time()
    save(state)
    state["dc"] = pod.get("dataCenterId", state["dc"])
    save(state)
    emit("created", **public_pod(pod))
    rate = state["cost_per_hr"]
    if rate is None or float(rate) > a.max_rate:
        terminate(pod["id"])
        state["deleted_at"] = time.time()
        save(state)
        raise RuntimeError(f"Accepted rate {rate} exceeds ceiling {a.max_rate}; terminated")
    log = (STATE_DIR / f"{a.name}.watchdog.log").open("a")
    guard = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "watchdog", a.name],
        stdout=log,
        stderr=log,
        start_new_session=True,
        cwd=HERE,
    )
    state["watchdog_pid"] = guard.pid
    save(state)
    until = time.time() + a.ready_timeout
    while time.time() < until:
        pod = request("/pods/" + state["pod_id"])
        conn = (pod.get("ssh") or {}).get("direct")
        if pod.get("cost"):
            if float(pod["cost"]) > a.max_rate:
                terminate(state["pod_id"])
                state["deleted_at"] = time.time()
                save(state)
                raise RuntimeError(
                    f"Running rate {pod['cost']} exceeds ceiling {a.max_rate}; terminated"
                )
            state["cost_per_hr"] = rate = pod["cost"]
        if pod.get("status") == "ERROR":
            raise RuntimeError("Pod entered ERROR")
        if conn and conn.get("host") and conn.get("port"):
            state["ssh"] = conn
            save(state)
            if remote(state, "true", check=False, timeout=30).returncode == 0:
                state["ready_at"] = time.time()
                save(state)
                emit(
                    "ready",
                    pod_id=state["pod_id"],
                    ssh=" ".join(ssh_argv(state)),
                    rate=rate,
                    deadline=time.strftime("%H:%M", time.localtime(state["deadline"])),
                )
                return
        time.sleep(10)
    raise RuntimeError("SSH readiness timed out; Pod still running, run `down`")


def cmd_down(a) -> None:
    state = load(a.name)
    ids = {state.get("pod_id")} - {None}
    ids |= {p["id"] for p in list_pods() if p.get("name") == a.name}
    for pod_id in ids:
        terminate(pod_id)
    state["deleted_at"] = time.time()
    if state.get("created_at"):
        hours = (state["deleted_at"] - state["created_at"]) / 3600
        state["billed_hours_est"] = round(hours, 3)
        state["cost_est"] = round(hours * float(state.get("cost_per_hr") or 0), 3)
    save(state)
    emit(
        "deleted",
        name=a.name,
        ids=sorted(ids),
        hours=state.get("billed_hours_est"),
        cost_est=state.get("cost_est"),
    )


def gpu_utilization(state: dict) -> int | None:
    """Highest utilization (%) over the Pod's GPUs, or None when it cannot be read."""
    try:
        r = remote(
            state,
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            timeout=30,
            check=False,
        )
        vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
        return max(vals) if r.returncode == 0 and vals else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def cmd_watchdog(a) -> None:
    """Delete the Pod at its deadline, or once every GPU has been below 5% for `idle_min` minutes.

    The idle rule guards against paying for a finished or stalled job. It only counts samples it
    could read, so an unreachable Pod is left to the deadline. It runs on the laptop: if the
    laptop sleeps, only RunPod billing (`runpod_cli.py pods`) remains as a check.
    """
    idle_since = None
    last_sample = 0.0
    while True:
        state = load(a.name)
        if state.get("deleted_at"):
            return
        reason = "deadline_teardown" if time.time() >= state["deadline"] else None
        idle_min = state.get("idle_min") or 0
        # A finish controller copying results back leaves the GPU idle on purpose.
        pulling = state.get("pulling_since")
        if pulling:
            idle_since = None
        if (
            not reason
            and not pulling
            and idle_min
            and state.get("ready_at")
            and time.time() - last_sample >= 60
        ):
            last_sample = time.time()
            util = gpu_utilization(state)
            if util is None:
                pass
            elif util >= 5:
                idle_since = None
            else:
                idle_since = idle_since or time.time()
                if time.time() - idle_since >= idle_min * 60:
                    reason = "idle_teardown"
        if reason:
            try:
                rsync_pull(state, JOBS, str(STATE_DIR / f"{a.name}-salvage"))
                emit("salvaged_logs", name=a.name)
            except (subprocess.CalledProcessError, OSError) as exc:
                emit("salvage_failed", error=str(exc))
            try:
                cmd_down(a)
                emit(reason, name=a.name)
                return
            except (ApiError, RuntimeError, OSError) as exc:
                emit("teardown_retry", error=str(exc))
        time.sleep(20)


def cmd_extend(a) -> None:
    state = load(a.name)
    state["deadline"] = max(state["deadline"], time.time()) + a.hours * 3600
    save(state)
    emit(
        "extended",
        name=a.name,
        deadline=time.strftime("%Y-%m-%d %H:%M", time.localtime(state["deadline"])),
    )


def cmd_ssh(a) -> None:
    state = load(a.name)
    if a.command:
        r = remote(state, " ".join(a.command), timeout=a.timeout, check=False)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(r.returncode)
    os.execvp("ssh", ssh_argv(state))


def cmd_sync(a) -> None:
    """rsync a local dir to the Pod (code only; secrets, local/ and .docs/ are excluded)."""
    state = load(a.name)
    c = state["ssh"]
    rsh = (
        f"ssh -i {KEY} -p {c['port']} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        f"-o UserKnownHostsFile={KNOWN_HOSTS}"
    )
    # `local` may be a symlink into the central private docs clone at `.docs/`.
    excludes = [
        ".env",
        ".env.*",
        "local/",
        "/local",
        "/.docs/",
        ".git/",
        "__pycache__/",
        ".venv/",
        "*.pyc",
        "AGENTS.md",
        ".agents/",
        "output/",
        "wandb/",
        ".pytest_cache/",
        "*.egg-info/",
    ]
    argv = ["rsync", "-rlptz", "--delete", "-e", rsh]
    for e in excludes:
        argv += ["--exclude", e]
    argv += [a.src.rstrip("/") + "/", f"root@{c['host']}:{a.dst}"]
    subprocess.run(["ssh", *ssh_argv(state)[1:], f"mkdir -p {shlex.quote(a.dst)}"], check=True)
    subprocess.run(argv, check=True)
    emit("synced", src=a.src, dst=a.dst)


def cmd_run(a) -> None:
    """Start a detached job: /workspace/jobs/<job>/{cmd.sh,job.log,exit.code}."""
    state = load(a.name)
    d = f"{JOBS}/{a.job}"
    if a.hf_prefix:
        # Default artifact path: the Pod uploads to the private HF org itself (Aaliyan,
        # 2026-11-18), because Pod-to-laptop copies ran at ~100 kB/s. The token travels over
        # SSH stdin into a 0600 file and is never printed or put on a command line.
        from huggingface_hub import get_token

        remote(
            state,
            "umask 077; mkdir -p /root/.cache/huggingface; cat > /root/.cache/huggingface/token",
            input_text=get_token(),
        )
        remote(state, "cat > /workspace/hf_prefix", input_text=a.hf_prefix)
        state["hf_prefix"] = a.hf_prefix
        save(state)
    script = Path(a.script).read_text()
    remote(state, f"mkdir -p {d} && rm -f {d}/exit.code && cat > {d}/cmd.sh", input_text=script)
    launch = (
        f"cd {d} && setsid -f bash -c 'echo $$ > {d}/pid; timeout {int(a.timeout_h * 3600)} "
        f"bash {d}/cmd.sh > {d}/job.log 2>&1; echo $? > {d}/exit.code' "
        f"</dev/null >/dev/null 2>&1; sleep 2; cat {d}/pid"
    )
    pid = remote(state, launch, timeout=60).stdout.strip()
    state.setdefault("jobs", []).append(
        {"job": a.job, "pid": pid, "started_at": time.time(), "script": a.script}
    )
    save(state)
    emit("job_started", job=a.job, pid=pid, log=f"{d}/job.log")


def cmd_job(a) -> None:
    state = load(a.name)
    d = f"{JOBS}/{a.job}"
    r = remote(
        state,
        f"tail -n {a.lines} {d}/job.log 2>/dev/null; echo '--- exit:'; "
        f"cat {d}/exit.code 2>/dev/null || echo running",
        check=False,
    )
    sys.stdout.write(r.stdout)


def rsync_pull(state: dict, remote_dir: str, local_dir: str, skip_models: bool = False) -> None:
    c = state["ssh"]
    rsh = (
        f"ssh -i {KEY} -p {c['port']} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        f"-o UserKnownHostsFile={KNOWN_HOSTS}"
    )
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-rlpt",
            "--partial",
            "--exclude",
            "optimizer.pt",
            "--exclude",
            "*.safetensors.tmp",
            *(["--exclude", "*.safetensors", "--exclude", "*.bin"] if skip_models else []),
            "-e",
            rsh,
            f"root@{c['host']}:{remote_dir.rstrip('/')}/",
            local_dir.rstrip("/") + "/",
        ],
        check=True,
    )


def verify_artifacts(state: dict, remote_dir: str, local_dir: str) -> list[str]:
    """Model files (*.safetensors, *.bin, *.pt) whose local sha256 is missing or differs."""
    r = remote(
        state,
        f"cd {remote_dir} && find . -type f \\( -name '*.safetensors' -o -name '*.bin' "
        "-o -name '*.pt' \\) ! -name optimizer.pt -exec sha256sum {} +",
        timeout=600,
    )
    bad = []
    for line in r.stdout.splitlines():
        digest, rel = line.split(maxsplit=1)
        path = Path(local_dir) / rel
        local = (
            subprocess.run(
                ["shasum", "-a", "256", str(path)], capture_output=True, text=True
            ).stdout.split(" ")[0]
            if path.exists()
            else ""
        )
        if local != digest:
            bad.append(rel)
    return bad


def verify_hf(state: dict, remote_dir: str, prefix: str) -> list[str]:
    """Model files on the Pod whose sha256 is missing or different on the private HF dataset."""
    from huggingface_hub import HfApi

    r = remote(
        state,
        f"cd {remote_dir} && find . -type f \\( -name '*.safetensors' -o -name '*.bin' "
        "-o -name '*.pt' \\) ! -name optimizer.pt -exec sha256sum {} +",
        timeout=600,
    )
    info = HfApi().dataset_info(HF_REPO, files_metadata=True)
    hf = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    bad = []
    for line in r.stdout.splitlines():
        digest, rel = line.split(maxsplit=1)
        if hf.get(f"{prefix.rstrip('/')}/{rel[2:]}") != digest:
            bad.append(rel)
    return bad


def cmd_pull(a) -> None:
    rsync_pull(load(a.name), a.remote_dir, a.local_dir)
    emit("pulled", src=a.remote_dir, dst=a.local_dir)


def cmd_finish(a) -> None:
    """Copy outputs back while the job runs; after it exits, pull, verify, then delete the Pod.

    The Pod is deleted only once every model file's sha256 matches the Pod's copy. If that
    never happens, the Pod stays up until its deadline (the watchdog's hard cost bound) and
    the failure is logged, so artifacts are never thrown away by an incomplete copy.
    """
    d = f"{JOBS}/{a.job}"
    last_sync = 0.0
    while True:
        state = load(a.name)
        if state.get("deleted_at"):
            emit("already_deleted", name=a.name)
            return
        r = remote(state, f"cat {d}/exit.code 2>/dev/null", check=False, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            break
        if time.time() - last_sync >= a.sync_min * 60:
            try:
                rsync_pull(
                    state, a.remote_dir, a.local_dir, skip_models=bool(state.get("hf_prefix"))
                )
                emit("synced", dst=a.local_dir)
            except subprocess.CalledProcessError as exc:
                emit("sync_failed", error=str(exc))
            last_sync = time.time()
        time.sleep(a.poll)
    emit("job_exited", job=a.job, code=r.stdout.strip())
    state["pulling_since"] = time.time()
    save(state)
    prefix = state.get("hf_prefix")
    bad = ["(not checked)"]
    while not load(a.name).get("deleted_at"):
        try:
            if prefix:
                # Model files go Pod -> HF; only small files come to the laptop.
                remote(
                    state,
                    "cd /workspace/code && HF_HUB_DISABLE_PROGRESS_BARS=1 "
                    "/workspace/venv/bin/python capstone/hf_sync.py",
                    timeout=1800,
                )
                rsync_pull(state, a.remote_dir, a.local_dir, skip_models=True)
                rsync_pull(state, d, a.local_dir.rstrip("/") + "/_job")
                bad = verify_hf(state, a.remote_dir, prefix)
            else:
                rsync_pull(state, a.remote_dir, a.local_dir)
                rsync_pull(state, d, a.local_dir.rstrip("/") + "/_job")
                bad = verify_artifacts(state, a.remote_dir, a.local_dir)
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
            emit("pull_retry", error=str(exc))
        if not bad:
            break
        emit("verify_pending", missing=bad[:10], count=len(bad))
        time.sleep(30)
    if bad:
        emit("artifacts_unverified", name=a.name, missing=bad)
        return
    emit("artifacts_verified", name=a.name)
    state = load(a.name)
    state.pop("pulling_since", None)
    save(state)
    if a.keep:
        emit("kept_alive", name=a.name)
        return
    cmd_down(a)


def cmd_list(_a) -> None:
    pods = [public_pod(p) for p in list_pods() if p.get("name", "").startswith("capstone-")]
    vols = [
        v
        for v in request("/network-volumes")["networkVolumes"]
        if v["name"].startswith("capstone-")
    ]
    print(json.dumps({"pods": pods, "volumes": vols}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="cmd", required=True)
    v = s.add_parser("volume-create")
    v.add_argument("--name", required=True)
    v.add_argument("--dc", required=True)
    v.add_argument("--size", type=int, required=True)
    u = s.add_parser("up")
    u.add_argument("name")
    u.add_argument("--gpu", required=True)
    u.add_argument("--count", type=int, default=1)
    u.add_argument(
        "--dc", required=True, help="one data center, or a comma list (first with capacity)"
    )
    u.add_argument("--volume")
    u.add_argument(
        "--hours",
        type=float,
        required=True,
        help="deadline after which the watchdog deletes the Pod",
    )
    u.add_argument(
        "--max-rate", type=float, required=True, help="terminate if the accepted $/h exceeds this"
    )
    u.add_argument("--purpose", required=True)
    u.add_argument(
        "--idle-min",
        type=float,
        default=20.0,
        help="delete the Pod after this many minutes with every GPU below 5%% (0 disables)",
    )
    u.add_argument("--image", default=IMAGE)
    u.add_argument("--disk", type=int, default=60)
    u.add_argument(
        "--min-cuda",
        default="12.8",
        help="13.0 for Pods that serve AnyJev (vLLM 0.29 cu130 wheels)",
    )
    u.add_argument("--ready-timeout", type=int, default=900)
    for name in ("down", "watchdog"):
        s.add_parser(name).add_argument("name")
    e = s.add_parser("extend")
    e.add_argument("name")
    e.add_argument("--hours", type=float, required=True)
    sh = s.add_parser("ssh")
    sh.add_argument("name")
    sh.add_argument("--timeout", type=int, default=600)
    sh.add_argument("command", nargs=argparse.REMAINDER)
    sy = s.add_parser("sync")
    sy.add_argument("name")
    sy.add_argument("src")
    sy.add_argument("dst")
    r = s.add_parser("run")
    r.add_argument("name")
    r.add_argument("job")
    r.add_argument("script")
    r.add_argument("--timeout-h", type=float, default=8)
    r.add_argument(
        "--hf-prefix",
        help="path in ic-org/rlvr-safety-dynamics-shared the Pod uploads /workspace/out to",
    )
    j = s.add_parser("job")
    j.add_argument("name")
    j.add_argument("job")
    j.add_argument("--lines", type=int, default=30)
    pl = s.add_parser("pull")
    pl.add_argument("name")
    pl.add_argument("remote_dir")
    pl.add_argument("local_dir")
    fi = s.add_parser("finish")
    fi.add_argument("name")
    fi.add_argument("job")
    fi.add_argument("remote_dir")
    fi.add_argument("local_dir")
    fi.add_argument("--poll", type=int, default=60)
    fi.add_argument(
        "--sync-min", type=float, default=5.0, help="minutes between copies while the job runs"
    )
    fi.add_argument("--keep", action="store_true", help="pull but leave the Pod running")
    s.add_parser("list")
    a = p.parse_args()
    {
        "volume-create": cmd_volume_create,
        "up": cmd_up,
        "down": cmd_down,
        "watchdog": cmd_watchdog,
        "extend": cmd_extend,
        "ssh": cmd_ssh,
        "sync": cmd_sync,
        "run": cmd_run,
        "job": cmd_job,
        "list": cmd_list,
        "pull": cmd_pull,
        "finish": cmd_finish,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
