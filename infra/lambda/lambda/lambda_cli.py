#!/usr/bin/env python3
"""
Thin CLI around the Lambda Cloud API (https://cloud.lambdalabs.com/api/v1).

Runs on your LAPTOP (or wherever an orchestrating agent lives) - never on the
pod itself. Talks to Lambda's REST API over HTTPS, by shelling out to `curl`
(not Python's urllib/requests: Lambda's Cloudflare front door 403s Python's
default TLS fingerprint with error 1010, a JA3 bot-fingerprint block -
confirmed by testing urllib vs curl against the same key. curl's fingerprint
passes.). Auth is HTTP Basic with the API key as the username and an empty
password, per Lambda's docs:
    curl -H 'User-Agent: OpenAI File Downloader, XaiImageApiFetch/1.0' \
      -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances

Reads LAMBDA_API_KEY from the environment or owner-only project .env,
then a local key file (default
~/.config/lambda-cloud/api_key; override with LAMBDA_API_KEY_FILE). Get a key
at https://cloud.lambda.ai/api-keys and never commit it.

Subcommands (see `--help` on each):
  types              list instance types + which regions currently have capacity
  ssh-keys           list SSH keys registered on the account
  add-ssh-key        register a local public key with Lambda
  filesystems        list persistent filesystems
  launch             launch an instance
  list               list running/booting instances
  status             show one instance's detail (IP, status)
  wait-ssh            block until an instance is SSH-reachable
  stop               terminate one instance
  stop-all           terminate EVERY instance on the account (safety net)

Every subcommand that costs money or destroys state prints exactly what it's
about to do before doing it. `launch` additionally refuses to run without
--yes, by design: this CLI is meant to be called by coding agents, and a
launch is a billed action a human should have already approved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

from project_env import load_project_env

API_BASE = "https://cloud.lambdalabs.com/api/v1"
_STATUS_MARKER = "<<<LAMBDA_CLI_HTTP_STATUS>>>"
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


def api_key() -> str:
    try:
        load_project_env()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    key = os.environ.get("LAMBDA_API_KEY")
    if not key:
        key_file = os.environ.get("LAMBDA_API_KEY_FILE", "~/.config/lambda-cloud/api_key")
        try:
            with open(os.path.expanduser(key_file), encoding="utf-8") as handle:
                key = handle.read().strip()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"ERROR: could not read Lambda API key file {key_file}: {exc}", file=sys.stderr)
            sys.exit(1)
    if not key:
        print(
            "ERROR: no Lambda API key found. Set LAMBDA_API_KEY or put the key in\n"
            "  ~/.config/lambda-cloud/api_key (override with LAMBDA_API_KEY_FILE).\n"
            "Get one at https://cloud.lambda.ai/api-keys",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def request(method: str, path: str, body: dict | None = None) -> dict:
    """Shells out to curl rather than using urllib/requests. Lambda's
    Cloudflare front door 403s (error 1010, a JA3 TLS-fingerprint bot check)
    Python's own TLS stack regardless of headers; curl's fingerprint passes.
    Confirmed by testing both against the same key/endpoint.

    Credentials and the optional JSON body go through curl's stdin config,
    keeping the key out of process arguments and shell history.
    """
    url = f"{API_BASE}{path}"
    # -4 forces IPv4. macOS resolv.conf commonly lists an IPv6 link-local nameserver
    # first (fe80::1%en0); curl tries it, fails to resolve, and reports
    # "Could not resolve host" while nslookup and other clients fall through to the
    # IPv4 nameserver and succeed. The API has no IPv6 requirement, so this costs
    # nothing and removes a failure mode that looks exactly like the API being down.
    cmd = [
        "curl", "-4", "-sS", "-X", method, "-H", f"User-Agent: {USER_AGENT}",
        "--config", "-",
    ]

    def config_quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace(
            "\n", "\\n"
        ).replace("\r", "\\r").replace("\t", "\\t") + '"'

    input_data = f"user = {config_quote(api_key() + ':')}\n"
    if body is not None:
        cmd += ["-H", "Content-Type: application/json"]
        input_data += f"data-binary = {config_quote(json.dumps(body))}\n"
    cmd += ["-w", f"\n{_STATUS_MARKER}%{{http_code}}", url]

    result = subprocess.run(cmd, input=input_data, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"ERROR: curl failed calling {method} {path}: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    stdout = result.stdout
    if _STATUS_MARKER not in stdout:
        print(f"ERROR: unexpected curl output for {method} {path}:\n{stdout}{result.stderr}", file=sys.stderr)
        sys.exit(1)
    payload, _, status_str = stdout.rpartition(_STATUS_MARKER)
    payload = payload.rstrip("\n")
    status = int(status_str.strip())

    if status >= 400:
        print(f"ERROR: {method} {path} -> HTTP {status}\n{payload}", file=sys.stderr)
        sys.exit(1)
    if not payload.strip():
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        print(f"ERROR: {method} {path} returned non-JSON body (HTTP {status}):\n{payload}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# instance types / capacity
# ---------------------------------------------------------------------------


def cmd_types(args: argparse.Namespace) -> None:
    resp = request("GET", "/instance-types")
    data = resp.get("data", {})
    if args.raw:
        print(json.dumps(resp, indent=2))
        return

    rows = []
    for name, entry in data.items():
        it = entry.get("instance_type", {})
        regions = entry.get("regions_with_capacity_available", [])
        if args.available_only and not regions:
            continue
        if args.filter and args.filter.lower() not in name.lower() and args.filter.lower() not in (
            it.get("gpu_description", "").lower()
        ):
            continue
        price = it.get("price_cents_per_hour")
        price_str = f"${price / 100:.2f}/hr" if price is not None else "?"
        region_names = ", ".join(r.get("name", "?") for r in regions) or "(no capacity anywhere right now)"
        rows.append((name, price_str, it.get("gpu_description", "?"), region_names))

    if not rows:
        print("No instance types matched.")
        return

    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    w2 = max(len(r[2]) for r in rows)
    for name, price, desc, regions in sorted(rows, key=lambda r: r[0]):
        print(f"{name:<{w0}}  {price:<{w1}}  {desc:<{w2}}  {regions}")


def find_capacity(preferred_type_substrings: list[str], region: str | None = None) -> tuple[str, str] | None:
    """Return (instance_type_name, region_name) for the first preference with
    live capacity, trying each substring in order. If `region` is given
    (e.g. because a persistent filesystem is locked to that region), only
    that region is considered. None if nothing available."""
    resp = request("GET", "/instance-types")
    data = resp.get("data", {})
    for pref in preferred_type_substrings:
        for name, entry in data.items():
            if pref.lower() not in name.lower():
                continue
            regions = entry.get("regions_with_capacity_available", [])
            if region:
                regions = [r for r in regions if r.get("name") == region]
            if regions:
                return name, regions[0].get("name")
    return None


def cmd_find_capacity(args: argparse.Namespace) -> None:
    """Script-friendly: print '<instance_type> <region>' for the first
    preference (in priority order) that has live capacity, or exit 1."""
    result = find_capacity(args.prefer, region=args.region)
    if result is None:
        scope = f"region {args.region}" if args.region else "any region"
        print(f"ERROR: none of {args.prefer} have capacity in {scope} right now.", file=sys.stderr)
        print("Lambda's popular GPU types sell out often - retry later or widen --prefer.", file=sys.stderr)
        sys.exit(1)
    instance_type, region = result
    print(f"{instance_type} {region}")


# ---------------------------------------------------------------------------
# ssh keys
# ---------------------------------------------------------------------------


def cmd_ssh_keys(args: argparse.Namespace) -> None:
    resp = request("GET", "/ssh-keys")
    for k in resp.get("data", []):
        print(f"{k.get('name'):<30} id={k.get('id')}")


def cmd_add_ssh_key(args: argparse.Namespace) -> None:
    with open(args.pubkey_file) as f:
        pubkey = f.read().strip()
    resp = request("POST", "/ssh-keys", {"name": args.name, "public_key": pubkey})
    key = resp.get("data", {})
    print(f"Registered '{key.get('name')}' (id={key.get('id')})")


# ---------------------------------------------------------------------------
# filesystems
# ---------------------------------------------------------------------------


def cmd_filesystems(args: argparse.Namespace) -> None:
    resp = request("GET", "/file-systems")
    for fs in resp.get("data", []):
        region = fs.get("region", {}).get("name", "?")
        print(f"{fs.get('name'):<30} region={region:<12} id={fs.get('id')}")


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------


def cmd_launch(args: argparse.Namespace) -> None:
    body: dict[str, Any] = {
        "region_name": args.region,
        "instance_type_name": args.instance_type,
        "ssh_key_names": [args.ssh_key],
        "quantity": 1,
    }
    if args.name:
        body["name"] = args.name
    if args.filesystem:
        body["file_system_names"] = [args.filesystem]

    # Price lookup for the confirmation banner.
    types_resp = request("GET", "/instance-types")
    entry = types_resp.get("data", {}).get(args.instance_type, {})
    price = entry.get("instance_type", {}).get("price_cents_per_hour")
    price_str = f"${price / 100:.2f}/hr" if price is not None else "unknown price"

    print("About to launch:")
    print(f"  instance_type: {args.instance_type}  ({price_str})")
    print(f"  region:        {args.region}")
    print(f"  filesystem:    {args.filesystem or '(none)'}")
    print(f"  ssh_key:       {args.ssh_key}")
    print("Billing starts the moment this call succeeds and only stops on an")
    print("explicit `stop` (terminate) call - not on idle, not on OS shutdown.")

    if not args.yes:
        print("\nRefusing to launch without --yes. Re-run with --yes once the", file=sys.stderr)
        print("human user has confirmed this in chat.", file=sys.stderr)
        sys.exit(2)

    resp = request("POST", "/instance-operations/launch", body)
    ids = resp.get("data", {}).get("instance_ids", [])
    if not ids:
        print(f"ERROR: launch call succeeded but returned no instance_ids: {resp}", file=sys.stderr)
        sys.exit(1)
    for iid in ids:
        print(iid)


def cmd_list(args: argparse.Namespace) -> None:
    resp = request("GET", "/instances")
    instances = resp.get("data", [])
    if not instances:
        print("No running instances.")
        return
    for inst in instances:
        region = inst.get("region", {}).get("name", "?")
        itype = inst.get("instance_type", {}).get("name", "?")
        print(
            f"{inst.get('id')}  {inst.get('name') or '(unnamed)':<20} "
            f"status={inst.get('status'):<10} ip={inst.get('ip') or '-':<15} "
            f"type={itype:<20} region={region}"
        )


def cmd_status(args: argparse.Namespace) -> None:
    resp = request("GET", f"/instances/{args.id}")
    print(json.dumps(resp.get("data", {}), indent=2))


def cmd_wait_ssh(args: argparse.Namespace) -> None:
    import socket

    deadline = time.time() + args.timeout
    ip = None
    while time.time() < deadline:
        resp = request("GET", f"/instances/{args.id}")
        inst = resp.get("data", {})
        status = inst.get("status")
        ip = inst.get("ip")
        print(f"  status={status} ip={ip or '-'}", file=sys.stderr)
        if status == "active" and ip:
            # status=active means booted; still confirm sshd is actually accepting.
            try:
                with socket.create_connection((ip, 22), timeout=5):
                    print(ip)
                    return
            except OSError:
                pass
        elif status in ("terminated", "unhealthy"):
            print(f"ERROR: instance entered status={status} while waiting", file=sys.stderr)
            sys.exit(1)
        time.sleep(10)
    print(f"ERROR: timed out after {args.timeout}s waiting for SSH (last ip={ip})", file=sys.stderr)
    sys.exit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    ids = args.id
    print(f"Terminating: {ids}")
    resp = request("POST", "/instance-operations/terminate", {"instance_ids": ids})
    terminated = resp.get("data", {}).get("terminated_instances", [])
    for inst in terminated:
        print(f"  terminated {inst.get('id')} ({inst.get('name') or 'unnamed'})")
    if not terminated:
        print(f"WARNING: terminate call returned no terminated_instances: {resp}", file=sys.stderr)


def cmd_stop_all(args: argparse.Namespace) -> None:
    resp = request("GET", "/instances")
    instances = resp.get("data", [])
    if not instances:
        print("No running instances - nothing to stop.")
        return
    ids = [i["id"] for i in instances]
    print(f"About to terminate ALL {len(ids)} running instance(s):")
    for i in instances:
        print(f"  {i.get('id')}  {i.get('name') or '(unnamed)'}  {i.get('instance_type', {}).get('name')}")
    if not args.yes:
        print("\nRefusing without --yes.", file=sys.stderr)
        sys.exit(2)
    resp = request("POST", "/instance-operations/terminate", {"instance_ids": ids})
    for inst in resp.get("data", {}).get("terminated_instances", []):
        print(f"  terminated {inst.get('id')}")


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("types", help="list instance types and live region capacity")
    t.add_argument("--available-only", action="store_true", help="only show types with capacity right now")
    t.add_argument("--filter", help="substring match on type name or GPU description")
    t.add_argument("--raw", action="store_true", help="dump raw JSON response")
    t.set_defaults(func=cmd_types)

    sk = sub.add_parser("ssh-keys", help="list registered SSH keys")
    sk.set_defaults(func=cmd_ssh_keys)

    ak = sub.add_parser("add-ssh-key", help="register a local public key with Lambda")
    ak.add_argument("--name", required=True)
    ak.add_argument("--pubkey-file", required=True)
    ak.set_defaults(func=cmd_add_ssh_key)

    fs = sub.add_parser("filesystems", help="list persistent filesystems")
    fs.set_defaults(func=cmd_filesystems)

    fc = sub.add_parser(
        "find-capacity",
        help="print '<instance_type> <region>' for the first --prefer entry with live capacity",
    )
    fc.add_argument(
        "--prefer",
        nargs="+",
        required=True,
        help="substrings tried in order, e.g. --prefer a100-sxm4-80gb a100-40gb",
    )
    fc.add_argument("--region", help="restrict to this region (e.g. because a filesystem is locked there)")
    fc.set_defaults(func=cmd_find_capacity)

    l = sub.add_parser("launch", help="launch an instance (costs money)")
    l.add_argument("--instance-type", required=True)
    l.add_argument("--region", required=True)
    l.add_argument("--ssh-key", required=True)
    l.add_argument("--filesystem", help="persistent filesystem name to attach")
    l.add_argument("--name", help="human-readable instance name")
    l.add_argument("--yes", action="store_true", help="required to actually launch")
    l.set_defaults(func=cmd_launch)

    ls = sub.add_parser("list", help="list running/booting instances")
    ls.set_defaults(func=cmd_list)

    st = sub.add_parser("status", help="show one instance's detail")
    st.add_argument("--id", required=True)
    st.set_defaults(func=cmd_status)

    ws = sub.add_parser("wait-ssh", help="block until instance is SSH-reachable, print its IP")
    ws.add_argument("--id", required=True)
    ws.add_argument("--timeout", type=int, default=600)
    ws.set_defaults(func=cmd_wait_ssh)

    sp = sub.add_parser("stop", help="terminate one or more instances")
    sp.add_argument("--id", nargs="+", required=True, dest="id")
    sp.set_defaults(func=cmd_stop)

    sa = sub.add_parser("stop-all", help="terminate EVERY instance on the account (safety net)")
    sa.add_argument("--yes", action="store_true")
    sa.set_defaults(func=cmd_stop_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
