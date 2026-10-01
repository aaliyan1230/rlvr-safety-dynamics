#!/usr/bin/env python3
"""Read-only RunPod REST v2 inventory, billing, and connection inspection."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlencode

API_BASE = "https://api.runpod.io/v2"
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"
MARKER = "<<<RUNPOD_HTTP_STATUS>>>"


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def api_key(path: Path = PROJECT_ENV) -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key and path.exists():
        if path.stat().st_mode & 0o077:
            raise ValueError("Restrict the project .env to owner access with chmod 600")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().removeprefix("export ")
            name, separator, value = line.partition("=")
            if separator and name.strip() == "RUNPOD_API_KEY":
                try:
                    parts = shlex.split(value, comments=True)
                except ValueError:
                    raise ValueError("Invalid quoting in RUNPOD_API_KEY") from None
                if len(parts) != 1:
                    raise ValueError("RUNPOD_API_KEY must be one quoted or unquoted value")
                key = parts[0]
    if not key:
        raise ValueError("Set RUNPOD_API_KEY in the environment or owner-only project .env")
    if any(character in key for character in "\r\n\x00"):
        raise ValueError("Invalid control character in RUNPOD_API_KEY")
    return key


def config_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def request(path: str, params: dict | None = None) -> dict:
    """Never put credentials in command arguments or echo provider bodies on error."""
    if not path.startswith("/") or "?" in path or "#" in path:
        raise ValueError("Expected a relative API path without query or fragment")
    key = api_key()
    url = API_BASE + path
    if params:
        url += "?" + urlencode(params)
    command = [
        "curl", "-4", "--silent", "--show-error", "--connect-timeout", "10",
        "--max-time", "45", "--request", "GET", "--config", "-",
        "--header", f"User-Agent: {USER_AGENT}",
        "--write-out", f"\n{MARKER}%{{http_code}}", url,
    ]
    credentials = "header = " + config_quote("Authorization: Bearer " + key) + "\n"
    result = subprocess.run(
        command, input=credentials, capture_output=True, text=True, check=False,
        env={name: value for name, value in os.environ.items() if name != "RUNPOD_API_KEY"},
    )
    if result.returncode:
        raise ApiError(f"RunPod transport failed (curl exit {result.returncode}); retry reads only")
    payload, separator, status_text = result.stdout.rpartition(MARKER)
    if not separator or not status_text.strip().isdigit():
        raise ApiError("RunPod returned an invalid HTTP response")
    status = int(status_text.strip())
    if status >= 400:
        raise ApiError(f"RunPod GET {path} returned HTTP {status}", status)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise ApiError("RunPod returned an invalid JSON response") from None
    if not isinstance(data, dict):
        raise ApiError("RunPod returned an unexpected response shape")
    return data


def list_pods() -> list[dict]:
    pods = []
    cursor = None
    seen = set()
    while True:
        response = request("/pods", {"cursor": cursor} if cursor else None)
        batch = response.get("pods")
        if not isinstance(batch, list):
            raise ApiError("RunPod inventory missing pods; cannot assume no resources")
        pods.extend(batch)
        pagination = response.get("pagination", {})
        if not pagination.get("hasNextPage"):
            return pods
        cursor = pagination.get("nextCursor")
        if not cursor or cursor in seen:
            raise ApiError("RunPod inventory pagination did not advance")
        seen.add(cursor)


def public_pod(pod: dict) -> dict:
    # Full pod objects include environment variables, which can contain secrets.
    fields = ("id", "name", "status", "cloud", "gpu", "cpu", "cost", "dataCenterId",
              "cudaVersion", "runtime", "ssh", "mounts", "createdAt", "startedAt")
    return {name: pod[name] for name in fields if name in pod}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pods", help="List all pods, excluding secret-bearing configuration")
    status = commands.add_parser("status")
    status.add_argument("id")
    commands.add_parser("volumes")
    commands.add_parser("ssh-keys")
    gpus = commands.add_parser("gpus")
    gpus.add_argument("--filter", default="")
    gpus.add_argument("--min-cuda-version")
    commands.add_parser("datacenters")
    billing = commands.add_parser("billing")
    billing.add_argument("--start-time", required=True)
    billing.add_argument("--end-time", required=True)
    billing.add_argument("--bucket-size", choices=["hour", "day", "week", "month"], default="day")
    args = parser.parse_args(argv)
    try:
        if args.command == "pods":
            data = {"pods": [public_pod(pod) for pod in list_pods()]}
        elif args.command == "status":
            data = public_pod(request("/pods/" + quote(args.id, safe="")))
        elif args.command == "gpus":
            params = {"include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "count": 1}
            if args.min_cuda_version:
                params["minCudaVersion"] = args.min_cuda_version
            data = request("/catalog/gpus", params)
            rows = data.get("gpus")
            if not isinstance(rows, list):
                raise ApiError("RunPod catalog missing gpus")
            data = {"gpus": [row for row in rows if args.filter.lower() in (
                row.get("id", "") + " " + row.get("name", "")
            ).lower()]}
        elif args.command == "billing":
            data = request("/billing", {
                "startTime": args.start_time, "endTime": args.end_time,
                "bucketSize": args.bucket_size,
            })
        else:
            path = {"volumes": "/network-volumes", "ssh-keys": "/account/ssh-keys",
                    "datacenters": "/catalog/datacenters"}[args.command]
            data = request(path)
        print(json.dumps(data, indent=2))
        return 0
    except (ApiError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
