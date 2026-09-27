"""Run a workload with optional Lambda credentials delivered through SSH stdin."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess

from lambda_cli import api_key
from project_env import load_project_env


def forwarding_enabled(override: bool | None) -> bool:
    if override is not None:
        return override
    value = os.environ.get("LAMBDA_FORWARD_API_KEY", "0").lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("LAMBDA_FORWARD_API_KEY must be 0, 1, false, or true")
    return value in {"1", "true"}


def run_remote(
    ssh_args: list[str], env_file: str, repo_dir: str, command: str, key: str | None
) -> int:
    script = "set +x\nset -euo pipefail\n"
    script += f"source {shlex.quote(env_file)}\nset +x\n"
    # Persistent activation files carry cache/venv settings, never credentials.
    script += "unset LAMBDA_API_KEY LAMBDA_API_KEY_FILE\n"
    if key is not None:
        script += f"export LAMBDA_API_KEY={shlex.quote(key)}\n"
    script += f"cd {shlex.quote(repo_dir)}\nexec bash -c {shlex.quote(command)}\n"
    child_env = {
        name: value for name, value in os.environ.items()
        if name not in {"LAMBDA_API_KEY", "LAMBDA_API_KEY_FILE"}
    }
    result = subprocess.run(
        ["ssh", *ssh_args, "bash", "-s"], input=script, text=True, env=child_env, check=False
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--forward-lambda-api-key", dest="forward", action="store_true")
    group.add_argument("--no-forward-lambda-api-key", dest="forward", action="store_false")
    parser.set_defaults(forward=None)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--env-file")
    parser.add_argument("--repo-dir")
    parser.add_argument("--cmd")
    parser.add_argument("ssh_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        load_project_env()
        key = api_key() if forwarding_enabled(args.forward) else None
    except ValueError as exc:
        parser.error(str(exc))
    if args.check_env:
        return 0
    if not all((args.env_file, args.repo_dir, args.cmd, args.ssh_args)):
        parser.error("--env-file, --repo-dir, --cmd and SSH arguments are required")
    ssh_args = args.ssh_args[1:] if args.ssh_args[0] == "--" else args.ssh_args
    return run_remote(ssh_args, args.env_file, args.repo_dir, args.cmd, key)


if __name__ == "__main__":
    raise SystemExit(main())
