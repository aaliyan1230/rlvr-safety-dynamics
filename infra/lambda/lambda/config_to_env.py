#!/usr/bin/env python3
"""
Minimal parser for lambda/config.yaml's specific (flat key: value, plus one
YAML list) shape - deliberately NOT a general YAML parser, so this has zero
dependencies (no pyyaml needed on your laptop or the pod). If config.yaml
grows real YAML nesting, switch to pyyaml instead of extending this.

Prints shell-sourceable `KEY=value` lines (bash arrays for lists), e.g.:
    eval "$(python3 lambda/config_to_env.py)"
"""

import shlex
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def parse(path: Path) -> dict:
    result: dict = {}
    current_list_key = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split(" #", 1)[0].rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith("  - "):
            if current_list_key is None:
                print(f"ERROR parsing config.yaml: list item with no preceding key: {raw_line}", file=sys.stderr)
                sys.exit(1)
            item = line.strip()[2:].strip().strip('"').strip("'")
            result.setdefault(current_list_key, []).append(item)
            continue
        if ":" not in line:
            print(f"ERROR parsing config.yaml: unrecognized line: {raw_line}", file=sys.stderr)
            sys.exit(1)
        key, _, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value == "":
            # Bare "key:" with nothing after it - a YAML list follows.
            current_list_key = key
            result[key] = []
        else:
            current_list_key = None
            result[key] = raw_value.strip('"').strip("'")
    return result


def main() -> None:
    cfg = parse(CONFIG_PATH)
    for key, value in cfg.items():
        env_key = key.upper()
        if isinstance(value, list):
            quoted = " ".join(shlex.quote(v) for v in value)
            print(f"{env_key}=({quoted})")
        else:
            print(f"{env_key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
