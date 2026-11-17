"""Print role shares from a training run's rollouts.jsonl in windows of optimizer steps."""

from __future__ import annotations

import json
import sys
from collections import Counter


def shares(rows: list[dict], permission: str) -> dict[str, float]:
    counts: Counter = Counter()
    for r in rows:
        counts.update(r["roles"].get(permission, {}))
    n = sum(counts.values()) or 1
    return {k: round(v / n, 3) for k, v in sorted(counts.items())}


def main() -> None:
    path, window = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rows = [json.loads(line) for line in open(path)]
    for i in range(0, len(rows), window):
        w = rows[i : i + window]
        reward = sum(r["mean_reward"] for r in w) / len(w)
        f, a = shares(w, "forbidden"), shares(w, "allowed")
        print(
            f"steps {w[0]['step']:>3}-{w[-1]['step']:<3} reward {reward:.3f} "
            f"forbidden boundary {f.get('boundary', 0):.3f} clarify {f.get('clarify', 0):.3f} "
            f"| allowed boundary {a.get('boundary', 0):.3f}"
        )


if __name__ == "__main__":
    main()
