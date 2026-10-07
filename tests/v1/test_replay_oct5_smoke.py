"""Replay the saved 5 October smoke records through the frozen v1 scorer.

The records live in the private docs checkout (`local/`), so the test skips elsewhere.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from rlvr_safety.io import read_jsonl
from rlvr_safety.permission.v1.benchmark import _score

RUN = (
    Path(__file__).resolve().parents[2]
    / "local/runs/permission-smoke-live-2026-10-05-retry4/retrieved/benchmark"
)


@unittest.skipUnless((RUN / "results.jsonl").is_file(), "private smoke records not available")
class ReplayOctoberFifthSmokeTests(unittest.TestCase):
    def test_v1_scorer_reproduces_every_saved_score(self):
        scenarios = {s["source_id"]: s for s in read_jsonl(RUN / "scenarios.jsonl")}
        requests = {r["episode_id"]: r for r in read_jsonl(RUN / "requests.jsonl")}
        rows = list(read_jsonl(RUN / "results.jsonl"))
        self.assertEqual(len(rows), 26)
        for row in rows:
            request = requests[row["episode_id"]]
            replayed = _score(scenarios[row["source_id"]], request, row["response"])
            self.assertEqual(replayed, row["score"], row["condition_id"])


if __name__ == "__main__":
    unittest.main()
