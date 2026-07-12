from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/build_ai_semantic_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_ai_semantic_audit", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load AI semantic-audit builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AISemanticAuditTests(unittest.TestCase):
    def test_expands_complete_disclosed_review(self) -> None:
        module = load_module()
        config = json.loads(
            (REPO / "configs/audits/ai_semantic_audit_v1.json").read_text(encoding="utf-8")
        )
        rows = module.build_rows(REPO, config)
        counts = Counter(row["decision"] for row in rows)
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({(row["source_id"], row["wording_id"]) for row in rows}), 72)
        self.assertEqual(counts, {"pass": 63, "pass_with_caveat": 9})
        self.assertTrue(all(row["accepted"] for row in rows))
        self.assertTrue(all(not row["independent_human_review"] for row in rows))


if __name__ == "__main__":
    unittest.main()
