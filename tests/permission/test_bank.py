from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rlvr_safety.cli import permission as cli
from rlvr_safety.io import read_jsonl, write_jsonl
from rlvr_safety.permission.bank import merge_reviewed

from .helpers import scenario


def accepted(source_id, supersedes=None):
    row = scenario()
    row["source_id"] = source_id
    row["skeleton_id"] = f"skel_{source_id}"
    row["review"] = {"status": "accepted", "reviewers": ["ayesha", "aaliyan"]}
    if supersedes:
        row["supersedes"] = supersedes
    return row


class BankMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_jsonl(self.root / "a.jsonl", [accepted("s1"), accepted("s2")])
        write_jsonl(self.root / "b.jsonl", [accepted("s3", supersedes="s2")])

    def test_superseded_record_is_excluded_with_an_audit_trail(self):
        kept, audit = merge_reviewed([self.root / "a.jsonl", self.root / "b.jsonl"], ["s2"])
        self.assertEqual([r["source_id"] for r in kept], ["s1", "s3"])
        self.assertEqual(audit["excluded"], {"s2": {"superseded_by": "s3"}})
        self.assertEqual(audit["statuses"], {"accepted": 2})
        self.assertEqual(set(audit["inputs"]), {"a.jsonl", "b.jsonl"})
        self.assertEqual(len(audit["inputs"]["a.jsonl"]["sha256"]), 64)

    def test_exclusion_without_a_superseding_record_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no included record supersedes"):
            merge_reviewed([self.root / "a.jsonl"], ["s2"])

    def test_unknown_exclusions_and_duplicates_are_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown records"):
            merge_reviewed([self.root / "a.jsonl"], ["ghost"])
        write_jsonl(self.root / "dup.jsonl", [accepted("s1")])
        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            merge_reviewed([self.root / "a.jsonl", self.root / "dup.jsonl"], [])

    def test_unreviewed_records_block_the_merge_unless_explicitly_allowed(self):
        pending = deepcopy(accepted("s4"))
        pending["review"] = {"status": "pending", "reviewers": []}
        write_jsonl(self.root / "p.jsonl", [pending])
        with self.assertRaisesRegex(ValueError, "without accepted review"):
            merge_reviewed([self.root / "a.jsonl", self.root / "p.jsonl"], [])
        kept, _ = merge_reviewed(
            [self.root / "a.jsonl", self.root / "p.jsonl"], [], require_accepted=False
        )
        self.assertEqual(len(kept), 3)

    def test_cli_writes_new_files_only(self):
        out = self.root / "merged.jsonl"
        args = [
            "bank",
            "merge",
            "--inputs",
            str(self.root / "a.jsonl"),
            str(self.root / "b.jsonl"),
            "--exclude",
            "s2",
            "--out",
            str(out),
        ]
        self.assertEqual(cli.main(args), 0)
        self.assertEqual(len(list(read_jsonl(out))), 2)
        audit = json.loads(out.with_suffix(".merge-audit.json").read_text())
        self.assertEqual(audit["included"], 2)
        self.assertEqual(cli.main(args), 2)


if __name__ == "__main__":
    unittest.main()
