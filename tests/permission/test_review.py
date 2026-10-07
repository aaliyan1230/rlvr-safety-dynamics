from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.cli import permission as cli
from rlvr_safety.io import read_jsonl, write_jsonl
from rlvr_safety.permission import review

from .helpers import control, scenario

SOURCES = ("fixture_transfer", "fixture_control")


def bank():
    return [scenario(), control()]


def fill(path: Path, decisions: dict[str, str], default: str = "pending") -> Path:
    """Fill one reviewer's sheet in place: ``decisions`` maps source_id to a decision."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["decision"] = decisions.get(row["source_id"], default)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.counts = review.export_queue(bank(), ["ayesha", "aaliyan"], self.root / "queues")
        self.ayesha = self.root / "queues/queue-ayesha.csv"
        self.aaliyan = self.root / "queues/queue-aaliyan.csv"

    def test_each_reviewer_gets_their_own_sheet_with_only_their_rows(self):
        self.assertEqual(
            self.counts, {"ayesha": 3, "aaliyan": 3}
        )  # scenario: 2 wordings + control: 1
        for path, name in ((self.ayesha, "ayesha"), (self.aaliyan, "aaliyan")):
            with path.open(encoding="utf-8", newline="") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual({r["reviewer"] for r in table}, {name})
            self.assertEqual({r["decision"] for r in table}, {"pending"})
            self.assertEqual({r["record_type"] for r in table}, {"scenario", "control"})
        with self.assertRaises(FileExistsError):
            review.export_queue(bank(), ["ayesha", "aaliyan"], self.root / "queues")

    def test_two_distinct_reviewers_are_required(self):
        for reviewers in (["ayesha"], ["ayesha", "ayesha"], ["a b", "c"]):
            with self.subTest(reviewers), self.assertRaisesRegex(ValueError, "two distinct"):
                review.export_queue(bank(), reviewers, self.root / "other")

    def test_acceptance_needs_two_named_reviewers_to_accept_every_wording(self):
        fill(self.ayesha, {s: "accept" for s in SOURCES})
        fill(self.aaliyan, {s: "accept" for s in SOURCES})
        updated, audit = review.apply_reviews(bank(), [self.ayesha, self.aaliyan])
        for record in updated:
            self.assertEqual(record["review"]["status"], "accepted")
            self.assertEqual(record["review"]["reviewers"], ["aaliyan", "ayesha"])
        self.assertEqual(set(audit["queues"]), {"queue-ayesha.csv", "queue-aaliyan.csv"})

    def test_a_single_reviewer_or_pending_decisions_keep_the_record_pending(self):
        fill(self.ayesha, {s: "accept" for s in SOURCES})
        updated, _ = review.apply_reviews(bank(), [self.ayesha])
        self.assertEqual({r["review"]["status"] for r in updated}, {"pending"})
        fill(self.aaliyan, {"fixture_control": "accept"})  # the scenario stays pending
        updated, _ = review.apply_reviews(bank(), [self.ayesha, self.aaliyan])
        status = {r["source_id"]: r["review"]["status"] for r in updated}
        self.assertEqual(status, {"fixture_transfer": "pending", "fixture_control": "accepted"})
        self.assertEqual(
            {
                tuple(r["review"]["reviewers"])
                for r in updated
                if r["review"]["status"] == "pending"
            },
            {()},
        )

    def test_any_rejection_rejects_and_revise_keeps_pending(self):
        fill(self.ayesha, {"fixture_transfer": "accept", "fixture_control": "accept"})
        fill(self.aaliyan, {"fixture_transfer": "reject", "fixture_control": "revise"})
        updated, _ = review.apply_reviews(bank(), [self.ayesha, self.aaliyan])
        status = {r["source_id"]: r["review"]["status"] for r in updated}
        self.assertEqual(status, {"fixture_transfer": "rejected", "fixture_control": "pending"})

    def test_invalid_sheets_are_rejected(self):
        text = self.ayesha.read_text().replace("pending", "maybe", 1)
        (self.root / "bad.csv").write_text(text)
        with self.assertRaisesRegex(ValueError, "unknown decision"):
            review.apply_reviews(bank(), [self.root / "bad.csv"])
        (self.root / "cols.csv").write_text("a,b\n1,2\n")
        with self.assertRaisesRegex(ValueError, "expected queue columns"):
            review.apply_reviews(bank(), [self.root / "cols.csv"])
        with self.assertRaisesRegex(ValueError, "unknown record/wording"):
            review.apply_reviews(bank()[:1], [self.ayesha])
        with self.assertRaisesRegex(ValueError, "duplicate decision"):
            review.apply_reviews(bank(), [self.ayesha, self.ayesha])

    def test_packet_shows_prompts_state_and_provenance(self):
        text = review.render_packet(bank())
        for fragment in (
            "## fixture_transfer (scenario, resource_allocation)",
            "## fixture_control (control, neutral_control)",
            "allowed · mcq",
            "forbidden · executable",
            "allowed · option_free",
            "neutral IDs",
            "Answer with one letter only, for example: B",
            "Execute only archive_b once",
        ):
            self.assertIn(fragment, text)
        self.assertRegex(text, r"Execute only op_[a-z] once")
        prompts = "".join(chunk.split("```")[0] for chunk in text.split("```text")[1:])
        self.assertNotIn("{requested_id}", prompts)

    def test_cli_round_trip_writes_new_files_only(self):
        write_jsonl(self.root / "bank.jsonl", bank())
        out_dir = self.root / "cli-queues"
        code = cli.main(
            [
                "review",
                "export",
                "--bank",
                str(self.root / "bank.jsonl"),
                "--reviewers",
                "ayesha,aaliyan",
                "--out-dir",
                str(out_dir),
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            cli.main(
                [
                    "review",
                    "packet",
                    "--bank",
                    str(self.root / "bank.jsonl"),
                    "--out",
                    str(self.root / "packet.md"),
                ]
            ),
            0,
        )
        for name in ("ayesha", "aaliyan"):
            fill(out_dir / f"queue-{name}.csv", {s: "accept" for s in SOURCES})
        out = self.root / "bank.reviewed.jsonl"
        args = [
            "review",
            "apply",
            "--bank",
            str(self.root / "bank.jsonl"),
            "--queues",
            str(out_dir / "queue-ayesha.csv"),
            str(out_dir / "queue-aaliyan.csv"),
            "--out",
            str(out),
        ]
        self.assertEqual(cli.main(args), 0)
        self.assertEqual({r["review"]["status"] for r in read_jsonl(out)}, {"accepted"})
        audit = json.loads(out.with_suffix(".review-audit.json").read_text())
        self.assertEqual(audit["records"]["fixture_transfer"]["reviewers"], ["aaliyan", "ayesha"])
        self.assertEqual(cli.main(args), 2)  # refuses to overwrite


if __name__ == "__main__":
    unittest.main()
