from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.agent_tasks.core import (
    CallableBackend,
    CommandBackend,
    audit,
    ingest,
    load_manifest,
    make_task,
    pending_tasks,
    run_backend,
    verify_task,
    write_plan,
)

SCHEMA = {"answer": "int"}


def tasks(n=3):
    return [make_task("demo", "demo-rubric-1", "Add one.", {"x": i}, SCHEMA) for i in range(n)]


def validate(task, envelope):
    payload = envelope["payload"]
    if payload.get("answer") != task["input"]["x"] + 1:
        raise ValueError("wrong answer")
    return payload


def solver(task):
    return json.dumps({"answer": task["input"]["x"] + 1})


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "job"

    def tearDown(self):
        self.tmp.cleanup()

    def test_task_hash_detects_tampering(self):
        task = tasks(1)[0]
        verify_task(task)
        task["input"]["x"] = 99
        with self.assertRaisesRegex(ValueError, "does not match its hash"):
            verify_task(task)

    def test_plan_is_idempotent_and_refuses_conflicts(self):
        planned = tasks()
        write_plan(self.dir, "demo", "demo-rubric-1", planned, refs={planned[0]["task_id"]: ["r"]})
        first = (self.dir / "manifest.json").read_text()
        write_plan(self.dir, "demo", "demo-rubric-1", planned)
        self.assertEqual(json.loads(first)["tasks"], load_manifest(self.dir)["tasks"])
        self.assertEqual(len(pending_tasks(self.dir)), 3)
        with self.assertRaisesRegex(ValueError, "different job or rubric"):
            write_plan(self.dir, "demo", "other-rubric", planned)
        forged = dict(planned[0], input={"x": 5})
        with self.assertRaisesRegex(ValueError, "does not match its hash"):
            write_plan(self.dir, "demo", "demo-rubric-1", [forged])
        with self.assertRaisesRegex(ValueError, "duplicate task IDs"):
            write_plan(self.dir, "demo", "demo-rubric-1", [planned[0], planned[0]])

    def test_end_to_end_with_a_callable_backend(self):
        write_plan(self.dir, "demo", "demo-rubric-1", tasks())
        backend = CallableBackend(solver, agent="unit-test", model_reported="none")
        done = run_backend(self.dir, backend, limit=2)
        self.assertEqual(len(done), 2)
        counts = ingest(self.dir, validate, operator="tester", now="2026-10-07T00:00:00+00:00")
        self.assertEqual(counts, {"accepted": 2, "rejected": 0, "unchanged": 0, "missing": 1})
        run_backend(self.dir, backend)
        counts = ingest(self.dir, validate, operator="tester")
        self.assertEqual(counts["accepted"], 1)
        self.assertEqual(counts["unchanged"], 2)
        report = audit(self.dir)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual((report["tasks"], report["accepted"], report["unanswered"]), (3, 3, 0))
        rows = [json.loads(x) for x in (self.dir / "ingested.jsonl").read_text().splitlines()]
        self.assertEqual({r["operator"] for r in rows}, {"tester"})
        self.assertTrue(all(r["agent"] == "unit-test" and r["response_sha256"] for r in rows))

    def test_rejections_are_recorded_with_reasons_and_never_deleted(self):
        planned = tasks(1)
        write_plan(self.dir, "demo", "demo-rubric-1", planned)
        envelope = {
            "task_id": planned[0]["task_id"],
            "task_sha256": planned[0]["task_sha256"],
            "agent": "a",
            "model_reported": "m",
            "payload": {"answer": 41},
        }
        path = self.dir / "responses" / f"{planned[0]['task_id']}.json"
        path.write_text(json.dumps(envelope))
        counts = ingest(self.dir, validate, operator="t")
        self.assertEqual((counts["accepted"], counts["rejected"]), (0, 1))
        counts = ingest(self.dir, validate, operator="t")
        self.assertEqual((counts["rejected"], counts["unchanged"]), (0, 1))
        rejections = [
            json.loads(x) for x in (self.dir / "rejections.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rejections), 1)
        self.assertIn("wrong answer", rejections[0]["reason"])
        report = audit(self.dir)
        self.assertEqual((report["accepted"], report["rejected_only"]), (0, 1))

    def test_envelope_must_name_its_task_and_identify_its_agent(self):
        planned = tasks(2)
        write_plan(self.dir, "demo", "demo-rubric-1", planned)
        good = {"task_sha256": planned[0]["task_sha256"], "payload": {"answer": 1}}
        cases = [
            {**good, "task_id": planned[0]["task_id"], "agent": "", "model_reported": "m"},
            {**good, "task_id": planned[0]["task_id"], "model_reported": "m"},
            {**good, "task_id": planned[1]["task_id"], "agent": "a", "model_reported": "m"},
            {
                **good,
                "task_sha256": "0" * 64,
                "task_id": planned[0]["task_id"],
                "agent": "a",
                "model_reported": "m",
            },
        ]
        for case, task in zip(cases, [planned[0], planned[0], planned[0], planned[0]], strict=True):
            (self.dir / "responses" / f"{task['task_id']}.json").write_text(json.dumps(case))
            ingest(self.dir, validate, operator="t")
            (self.dir / "responses" / f"{task['task_id']}.json").unlink()
        rejected = (self.dir / "rejections.jsonl").read_text().splitlines()
        self.assertEqual(len(rejected), 4)

    def test_response_changed_after_ingestion_is_detected(self):
        write_plan(self.dir, "demo", "demo-rubric-1", tasks(1))
        run_backend(self.dir, CallableBackend(solver, "a", "m"))
        ingest(self.dir, validate, operator="t")
        path = next((self.dir / "responses").glob("*.json"))
        envelope = json.loads(path.read_text())
        envelope["agent"] = "someone else"
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "changed after ingestion"):
            ingest(self.dir, validate, operator="t")
        report = audit(self.dir)
        self.assertFalse(report["ok"])

    def test_audit_reports_orphan_responses_and_edited_tasks(self):
        planned = tasks(1)
        write_plan(self.dir, "demo", "demo-rubric-1", planned)
        (self.dir / "responses" / "stray-123.json").write_text("{}")
        report = audit(self.dir)
        self.assertFalse(report["ok"])
        self.assertTrue(any("without tasks" in p for p in report["problems"]))
        (self.dir / "responses" / "stray-123.json").unlink()
        task_path = self.dir / "tasks" / f"{planned[0]['task_id']}.json"
        edited = json.loads(task_path.read_text())
        edited["input"]["x"] = 7
        task_path.write_text(json.dumps(edited))
        self.assertFalse(audit(self.dir)["ok"])

    def test_command_backend_runs_any_cli_per_task(self):
        write_plan(self.dir, "demo", "demo-rubric-1", tasks(2))
        script = (
            "import json,sys; t=json.load(sys.stdin); "
            "print(json.dumps({'answer': t['input']['x']+1}))"
        )
        backend = CommandBackend([sys.executable, "-c", script], "echo-cli", "none")
        run_backend(self.dir, backend)
        counts = ingest(self.dir, validate, operator="t")
        self.assertEqual(counts["accepted"], 2)
        row = json.loads((self.dir / "ingested.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["backend"]["type"], "command")

    def test_failing_command_does_not_write_a_response(self):
        write_plan(self.dir, "demo", "demo-rubric-1", tasks(1))
        backend = CommandBackend([sys.executable, "-c", "raise SystemExit(3)"], "bad", "none")
        with self.assertRaises(subprocess.CalledProcessError):
            run_backend(self.dir, backend)
        self.assertEqual(len(pending_tasks(self.dir)), 1)


if __name__ == "__main__":
    unittest.main()
