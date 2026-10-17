from __future__ import annotations

import json
import unittest

from rlvr_safety.agent_tasks.jobs import critique, ideation
from tests.permission.helpers import scenario


def mechanic(i):
    return {"mechanic_id": f"m_{i}", "setting": "s", "boundary": "b", "alternative": "a",
            "operator_authority": "o"}


class IdeationTests(unittest.TestCase):
    def test_round_trip_and_defects(self):
        brief = {"brief_id": "conf_resource", "family": "resource_allocation", "count": 2}
        tasks, refs = ideation.build_tasks(brief, [{"mechanic_id": "old"}])
        task = tasks[0]
        self.assertEqual(task["input"]["existing"], [{"mechanic_id": "old"}])
        ok = ideation.validate(task, {"payload": {"mechanics": [mechanic(1), mechanic(2)]}})
        self.assertEqual(len(ok["mechanics"]), 2)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            ideation.validate(task, {"payload": {"mechanics": [mechanic(1)]}})
        with self.assertRaisesRegex(ValueError, "unique"):
            ideation.validate(task, {"payload": {"mechanics": [mechanic(1), mechanic(1)]}})
        with self.assertRaisesRegex(ValueError, "unknown family"):
            ideation.build_tasks(dict(brief, family="x"), [])


class CritiqueTests(unittest.TestCase):
    def payload(self, status="pass"):
        findings = [{"check": c, "status": "pass", "detail": ""} for c in critique.CHECKS]
        findings[1]["status"] = status
        verdict = {"pass": "ok", "concern": "minor", "fail": "revise"}[status]
        return {"findings": findings, "verdict": verdict}

    def test_tasks_hide_provenance_and_flags_are_collected(self):
        record = scenario()
        record["authoring"] = {"agent": "secret"}
        tasks, refs = critique.build_tasks([record])
        self.assertNotIn("authoring", json.dumps(tasks[0]["input"]))
        self.assertNotIn("source_id", tasks[0]["input"]["record"])
        result = critique.validate(tasks[0], {"payload": self.payload("fail")})
        rows = critique.flags([{"task_id": tasks[0]["task_id"], "record": result,
                                "agent": "a", "model_reported": "m"}], refs)
        self.assertEqual(rows[0]["verdict"], "revise")
        self.assertEqual(list(rows[0]["flags"]), ["alternative_fully_completes_task"])

    def test_inconsistent_or_incomplete_findings_are_rejected(self):
        task = critique.build_tasks([scenario()])[0][0]
        bad = self.payload("fail")
        bad["verdict"] = "ok"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            critique.validate(task, {"payload": bad})
        short = self.payload()
        short["findings"] = short["findings"][:-1]
        with self.assertRaisesRegex(ValueError, "missing"):
            critique.validate(task, {"payload": short})


if __name__ == "__main__":
    unittest.main()
