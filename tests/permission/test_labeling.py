from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.agent_tasks.core import CallableBackend, ingest, run_backend, write_plan
from rlvr_safety.agent_tasks.jobs import freeform_prelabel, mcq_judge
from rlvr_safety.io import read_jsonl
from rlvr_safety.permission import labeling
from rlvr_safety.permission.benchmark import prepare_requests, run_benchmark
from rlvr_safety.permission.prompts import build_conditions
from rlvr_safety.permission.scripted import ScriptedProvider

from .helpers import scenario

PLAN = {"mcq": {"orders": "all"}, "executable": {"orders": "one"}, "option_free": {"count": 1}}


def prepare(tmp: Path):
    scenarios = [scenario()]
    requests = prepare_requests(build_conditions(scenarios, PLAN, min_wordings=2))
    run_benchmark(scenarios, requests, ScriptedProvider(scenarios, "noisy"), tmp / "run")
    rows = list(read_jsonl(tmp / "run/results.jsonl"))
    by_request = {r["episode_id"]: r for r in requests}
    by_source = {s["source_id"]: s for s in scenarios}
    return rows, by_request, by_source


def fill(csv_in: Path, csv_out: Path, chooser):
    with csv_in.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = chooser(row)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class KappaTests(unittest.TestCase):
    def test_kappa_and_agreement(self):
        self.assertAlmostEqual(
            labeling.cohens_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]), 1.0
        )
        self.assertAlmostEqual(
            labeling.cohens_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]), 0.0
        )
        self.assertIsNone(labeling.cohens_kappa(["A", "A"], ["A", "A"]))
        with self.assertRaises(ValueError):
            labeling.cohens_kappa([], [])
        result = labeling.agreement(
            {"t1": "A", "t2": "B", "t3": "C"}, {"t1": "A", "t2": "A", "t4": "B"}
        )
        self.assertEqual(
            (result["n"], result["only_in_first"], result["only_in_second"]), (2, 1, 1)
        )
        self.assertAlmostEqual(result["exact_agreement"], 0.5)
        self.assertEqual(result["disagreements"], ["t2"])
        self.assertEqual(labeling.agreement({}, {"x": "A"})["n"], 0)


class FreeformLabellingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.rows, self.requests, self.scenarios = prepare(self.dir)
        tasks, self.refs = freeform_prelabel.build_tasks(self.rows, self.requests, self.scenarios)
        self.job = self.dir / "pre"
        write_plan(self.job, freeform_prelabel.JOB, freeform_prelabel.RUBRIC_ID, tasks, self.refs)

    def test_export_is_blinded_and_refuses_to_overwrite(self):
        info = labeling.export_csv(self.job, self.dir / "sheet.csv")
        self.assertEqual(info["rows"], len(self.refs))
        text = (self.dir / "sheet.csv").read_text().lower()
        for word in ("forbids", "permits", "boundary", "alternative", "use_other", "ai_label"):
            self.assertNotIn(word, text)
        header = (self.dir / "sheet.csv").read_text().splitlines()[0]
        self.assertEqual(header, "task_id,situation,option_A,option_B,option_C,answer,label,notes")
        with self.assertRaises(FileExistsError):
            labeling.export_csv(self.job, self.dir / "sheet.csv")

    def test_import_validates_labels_and_coverage(self):
        labeling.export_csv(self.job, self.dir / "sheet.csv")
        fill(self.dir / "sheet.csv", self.dir / "bad.csv", lambda r: "Z")
        with self.assertRaisesRegex(ValueError, "not in"):
            labeling.import_csv(self.job, self.dir / "bad.csv", "ayesha")
        fill(self.dir / "sheet.csv", self.dir / "empty.csv", lambda r: "")
        with self.assertRaisesRegex(ValueError, "no label"):
            labeling.import_csv(self.job, self.dir / "empty.csv", "ayesha")
        fill(self.dir / "sheet.csv", self.dir / "ok.csv", lambda r: "none")
        info = labeling.import_csv(self.job, self.dir / "ok.csv", "ayesha")
        self.assertEqual(info["labelled"], len(self.refs))
        with self.assertRaises(FileExistsError):
            labeling.import_csv(self.job, self.dir / "ok.csv", "ayesha")
        with self.assertRaisesRegex(ValueError, "identifier"):
            labeling.import_csv(self.job, self.dir / "ok.csv", "bad name")

    def test_partial_sheet_and_unknown_tasks(self):
        labeling.export_csv(self.job, self.dir / "sheet.csv")
        counter = {"n": 0}

        def half(row):
            counter["n"] += 1
            return "none" if counter["n"] % 2 else ""

        fill(self.dir / "sheet.csv", self.dir / "half.csv", half)
        with self.assertRaisesRegex(ValueError, "no label"):
            labeling.import_csv(self.job, self.dir / "half.csv", "p")
        info = labeling.import_csv(self.job, self.dir / "half.csv", "p", allow_partial=True)
        self.assertLess(info["labelled"], len(self.refs))
        lines = (self.dir / "sheet.csv").read_text().splitlines()
        (self.dir / "unknown.csv").write_text(
            "\n".join(lines[:1] + ["task-x,s,a,b,c,ans,none,"]) + "\n"
        )
        with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
            labeling.import_csv(self.job, self.dir / "unknown.csv", "q", allow_partial=True)

    def test_two_humans_ai_and_consensus(self):
        labeling.export_csv(self.job, self.dir / "sheet.csv")
        fill(self.dir / "sheet.csv", self.dir / "a.csv", lambda r: "none")
        toggle = {"n": 0}

        def second(row):
            toggle["n"] += 1
            return "none" if toggle["n"] != 1 else "unjudgeable"

        fill(self.dir / "sheet.csv", self.dir / "b.csv", second)
        labeling.import_csv(self.job, self.dir / "a.csv", "ayesha")
        labeling.import_csv(self.job, self.dir / "b.csv", "aaliyan")
        run_backend(
            self.job,
            CallableBackend(
                lambda t: json.dumps({"matches": "none", "evidence": ""}), "unit", "none"
            ),
        )
        ingest(self.job, freeform_prelabel.validate, operator="t")
        a, b = (
            labeling.load_human_labels(self.job, "ayesha"),
            labeling.load_human_labels(self.job, "aaliyan"),
        )
        ai = labeling.load_ai_labels(self.job)
        human_vs_human = labeling.agreement(a, b)
        self.assertEqual(len(human_vs_human["disagreements"]), 1)
        self.assertEqual(labeling.agreement(a, ai)["exact_agreement"], 1.0)
        agreed = labeling.consensus(self.job, ["ayesha", "aaliyan"])
        self.assertEqual(len(agreed), len(a) - 1)
        disputed = human_vs_human["disagreements"][0]
        self.assertNotIn(disputed, agreed)
        (self.job / "human_labels" / "adjudicated.jsonl").write_text(
            json.dumps({"task_id": disputed, "labeller": "adjudicated", "label": "none"}) + "\n"
        )
        self.assertEqual(labeling.consensus(self.job, ["ayesha", "aaliyan"])[disputed], "none")

    def test_labels_translate_to_episode_roles(self):
        key = {"A": "use_own", "B": "ask", "C": "use_other"}
        ref_episode = next(iter(self.refs.values()))[0]["episode_id"]
        task_id = next(iter(self.refs))
        source_of = {r["episode_id"]: r["source_id"] for r in self.rows}
        for ref in self.refs[task_id]:
            ref["letter_key"] = key
        (self.job / "refs.json").write_text(json.dumps(self.refs))
        letter = next(k for k, v in key.items() if v == "use_other")
        roles = labeling.episode_role_labels(self.job, {task_id: letter}, self.scenarios, source_of)
        self.assertEqual(roles[ref_episode], "boundary")


class McqLabellingTests(unittest.TestCase):
    def test_mcq_sheet_labels_map_to_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows, requests, scenarios = prepare(tmp)
            tasks, refs = mcq_judge.build_tasks(rows, requests, scenarios)
            job = tmp / "judge"
            write_plan(job, mcq_judge.JOB, mcq_judge.RUBRIC_ID, tasks, refs)
            info = labeling.export_csv(job, tmp / "sheet.csv")
            self.assertEqual(info["allowed_labels"], ["A", "B", "C", "none"])
            self.assertEqual(
                (tmp / "sheet.csv").read_text().splitlines()[0],
                "task_id,option_A,option_B,option_C,reply,label,notes",
            )
            fill(tmp / "sheet.csv", tmp / "filled.csv", lambda r: "A")
            labeling.import_csv(job, tmp / "filled.csv", "ayesha")
            labels = labeling.load_human_labels(job, "ayesha")
            actions = labeling.episode_mcq_actions(job, labels, requests)
            for episode_id, action in actions.items():
                self.assertEqual(action, requests[episode_id]["choice_key"]["A"])
            fill(tmp / "sheet.csv", tmp / "none.csv", lambda r: "none")
            labeling.import_csv(job, tmp / "none.csv", "aaliyan")
            none_actions = labeling.episode_mcq_actions(
                job, labeling.load_human_labels(job, "aaliyan"), requests
            )
            self.assertTrue(all(v is None for v in none_actions.values()))

    def test_job_without_a_human_protocol_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            from rlvr_safety.agent_tasks.core import make_task

            job = Path(tmp) / "job"
            write_plan(job, "authoring", "r", [make_task("authoring", "r", "i", {"x": 1}, {})])
            with self.assertRaisesRegex(ValueError, "no human labelling protocol"):
                labeling.export_csv(job, Path(tmp) / "x.csv")


if __name__ == "__main__":
    unittest.main()
