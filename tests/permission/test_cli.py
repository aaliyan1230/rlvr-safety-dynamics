from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.cli import permission as cli
from rlvr_safety.io import write_jsonl
from rlvr_safety.permission.experiment import write_bundle

from .helpers import control, scenario
from .test_experiment import metadata_fetcher, spec, template_fetcher

GATES = {
    "gate_set_id": "cli-test",
    "gates": [
        {"id": "G1", "kind": "records_complete"},
        {"id": "G2", "kind": "no_censoring"},
        {
            "id": "G3",
            "kind": "controls_followed",
            "params": {
                "filter": {"workload": "controls", "id_arm": "semantic"},
                "expected": 6,
                "min_followed": 6,
            },
        },
        {"id": "G4", "kind": "mcq_resolved_by_rule", "params": {"min_fraction": 1.0}},
    ],
    "format_requires": {"executable": ["G3"], "mcq": ["G4"]},
}


def call(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in argv])
    text = out.getvalue()
    return code, (json.loads(text) if text.strip().startswith(("{", "[")) else text), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_jsonl(self.root / "scenarios.jsonl", [scenario()])
        write_jsonl(self.root / "controls.jsonl", [control()])
        (self.root / "gates.json").write_text(json.dumps(GATES))
        (self.root / "envelope.json").write_text("{}")
        (self.root / "spec.json").write_text(json.dumps(spec()))

    def dry_run(self, policy):
        return call(
            "dry-run",
            "--spec",
            self.root / "spec.json",
            "--policy",
            policy,
            "--out-dir",
            self.root / "dry",
            "--root",
            self.root,
        )

    def test_check_bank_exit_codes(self):
        code, report, _ = call(
            "check-bank", "--bank", self.root / "scenarios.jsonl", "--min-wordings", 2
        )
        self.assertEqual((code, report["passed"]), (0, True))
        broken = scenario()
        broken["actions"][1]["effects"] = []
        write_jsonl(self.root / "broken.jsonl", [broken])
        code, report, _ = call("check-bank", "--bank", self.root / "broken.jsonl")
        self.assertEqual((code, report["passed"]), (1, False))
        code, _, err = call("check-bank", "--bank", self.root / "missing.jsonl")
        self.assertEqual(code, 2)

    def test_dry_run_compliant_passes_and_first_listed_fails_the_gates(self):
        code, report, _ = self.dry_run("compliant")
        self.assertEqual((code, report["gates_passed"]), (0, True))
        analysis = json.loads((self.root / "dry/compliant/analysis.json").read_text())
        self.assertAlmostEqual(analysis["sensitivity"]["executable"]["sensitivity"], 1.0)
        self.assertIn("### Gates", (self.root / "dry/compliant/report.md").read_text())
        code, report, _ = self.dry_run("first_listed")
        self.assertEqual(report["gates_passed"], False)

    def test_analyze_reads_a_bundle_and_retrieved_results(self):
        write_bundle(
            self.root / "spec.json",
            self.root / "bundle",
            root=self.root,
            metadata_fetcher=metadata_fetcher,
            template_fetcher=template_fetcher,
        )
        self.dry_run("compliant")
        run_dir = self.root / "retrieved"
        shutil.copytree(self.root / "dry/compliant/benchmark", run_dir / "step0/benchmark")
        code, result, _ = call(
            "analyze",
            "--bundle",
            self.root / "bundle",
            "--run-dir",
            run_dir,
            "--out",
            self.root / "analysis",
        )
        self.assertEqual((code, result["step0"]["gates_passed"]), (0, True))
        report = (self.root / "analysis/report.md").read_text()
        self.assertIn("# Analysis of `permission-test-2026-10-08`", report)
        self.assertIn("## Model `step0`", report)

    def test_author_flow_with_the_file_backend_and_a_command_backend(self):
        job = self.root / "author"
        brief = {
            "brief_id": "dev-resource",
            "family": "resource_allocation",
            "count": 1,
            "requirements": [],
        }
        (self.root / "brief.json").write_text(json.dumps(brief))
        code, planned, _ = call(
            "author", "plan", "--brief", self.root / "brief.json", "--job-dir", job
        )
        self.assertEqual((code, planned["tasks"]), (0, 1))
        record = scenario()
        for key in ("schema_version", "split", "source_id", "review"):
            record.pop(key)
        record["skeleton_id"] = "fresh_skeleton"
        (self.root / "payload.json").write_text(json.dumps({"records": [record]}))
        script = self.root / "answer.py"
        script.write_text(
            "import json,sys\njson.load(sys.stdin)\n"
            f"print(open({str(self.root / 'payload.json')!r}).read())\n"
        )
        code, answered, _ = call(
            "author",
            "run",
            "--job-dir",
            job,
            "--command",
            f"{sys.executable} {script}",
            "--agent",
            "scripted-cli",
            "--model-reported",
            "none",
        )
        self.assertEqual((code, len(answered["answered"])), (0, 1))
        code, counts, _ = call("author", "ingest", "--job-dir", job, "--operator", "tester")
        self.assertEqual(counts["accepted"], 1)
        self.assertEqual(call("author", "audit", "--job-dir", job)[0], 0)
        code, applied, _ = call(
            "author",
            "apply",
            "--job-dir",
            job,
            "--out",
            self.root / "bank.jsonl",
            "--owner",
            "Ayesha",
        )
        self.assertEqual((code, applied["records"], applied["checks_passed"]), (0, 1, True))
        bank = [json.loads(x) for x in (self.root / "bank.jsonl").read_text().splitlines()]
        self.assertEqual(bank[0]["authoring"]["accountable_owner"], "Ayesha")
        self.assertEqual(bank[0]["review"]["status"], "pending")

    def test_judge_and_labelling_flow(self):
        self.dry_run("noisy")
        run = self.root / "dry/noisy/benchmark"
        job = self.root / "judge"
        code, planned, _ = call("judge", "plan", "--run-dir", run, "--job-dir", job)
        self.assertEqual(code, 0)
        self.assertGreater(planned["tasks"], 0)
        script = self.root / "judge.py"
        script.write_text(
            "import json,sys\njson.load(sys.stdin)\n"
            "print(json.dumps({'selected': 'none', 'evidence': 'unclear', 'confidence': 'low'}))\n"
        )
        call(
            "judge",
            "run",
            "--job-dir",
            job,
            "--command",
            f"{sys.executable} {script}",
            "--agent",
            "scripted-cli",
            "--model-reported",
            "none",
        )
        self.assertEqual(
            call("judge", "ingest", "--job-dir", job, "--operator", "t")[1]["missing"], 0
        )
        self.assertEqual(call("judge", "audit", "--job-dir", job)[0], 0)
        code, applied, _ = call(
            "judge",
            "apply",
            "--job-dir",
            job,
            "--run-dir",
            run,
            "--out",
            self.root / "judge_scores.jsonl",
        )
        self.assertGreater(applied["records"], 0)
        code, info, _ = call("label", "export", "--job-dir", job, "--csv", self.root / "sheet.csv")
        self.assertEqual(code, 0)
        text = (self.root / "sheet.csv").read_text().splitlines()
        header, rows = text[0].split(","), text[1:]
        filled = [header and ",".join([r.rsplit(",", 2)[0], "none", ""]) for r in rows]
        (self.root / "filled.csv").write_text("\n".join([text[0], *filled]) + "\n")
        code, imported, _ = call(
            "label",
            "import",
            "--job-dir",
            job,
            "--csv",
            self.root / "filled.csv",
            "--labeller",
            "ayesha",
        )
        self.assertEqual((code, imported["labelled"]), (0, len(rows)))
        code, agree, _ = call(
            "label", "agree", "--job-dir", job, "--first", "ayesha", "--second", "ai"
        )
        self.assertEqual((code, agree["exact_agreement"]), (0, 1.0))

    def test_every_command_group_parses(self):
        parser = cli.build_parser()
        for argv in (
            ["sync", "push", "--repo", "r", "--path", "p", "--run-dir", "d", "--message", "m"],
            ["sync", "verify", "--repo", "r", "--path", "p", "--run-dir", "d", "--revision", "x"],
            ["sync", "pull", "--repo", "r", "--path", "p", "--dest", "d", "--revision", "x"],
            ["bundle", "--spec", "s", "--out-dir", "o"],
        ):
            self.assertTrue(parser.parse_args(argv).func)


if __name__ == "__main__":
    unittest.main()
