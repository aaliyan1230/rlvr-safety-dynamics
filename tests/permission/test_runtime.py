from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlvr_safety.permission.runtime import compare_runtime, lock_versions, verify_runtime_lock


class RuntimeTests(unittest.TestCase):
    def test_lock_normalizes_names_and_rejects_duplicate_aliases(self):
        self.assertEqual(
            lock_versions("Jinja2==3.1.6\nhuggingface_hub==0.34.4\n"),
            {"jinja2": "3.1.6", "huggingface-hub": "0.34.4"},
        )
        for text in (
            "Foo_Bar==1\nfoo.bar==1",
            "torch==2; python_version > '3'",
            "torch @ https://example.test/wheel",
            "torch==2==3",
        ):
            with self.subTest(text), self.assertRaises(ValueError):
                lock_versions(text)

    def test_full_environment_comparison_detects_transitive_and_cuda_drift(self):
        locked = "torch==2.8.0+cu128\ntokenizers==0.22.2\n"
        self.assertTrue(compare_runtime(locked, locked)["passed"])
        report = compare_runtime(locked, "torch==2.8.0+cu126\nextra==1\n")
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing"], ["tokenizers"])
        self.assertEqual(report["unexpected"], ["extra"])
        self.assertEqual(report["mismatched"]["torch"]["observed"], "2.8.0+cu126")

    def test_failed_runtime_check_keeps_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime-lock.txt").write_text("tokenizers==0.22.2\n")
            with (
                patch(
                    "rlvr_safety.permission.runtime.subprocess.check_output",
                    return_value="tokenizers==0.23.0\n",
                ),
                self.assertRaisesRegex(ValueError, "runtime differs"),
            ):
                verify_runtime_lock(root / "runtime-lock.txt", root / "runtime_check.json")
            report = json.loads((root / "runtime_check.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["lock_sha256"]), 64)

    def test_workload_refuses_drift_before_model_download_or_load(self):
        script = Path(__file__).resolve().parents[2] / "infra/runpod/permission_workload.py"
        spec = importlib.util.spec_from_file_location("permission_workload_test", script)
        workload = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiment.json").write_text('{"models": []}')
            (root / "scenarios.jsonl").write_text("")
            (root / "requests.jsonl").write_text("")
            (root / "runtime-lock.txt").write_text("tokenizers==0.22.2\n")
            with (
                patch.object(workload, "verify_manifest"),
                patch.object(workload, "HFPermissionProvider") as provider,
                patch("sys.argv", [str(script), "--bundle", str(root)]),
                patch(
                    "rlvr_safety.permission.runtime.subprocess.check_output",
                    return_value="tokenizers==0.23.0\n",
                ),
                self.assertRaisesRegex(ValueError, "runtime differs"),
            ):
                workload.main()
            provider.assert_not_called()
            self.assertFalse(json.loads((root / "out/runtime_check.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
