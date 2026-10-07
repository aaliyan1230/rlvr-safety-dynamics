from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlvr_safety.io import sha256_file, write_jsonl
from rlvr_safety.permission.experiment import write_bundle
from rlvr_safety.provenance import verify_manifest

from .helpers import control, scenario
from .test_experiment import envelope, metadata_fetcher, spec, template_fetcher

HERE = Path(__file__).resolve().parents[2] / "infra/runpod"
sys.path.insert(0, str(HERE))
try:
    loaded = importlib.util.spec_from_file_location("permission_run", HERE / "permission_run.py")
    runner = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(runner)
finally:
    sys.path.remove(str(HERE))

GPU_ID = "NVIDIA A100-SXM4-80GB"


def gpu_catalog(availability="LOW", price=1.59, datacenters=("US-KS-2",)):
    return {
        "gpus": [
            {
                "id": GPU_ID,
                "price": {"secure": price},
                "dataCenters": [{"id": d, "availability": availability} for d in datacenters],
            }
        ]
    }


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_jsonl(self.root / "scenarios.jsonl", [scenario()])
        write_jsonl(self.root / "controls.jsonl", [control()])
        (self.root / "gates.json").write_text('{"gates": []}\n')
        test_envelope = envelope()
        test_envelope["usage_window_start"] = "2026-10-01T00:00:00Z"
        test_envelope["usage_baseline_usd"] = 0.5
        test_envelope["valid_through"] = "2999-01-01"
        (self.root / "envelope.json").write_text(json.dumps(test_envelope))
        test_spec = spec()
        test_spec["launch"]["gpu"] = GPU_ID
        (self.root / "spec.json").write_text(json.dumps(test_spec))
        write_bundle(
            self.root / "spec.json",
            self.root / "bundle",
            root=self.root,
            metadata_fetcher=metadata_fetcher,
            template_fetcher=template_fetcher,
        )
        self.bundle = self.root / "bundle"
        self.key = self.root / "key"
        self.key.write_text("private fixture")
        self.key.with_suffix(".pub").write_text("public fixture")
        self.args = SimpleNamespace(
            bundle=self.bundle,
            root=self.root,
            balance_usd=50.0,
            balance_source="test",
            yes=True,
            private_key=self.key,
            output=self.root / "run",
            deadline=None,
        )

    def api(self, *, pods=None, volumes=None, catalog=None, spent=1.0, created=None):
        def request(path, params=None, *, method="GET", body=None):
            if path == "/network-volumes":
                return {"networkVolumes": volumes or []}
            if path == "/catalog/gpus":
                return catalog or gpu_catalog()
            if path == "/billing":
                return {"metadata": {"totals": {"totalAmount": 0.5 + spent}}}
            if path == "/account/ssh-keys":
                return {"keys": ["public fixture"]}
            if path == "/pods" and method == "POST":
                if created is not None:
                    created.append(body)
                return {"id": "paid-pod", "cost": 1.59}
            if path == "/pods/paid-pod":
                return {
                    "id": "paid-pod",
                    "status": "RUNNING",
                    "ssh": {"direct": {"host": "unused", "port": 22}},
                }
            raise AssertionError(path)

        return request

    def patched(self, request, pods=None):
        return (
            patch.object(runner, "list_pods", return_value=pods or []),
            patch.object(runner, "request", side_effect=request),
        )

    # ---- preflight -------------------------------------------------------------------
    def test_preflight_passes_and_reports_placement_and_spend(self):
        a, b = self.patched(self.api(spent=2.5))
        with a, b:
            report = runner.preflight(self.args)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["placement"]["datacenter"], "US-KS-2")
        self.assertAlmostEqual(report["usage_spent_under_envelope_usd"], 2.5)
        self.assertEqual(report["inventory"], {"pods": [], "volumes": []})

    def test_preflight_reports_each_blocking_problem_without_creating_anything(self):
        cases = {
            "resources": (self.api(volumes=[{"id": "v"}]), [{"id": "p"}], "account has resources"),
            "price": (self.api(catalog=gpu_catalog(price=9.0)), [], "exceeds the approved ceiling"),
            "capacity": (self.api(catalog=gpu_catalog(availability="NONE")), [], "no capacity"),
            "wrong dc": (
                self.api(catalog=gpu_catalog(datacenters=("EU-FR-1",))),
                [],
                "no capacity",
            ),
            "budget": (self.api(spent=19.0), [], "total_budget"),
        }
        for name, (request, pods, expected) in cases.items():
            a, b = self.patched(request, pods)
            with self.subTest(name), a, b:
                report = runner.preflight(self.args)
                self.assertFalse(report["ok"])
                self.assertTrue(any(expected in p for p in report["problems"]), report["problems"])
        self.args.balance_usd = 3.0
        a, b = self.patched(self.api())
        with a, b:
            self.assertFalse(runner.preflight(self.args)["ok"])

    def test_preflight_refuses_a_changed_bundle(self):
        (self.bundle / "gates.json").write_text("{}\n")
        a, b = self.patched(self.api())
        with a, b, self.assertRaisesRegex(ValueError, "checksum mismatch"):
            runner.preflight(self.args)

    def test_choose_datacenter_prefers_best_availability_among_allowed(self):
        launch = {
            **spec()["launch"],
            "gpu": GPU_ID,
            "datacenters": ["US-MD-1", "US-KS-2", "US-WA-1"],
        }
        catalog = {
            "gpus": [
                {
                    "id": GPU_ID,
                    "price": {"secure": 1.59},
                    "dataCenters": [
                        {"id": "US-KS-2", "availability": "LOW"},
                        {"id": "US-MD-1", "availability": "HIGH"},
                        {"id": "EU-X", "availability": "HIGH"},
                    ],
                }
            ]
        }
        with patch.object(runner, "request", return_value=catalog):
            chosen = runner.choose_datacenter(launch)
        self.assertEqual(chosen["datacenter"], "US-MD-1")

    def test_spend_is_current_usage_minus_the_recorded_baseline(self):
        with patch.object(runner, "request", side_effect=self.api(spent=3.25)):
            value = runner.spent_under_envelope(
                {"usage_window_start": "2026-10-01T00:00:00Z", "usage_baseline_usd": 0.5},
                date(2026, 10, 9),
            )
        self.assertAlmostEqual(value, 3.25)

    # ---- launch ----------------------------------------------------------------------
    def retrieved_archive(self, complete=True):
        files = {
            "step0/benchmark/results.jsonl": b"evidence\n",
            "run_summary.json": json.dumps({"completed": complete}).encode(),
        }
        files["step0/benchmark/artifact_manifest.json"] = json.dumps(
            {
                "files": {
                    "results.jsonl": hashlib.sha256(
                        files["step0/benchmark/results.jsonl"]
                    ).hexdigest()
                }
            }
        ).encode()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return buffer.getvalue()

    def ssh_stub(self, exported, exit_code=b"0\n"):
        def ssh(connection, args, command, **kwargs):
            output = b""
            if command.startswith("sha256sum"):
                output = (sha256_file(self.root / "bundle.tar.gz") + " bundle\n").encode()
            elif command.startswith("cat ") and "exit.code" in command:
                output = exit_code
            elif "tar -czf -" in command:
                output = exported
            return SimpleNamespace(returncode=0, stdout=output)

        return ssh

    def launch_patches(self, request, ssh, pods=None):
        return (
            patch.object(runner, "list_pods", return_value=pods or []),
            patch.object(runner, "request", side_effect=request),
            patch.object(runner.subprocess, "Popen", return_value=SimpleNamespace(pid=123)),
            patch.object(runner, "ssh", side_effect=ssh),
            patch.object(runner.pilot, "cleanup"),
        )

    def run_with(self, request, ssh):
        a, b, c, d, e = self.launch_patches(request, ssh)
        with a, b, c, d, e as cleanup:
            runner.run(self.args)
            return cleanup

    def test_success_uses_container_disk_only_and_verifies_results_and_cleanup(self):
        created = []
        cleanup = self.run_with(self.api(created=created), self.ssh_stub(self.retrieved_archive()))
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertTrue(
            state["passed"] and state["outputs_retrieved"] and state["cleanup_verified"]
        )
        verify_manifest(self.args.output / "retrieved/step0/benchmark/artifact_manifest.json")
        cleanup.assert_called_once()
        body = created[0]
        self.assertNotIn("mounts", body)
        self.assertEqual(body["disk"], 50)
        self.assertEqual(body["dataCenterIds"], ["US-KS-2"])
        self.assertEqual(body["gpu"]["count"], 1)
        postflight = json.loads((self.args.output / "postflight.json").read_text())
        self.assertTrue(postflight["empty"])
        self.assertIn("usage_spent_under_envelope_usd", postflight)
        preflight = json.loads((self.args.output / "preflight.json").read_text())
        self.assertTrue(preflight["ok"])
        self.assertEqual(preflight["balance_source"], "test")

    def test_incomplete_run_summary_is_not_a_pass_but_still_cleans_up(self):
        a, b, c, d, e = self.launch_patches(
            self.api(), self.ssh_stub(self.retrieved_archive(complete=False))
        )
        with a, b, c, d, e as cleanup, self.assertRaisesRegex(RuntimeError, "incomplete run"):
            runner.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertFalse(state["passed"])
        self.assertTrue(state["cleanup_verified"])
        cleanup.assert_called_once()

    def test_failed_preflight_creates_no_resources_and_keeps_the_record(self):
        created = []
        a, b, c, d, e = self.launch_patches(
            self.api(created=created, spent=19.5), self.ssh_stub(b"")
        )
        with a, b, c, d, e as cleanup, self.assertRaisesRegex(RuntimeError, "preflight failed"):
            runner.run(self.args)
        self.assertEqual(created, [])
        cleanup.assert_not_called()
        self.assertFalse(json.loads((self.args.output / "preflight.json").read_text())["ok"])
        self.assertFalse((self.args.output / "state.json").exists())

    def test_launch_requires_explicit_yes(self):
        self.args.yes = False
        with patch.object(runner, "request") as api, self.assertRaisesRegex(ValueError, "--yes"):
            runner.run(self.args)
        api.assert_not_called()

    def test_expired_overall_deadline_prevents_resource_creation(self):
        self.args.deadline = runner.time.time() - 1
        created = []
        a, b, c, d, e = self.launch_patches(self.api(created=created), self.ssh_stub(b""))
        with a, b, c, d, e, self.assertRaisesRegex(ValueError, "overall deadline"):
            runner.run(self.args)
        self.assertEqual(created, [])

    def test_quote_above_ceiling_saves_ids_and_cleans_up(self):
        base = self.api()

        def request(path, params=None, *, method="GET", body=None):
            if path == "/pods" and method == "POST":
                return {"id": "paid-pod", "cost": 99}
            return base(path, params, method=method, body=body)

        a, b, c, d, e = self.launch_patches(request, self.ssh_stub(b""))
        with a, b, c, d, e as cleanup, self.assertRaisesRegex(RuntimeError, "approved ceiling"):
            runner.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertEqual(state["pod_ids"], ["paid-pod"])
        self.assertTrue(state["cleanup_verified"])
        cleanup.assert_called_once()

    def test_ambiguous_create_is_not_retried_and_keeps_the_cleanup_name(self):
        base = self.api()

        def request(path, params=None, *, method="GET", body=None):
            if path == "/pods" and method == "POST":
                raise RuntimeError("lost create reply")
            return base(path, params, method=method, body=body)

        a, b, c, d, e = self.launch_patches(request, self.ssh_stub(b""))
        with a, b as api, c, d, e as cleanup, self.assertRaisesRegex(RuntimeError, "lost create"):
            runner.run(self.args)
        self.assertEqual(sum(call.args[0] == "/pods" for call in api.call_args_list), 1)
        self.assertTrue(cleanup.call_args.args[0]["name"].startswith("rlvr-perm-"))

    def test_provider_rejection_is_recorded(self):
        base = self.api()
        problem = {"title": "Bad Request", "detail": "no available capacity"}

        def request(path, params=None, *, method="GET", body=None):
            if path == "/pods" and method == "POST":
                raise runner.ApiError("RunPod POST /pods returned HTTP 400", 400, problem)
            return base(path, params, method=method, body=body)

        a, b, c, d, e = self.launch_patches(request, self.ssh_stub(b""))
        with a, b, c, d, e as cleanup, self.assertRaises(runner.ApiError):
            runner.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertEqual(state["api_error"], {"status": 400, "problem": problem})
        cleanup.assert_called_once()

    def test_failed_retrieval_still_cleans_up(self):
        def ssh(connection, args, command, **kwargs):
            if command.startswith("sha256sum"):
                return SimpleNamespace(
                    returncode=0, stdout=(sha256_file(self.root / "bundle.tar.gz") + "\n").encode()
                )
            if "tar -czf -" in command:
                raise subprocess.TimeoutExpired("ssh", 90)
            return SimpleNamespace(returncode=0, stdout=b"0\n")

        a, b, c, d, e = self.launch_patches(self.api(), ssh)
        with a, b, c, d, e as cleanup, self.assertRaises(subprocess.TimeoutExpired):
            runner.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertFalse(state["passed"])
        self.assertTrue(state["cleanup_verified"])
        cleanup.assert_called_once()

    def test_resources_remaining_after_cleanup_are_an_error(self):
        base = self.api()
        calls = {"n": 0}

        def listing():
            calls["n"] += 1
            return [] if calls["n"] == 1 else [{"id": "leftover"}]

        a, b, c, d, e = self.launch_patches(base, self.ssh_stub(self.retrieved_archive()))
        with (
            a,
            b,
            c,
            d,
            e,
            patch.object(runner, "list_pods", side_effect=listing),
            self.assertRaisesRegex(RuntimeError, "resources remain"),
        ):
            runner.run(self.args)
        postflight = json.loads((self.args.output / "postflight.json").read_text())
        self.assertFalse(postflight["empty"])

    # ---- helpers ---------------------------------------------------------------------
    def test_result_archive_rejects_traversal_and_symlinks(self):
        for filename, typecode in (("../../outside", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                member = tarfile.TarInfo(filename)
                member.type = typecode
                member.linkname = "/etc/passwd"
                archive.addfile(member)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                runner.extract_results(buffer.getvalue(), self.root / "extracted")
        self.assertFalse((self.root / "extracted").exists())

    def test_remote_extraction_preserves_bytes_without_archive_ownership(self):
        target = self.root / "remote"
        target.mkdir()
        shutil.copyfile(self.root / "bundle.tar.gz", target / "bundle.tar.gz")
        subprocess.run(["bash", "-c", runner.extract_bundle_command(str(target))], check=True)
        verify_manifest(target / "bundle_manifest.json")

    def test_archive_must_match_the_prepared_bundle(self):
        runner.verify_archive(self.root / "bundle.tar.gz", self.bundle)
        (self.bundle / "requests.jsonl").write_text("changed\n")
        manifest_path = self.bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["requests.jsonl"] = sha256_file(self.bundle / "requests.jsonl")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "archive contents"):
            runner.verify_archive(self.root / "bundle.tar.gz", self.bundle)

    def test_bootstrap_installs_pinned_packages_and_runs_the_workload(self):
        script = runner.bootstrap("/workspace/x")
        self.assertIn("runtime-pins.txt", script)
        self.assertIn("permission_workload.py --bundle /workspace/x", script)
        self.assertNotIn("HF_TOKEN", script)
        self.assertNotIn("RUNPOD_API_KEY", script)


if __name__ == "__main__":
    unittest.main()
