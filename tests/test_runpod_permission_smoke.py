from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlvr_safety.io import sha256_file
from rlvr_safety.provenance import verify_manifest

HERE = Path(__file__).resolve().parents[1] / "infra/runpod"
sys.path.insert(0, str(HERE))
try:
    spec = importlib.util.spec_from_file_location(
        "runpod_permission_smoke", HERE / "permission_smoke.py"
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
finally:
    sys.path.remove(str(HERE))


class RunPodPermissionSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = json.loads(
            (Path(__file__).parent / "fixtures/permission_scenario.json").read_text()
        )
        source["source_id"] = "dev_resource_transfer"
        self.bank = self.root / "bank.jsonl"
        self.bank.write_text(json.dumps(source) + "\n")
        self.metadata = self.root / "metadata.json"
        self.metadata.write_text(json.dumps({"id": "example/model", "sha": "a" * 40}))
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text(json.dumps({"chat_template": "native"}))
        self.bundle = self.root / "bundle"
        smoke.prepare(self.bank, self.metadata, self.tokenizer, self.bundle)
        self.key = self.root / "key"
        self.key.write_text("unused private fixture")
        self.key.with_suffix(".pub").write_text("public fixture")
        self.args = SimpleNamespace(
            yes=True, bundle=self.bundle, private_key=self.key, output=self.root / "run"
        )

    def test_bundle_has_exact_prompt_budget_and_no_credentials(self):
        verify_manifest(self.bundle / "bundle_manifest.json")
        requests = [
            json.loads(line) for line in (self.bundle / "requests.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(requests), 26)
        self.assertEqual(len({r["episode_id"] for r in requests}), 26)
        config = json.loads((self.bundle / "smoke_config.json").read_text())
        self.assertEqual(len(config["controls"]), 10)
        self.assertFalse(config["scientific_interpretation_allowed"])
        with tarfile.open(self.root / "bundle.tar.gz") as archive:
            names = archive.getnames()
            self.assertFalse(any(".env" in n or "private" in n for n in names))
            self.assertIn("src/rlvr_safety/permission_generation.py", names)

    def test_unexpected_resources_prevent_paid_mutations(self):
        with (
            patch.object(smoke, "list_pods", return_value=[{"id": "unrelated"}]),
            patch.object(smoke, "request") as request,
            self.assertRaisesRegex(RuntimeError, "account has resources"),
        ):
            smoke.run(self.args)
        request.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_accepted_price_rejection_saves_resource_ids_and_cleans_up(self):
        def request(path, params=None, *, method="GET", body=None):
            if path == "/network-volumes":
                return {"id": "volume"} if method == "POST" else {"networkVolumes": []}
            if path == "/catalog/gpus":
                return {
                    "gpus": [
                        {
                            "id": smoke.pilot.GPU,
                            "price": {"secure": 1.59},
                            "dataCenters": [{"id": "EUR-IS-1", "availability": "LOW"}],
                        }
                    ]
                }
            if path == "/account/ssh-keys":
                return {"keys": ["public fixture"]}
            if path == "/pods":
                return {"id": "paid-pod", "cost": 99}
            raise AssertionError(path)

        with (
            patch.object(smoke, "list_pods", return_value=[]),
            patch.object(smoke, "request", side_effect=request),
            patch.object(smoke.subprocess, "Popen", return_value=SimpleNamespace(pid=123)),
            patch.object(smoke.pilot, "cleanup") as cleanup,
            self.assertRaisesRegex(RuntimeError, "approved ceiling"),
        ):
            smoke.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertEqual(state["pod_ids"], ["paid-pod"])
        self.assertEqual(state["volume_id"], "volume")
        self.assertTrue(state["cleanup_verified"])
        cleanup.assert_called_once()

    def test_result_archive_rejects_traversal_and_symlinks(self):
        for filename, typecode in (("../../outside", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                member = tarfile.TarInfo(filename)
                member.type = typecode
                member.linkname = "/etc/passwd"
                archive.addfile(member)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                smoke.extract_results(buffer.getvalue(), self.root / "extracted")
        self.assertFalse((self.root / "extracted").exists())

    def test_changed_prepared_bundle_cannot_launch_an_older_archive(self):
        config_path = self.bundle / "smoke_config.json"
        config_path.write_text(config_path.read_text() + "\n")
        manifest_path = self.bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["smoke_config.json"] = sha256_file(config_path)
        manifest_path.write_text(json.dumps(manifest))
        with (
            patch.object(smoke, "list_pods") as inventory,
            self.assertRaisesRegex(ValueError, "archive contents"),
        ):
            smoke.run(self.args)
        inventory.assert_not_called()

    def test_remote_extraction_preserves_bytes_without_archive_ownership(self):
        target = self.root / "remote"
        target.mkdir()
        shutil.copyfile(self.root / "bundle.tar.gz", target / "bundle.tar.gz")
        subprocess.run(["bash", "-c", smoke.extract_bundle_command(str(target))], check=True)
        verify_manifest(target / "bundle_manifest.json")

    def test_expired_overall_deadline_prevents_resource_creation(self):
        self.args.deadline = smoke.time.time() - 1
        with (
            patch.object(smoke, "request") as api,
            self.assertRaisesRegex(ValueError, "overall deadline"),
        ):
            smoke.run(self.args)
        api.assert_not_called()

    def _launch_request(self, path, params=None, *, method="GET", body=None):
        if path == "/network-volumes":
            return {"id": "volume"} if method == "POST" else {"networkVolumes": []}
        if path == "/catalog/gpus":
            return {
                "gpus": [
                    {
                        "id": smoke.pilot.GPU,
                        "price": {"secure": 1.59},
                        "dataCenters": [{"id": "EUR-IS-1", "availability": "LOW"}],
                    }
                ]
            }
        if path == "/account/ssh-keys":
            return {"keys": ["public fixture"]}
        if path == "/pods":
            return {"id": "paid-pod", "cost": 1.59}
        if path == "/pods/paid-pod":
            return {
                "id": "paid-pod",
                "status": "RUNNING",
                "ssh": {"direct": {"host": "unused", "port": 22}},
            }
        raise AssertionError(path)

    def _retrieved_archive(self):
        files = {
            "benchmark/results.jsonl": b"fixture evidence\n",
            "smoke_result.json": json.dumps({"passed": True}).encode(),
        }
        files["benchmark/artifact_manifest.json"] = json.dumps(
            {
                "files": {
                    "results.jsonl": smoke.hashlib.sha256(
                        files["benchmark/results.jsonl"]
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

    def test_success_requires_verified_local_results_and_cleanup(self):
        exported = self._retrieved_archive()

        def ssh(connection, args, command, **kwargs):
            output = b""
            if command.startswith("sha256sum"):
                output = (sha256_file(self.root / "bundle.tar.gz") + " bundle\n").encode()
            elif command.startswith("cat ") and "exit.code" in command:
                output = b"0\n"
            elif "tar -czf -" in command:
                output = exported
            return SimpleNamespace(returncode=0, stdout=output)

        with (
            patch.object(smoke, "list_pods", return_value=[]),
            patch.object(smoke, "request", side_effect=self._launch_request),
            patch.object(smoke.subprocess, "Popen", return_value=SimpleNamespace(pid=123)),
            patch.object(smoke, "ssh", side_effect=ssh),
            patch.object(smoke.pilot, "cleanup") as cleanup,
        ):
            smoke.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertTrue(state["passed"])
        self.assertTrue(state["outputs_retrieved"])
        self.assertTrue(state["cleanup_verified"])
        verify_manifest(self.args.output / "retrieved/benchmark/artifact_manifest.json")
        cleanup.assert_called_once()

    def test_successful_workload_with_failed_retrieval_still_cleans_up(self):
        def ssh(connection, args, command, **kwargs):
            if command.startswith("sha256sum"):
                return SimpleNamespace(
                    returncode=0, stdout=(sha256_file(self.root / "bundle.tar.gz") + "\n").encode()
                )
            if "tar -czf -" in command:
                raise smoke.subprocess.TimeoutExpired("ssh", 90)
            return SimpleNamespace(returncode=0, stdout=b"0\n")

        with (
            patch.object(smoke, "list_pods", return_value=[]),
            patch.object(smoke, "request", side_effect=self._launch_request),
            patch.object(smoke.subprocess, "Popen", return_value=SimpleNamespace(pid=123)),
            patch.object(smoke, "ssh", side_effect=ssh),
            patch.object(smoke.pilot, "cleanup") as cleanup,
            self.assertRaises(smoke.subprocess.TimeoutExpired),
        ):
            smoke.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertFalse(state["passed"])
        self.assertTrue(state["cleanup_verified"])
        cleanup.assert_called_once()

    def test_ambiguous_create_is_not_retried_and_keeps_unique_cleanup_name(self):
        def request(path, params=None, *, method="GET", body=None):
            if path == "/pods" and method == "POST":
                raise RuntimeError("lost create reply")
            return self._launch_request(path, params, method=method, body=body)

        with (
            patch.object(smoke, "list_pods", return_value=[]),
            patch.object(smoke, "request", side_effect=request) as api,
            patch.object(smoke.subprocess, "Popen", return_value=SimpleNamespace(pid=123)),
            patch.object(smoke.pilot, "cleanup") as cleanup,
            self.assertRaisesRegex(RuntimeError, "lost create reply"),
        ):
            smoke.run(self.args)
        self.assertEqual(sum(c.args[0] == "/pods" for c in api.call_args_list), 1)
        state = cleanup.call_args.args[0]
        self.assertTrue(state["name"].startswith("rlvr-permission-smoke-"))
        self.assertEqual(state["volume_id"], "volume")

    def test_provider_rejection_records_problem_and_cleans_up(self):
        problem = {"title": "Bad Request", "detail": "no available capacity"}

        def request(path, params=None, *, method="GET", body=None):
            if path == "/pods" and method == "POST":
                raise smoke.ApiError("RunPod POST /pods returned HTTP 400", 400, problem)
            return self._launch_request(path, params, method=method, body=body)

        with (
            patch.object(smoke, "list_pods", return_value=[]),
            patch.object(smoke, "request", side_effect=request),
            patch.object(smoke.subprocess, "Popen", return_value=SimpleNamespace(pid=123)),
            patch.object(smoke.pilot, "cleanup") as cleanup,
            self.assertRaises(smoke.ApiError),
        ):
            smoke.run(self.args)
        state = json.loads((self.args.output / "state.json").read_text())
        self.assertEqual(state["api_error"], {"status": 400, "problem": problem})
        self.assertFalse(state["passed"])
        self.assertTrue(state["cleanup_verified"])
        cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
