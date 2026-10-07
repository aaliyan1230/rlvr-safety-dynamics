from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rlvr_safety.permission import sync


class FakeHub:
    """In-memory dataset repo with the few HfApi calls the sync module uses."""

    def __init__(self, private=True):
        self.private = private
        self.files: dict[str, bytes] = {}
        self.created = []
        self.upload_args = None

    def create_repo(self, repo_id, **kwargs):
        self.created.append((repo_id, kwargs))

    def dataset_info(self, repo_id):
        return SimpleNamespace(private=self.private)

    def upload_folder(
        self, *, folder_path, repo_id, repo_type, path_in_repo, commit_message, ignore_patterns
    ):
        self.upload_args = {
            "repo_type": repo_type,
            "ignore": ignore_patterns,
            "message": commit_message,
        }
        for path in Path(folder_path).rglob("*"):
            relative = str(path.relative_to(folder_path))
            if path.is_file() and not sync._excluded(relative):
                self.files[f"{path_in_repo}/{relative}"] = path.read_bytes()
        return SimpleNamespace(oid="c0ffee")

    def download(self, *, repo_id, repo_type, filename, revision, local_dir):
        if filename not in self.files:
            raise FileNotFoundError(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    def snapshot(self, *, repo_id, repo_type, revision, allow_patterns, local_dir):
        prefix = allow_patterns[0].rstrip("*")
        for name, data in self.files.items():
            if name.startswith(prefix):
                target = Path(local_dir) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        return local_dir


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        (self.run / "step0/benchmark").mkdir(parents=True)
        (self.run / "step0/benchmark/results.jsonl").write_text('{"a": 1}\n')
        (self.run / "summary.json").write_text("{}\n")

    def test_push_creates_a_private_dataset_and_records_the_commit(self):
        hub = FakeHub()
        record = sync.push(self.run, "ic-org/shared", "runs/exp1", message="add exp1", api=hub)
        self.assertEqual(hub.created[0][1]["private"], True)
        self.assertEqual(record["commit"], "c0ffee")
        self.assertEqual(record["files"], 2)
        self.assertIn("runs/exp1/step0/benchmark/results.jsonl", hub.files)
        saved = json.loads((self.run / sync.RECORD_NAME).read_text())
        self.assertEqual(saved["repo_id"], "ic-org/shared")
        manifest = json.loads((self.run / sync.MANIFEST_NAME).read_text())
        self.assertEqual(set(manifest["files"]), {"step0/benchmark/results.jsonl", "summary.json"})
        self.assertEqual(manifest["files"]["summary.json"], hashlib.sha256(b"{}\n").hexdigest())

    def test_public_dataset_is_refused_before_any_upload(self):
        hub = FakeHub(private=False)
        with self.assertRaisesRegex(RuntimeError, "not private"):
            sync.push(self.run, "ic-org/open", "runs/exp1", message="m", api=hub)
        self.assertEqual(hub.files, {})
        self.assertIsNone(hub.upload_args)

    def test_credential_like_files_are_excluded_from_manifest_and_upload(self):
        (self.run / ".env").write_text("RUNPOD_API_KEY=secret")
        (self.run / "id_ed25519").write_text("private")
        (self.run / "known_hosts").write_text("host")
        (self.run / "ssh-failure.json").write_text("{}")
        (self.run / "watchdog.log").write_text("log")
        hub = FakeHub()
        record = sync.push(self.run, "ic-org/shared", "runs/exp1", message="m", api=hub)
        self.assertEqual(
            set(record["excluded"]),
            {".env", "id_ed25519", "known_hosts", "ssh-failure.json", "watchdog.log"},
        )
        for name in hub.files:
            self.assertNotIn(".env", name)
            self.assertNotIn("id_ed25519", name)
        self.assertIn(".env*", hub.upload_args["ignore"])

    def test_unsafe_repo_paths_are_rejected(self):
        for path in ("", "/abs", "../x", "a/../../b"):
            with self.subTest(path), self.assertRaises(ValueError):
                sync.push(self.run, "ic-org/shared", path, message="m", api=FakeHub())

    def test_empty_folder_is_not_uploaded(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "nothing to upload"):
            sync.push(empty, "ic-org/shared", "runs/x", message="m", api=FakeHub())

    def test_verify_detects_missing_and_altered_files(self):
        hub = FakeHub()
        sync.push(self.run, "ic-org/shared", "runs/exp1", message="m", api=hub)
        ok = sync.verify(self.run, "ic-org/shared", "runs/exp1", "c0ffee", download=hub.download)
        self.assertTrue(ok["ok"], ok)
        hub.files["runs/exp1/summary.json"] = b"tampered"
        del hub.files["runs/exp1/step0/benchmark/results.jsonl"]
        bad = sync.verify(self.run, "ic-org/shared", "runs/exp1", "c0ffee", download=hub.download)
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["mismatched"], ["summary.json"])
        self.assertEqual(bad["missing"], ["step0/benchmark/results.jsonl"])

    def test_pull_restores_the_folder_and_refuses_to_overwrite(self):
        hub = FakeHub()
        sync.push(self.run, "ic-org/shared", "runs/exp1", message="m", api=hub)
        dest = self.root / "pulled"
        sync.pull("ic-org/shared", "runs/exp1", dest, "c0ffee", snapshot=hub.snapshot)
        self.assertEqual((dest / "summary.json").read_text(), "{}\n")
        self.assertTrue((dest / "step0/benchmark/results.jsonl").is_file())
        with self.assertRaises(FileExistsError):
            sync.pull("ic-org/shared", "runs/exp1", dest, "c0ffee", snapshot=hub.snapshot)
        with self.assertRaises(FileNotFoundError):
            sync.pull(
                "ic-org/shared", "runs/other", self.root / "x", "c0ffee", snapshot=hub.snapshot
            )


if __name__ == "__main__":
    unittest.main()
