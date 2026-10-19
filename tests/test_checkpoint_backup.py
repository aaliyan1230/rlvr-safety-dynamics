from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rlvr_safety.checkpoint_backup import backup_private


@unittest.skipIf(
    importlib.util.find_spec("huggingface_hub") is None,
    "optional Hugging Face integration dependency is not installed",
)
class PrivateBackupTests(unittest.TestCase):
    def test_private_commit_is_downloaded_at_exact_revision_and_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "responses.jsonl").write_text('{"fixture":true}\n')
            remote = root / "remote"
            remote.mkdir()
            api = Mock()
            api.repo_info.return_value = SimpleNamespace(private=True)

            def commit(**kwargs):
                self.assertEqual(kwargs["repo_id"], "test-org/private-evidence")
                for operation in kwargs["operations"]:
                    content = operation.path_or_fileobj
                    data = content if isinstance(content, bytes) else Path(content).read_bytes()
                    (remote / Path(operation.path_in_repo).name).write_bytes(data)
                return SimpleNamespace(oid="a" * 40)

            api.create_commit.side_effect = commit

            def download(repo, filename, **kwargs):
                self.assertEqual(repo, "test-org/private-evidence")
                self.assertEqual(kwargs["revision"], "a" * 40)
                return str(remote / Path(filename).name)

            receipt = root / "receipt.json"
            result = backup_private(
                evidence,
                repository="test-org/private-evidence",
                prefix="test-campaign/batch",
                receipt=receipt,
                api=api,
                downloader=download,
            )
            self.assertTrue(result["private_verified"] and result["download_verification"])
            self.assertEqual(json.loads(receipt.read_text()), result)
            (remote / "evidence.tar.gz").write_bytes(b"corrupt")
            api.create_commit.side_effect = lambda **kwargs: SimpleNamespace(oid="a" * 40)
            with self.assertRaisesRegex(ValueError, "verification failed"):
                backup_private(
                    evidence,
                    repository="test-org/private-evidence",
                    prefix="test-campaign/batch",
                    receipt=root / "failed.json",
                    api=api,
                    downloader=download,
                )
            self.assertFalse((root / "failed.json").exists())

    def test_public_repository_or_credentials_refuse_before_upload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "responses.jsonl").write_text("{}\n")
            api = Mock()
            api.repo_info.return_value = SimpleNamespace(private=False)
            with self.assertRaisesRegex(ValueError, "not private"):
                backup_private(
                    evidence,
                    repository="test-org/private-evidence",
                    prefix="campaign/batch",
                    receipt=root / "receipt.json",
                    api=api,
                    downloader=Mock(),
                )
            api.create_commit.assert_not_called()
            api.repo_info.return_value = SimpleNamespace(private=True)
            (evidence / ".env").write_text("NEVER_UPLOAD=fixture")
            with self.assertRaisesRegex(ValueError, "credential"):
                backup_private(
                    evidence,
                    repository="test-org/private-evidence",
                    prefix="campaign/batch",
                    receipt=root / "receipt.json",
                    api=api,
                    downloader=Mock(),
                )
            api.create_commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
