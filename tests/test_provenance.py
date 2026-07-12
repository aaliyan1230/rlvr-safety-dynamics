from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.io import sha256_file
from rlvr_safety.provenance import ProvenanceError, verify_manifest


class ProvenanceTests(unittest.TestCase):
    def test_verifies_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            artifact = root / "artifact.txt"
            artifact.write_text("original", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"files": {artifact.name: sha256_file(artifact)}}),
                encoding="utf-8",
            )
            verify_manifest(manifest)
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaises(ProvenanceError):
                verify_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
