from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.io import read_jsonl


REPO = Path(__file__).resolve().parents[1]


class ParaphraseGeneratorTests(unittest.TestCase):
    def test_mock_generation_preserves_source_option_mapping(self) -> None:
        sources = list(read_jsonl(REPO / "data/choice_eval_targeted.jsonl"))[:3]
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source_path = tmp / "sources.jsonl"
            out_path = tmp / "out.jsonl"
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/gemini_generate_paraphrases.py"),
                    "--choice-prompts",
                    str(source_path),
                    "--out",
                    str(out_path),
                    "--per-source",
                    "1",
                    "--pack-id",
                    "test_pack",
                    "--mock",
                ],
                check=True,
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            generated = list(read_jsonl(out_path))

        self.assertEqual(
            [row["score_key"] for row in generated],
            [row["score_key"] for row in sources],
        )
        self.assertEqual(len({row["id"] for row in generated}), 3)
        self.assertTrue(all("test_pack" in row["id"] for row in generated))


if __name__ == "__main__":
    unittest.main()
