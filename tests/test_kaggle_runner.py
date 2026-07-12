from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlvr_safety.io import read_jsonl, write_jsonl
from rlvr_safety.kaggle_reproduction import select_prompt_rows, validate_runtime

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "kaggle/factorial_v1/run_factorial.py"
PILOT_RUNNER = REPO / "kaggle/tulu_endpoint_v1/run_endpoint_pilot.py"


class KaggleRunnerTests(unittest.TestCase):
    @staticmethod
    def load_runner(path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_configure_package_path_adds_src_directory(self) -> None:
        module = self.load_runner(RUNNER, "factorial_kaggle_runner")

        with tempfile.TemporaryDirectory() as raw_tmp:
            dataset = Path(raw_tmp) / "dataset"
            package_dir = dataset / "src/rlvr_safety"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            with patch.object(module, "INPUT", Path(raw_tmp)):
                source_root = module.configure_package_path()
            self.assertEqual(source_root, dataset / "src")
            self.assertEqual(Path(sys.path[0]), dataset / "src")
            sys.path.pop(0)

    def test_endpoint_pilot_filter_is_explicit_and_writes_selected_rows(self) -> None:
        module = self.load_runner(PILOT_RUNNER, "endpoint_kaggle_runner")
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            package_dir = root / "dataset/src/rlvr_safety"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            source = root / "prompts.jsonl"
            write_jsonl(
                source,
                [
                    {
                        "id": "keep",
                        "source_id": "source",
                        "wording_id": "original",
                        "option_order": "012",
                    },
                    {
                        "id": "drop",
                        "source_id": "source",
                        "wording_id": "p1",
                        "option_order": "012",
                    },
                ],
            )
            config = {
                "prompt_filter": {
                    "wording_ids": ["original"],
                    "option_orders": ["012"],
                },
                "design": {"conditions_per_checkpoint": 1, "source_items": 1},
            }
            with (
                patch.object(module, "INPUT", root),
                patch.object(module, "OUT_DIR", root / "output"),
            ):
                selected, output = module.filtered_prompts(source, config)
            self.assertEqual([row["id"] for row in selected], ["keep"])
            self.assertEqual([row["id"] for row in read_jsonl(output)], ["keep"])
            sys.path.pop(0)

    def test_historical_reproduction_filters_match_declared_cells(self) -> None:
        prompts = list(read_jsonl(REPO / "data/choice_factorial_v1.jsonl"))
        cases = (
            ("historical_stage_reproduction_v1.json", 72),
            ("historical_paraphrase_reproduction_v1.json", 72),
        )
        for config_name, expected in cases:
            config = json.loads(
                (REPO / "configs/experiments" / config_name).read_text(encoding="utf-8")
            )
            selected = select_prompt_rows(prompts, config["prompt_filter"])
            self.assertEqual(len(selected), expected)
            self.assertEqual(
                {row["wording_id"] for row in selected}, {"original", "p1", "p2"}
            )
            self.assertEqual(len({row["source_id"] for row in selected}), 24)

    def test_historical_runtime_gate_detects_package_drift(self) -> None:
        config = {
            "expected_runtime": {
                "python": "3.12.13",
                "packages": {"transformers": "5.0.0"},
            }
        }
        with patch(
            "rlvr_safety.kaggle_reproduction.platform.python_version",
            return_value="3.12.13",
        ):
            validate_runtime(config, {"transformers": "5.0.0"})
            with self.assertRaisesRegex(RuntimeError, "package runtime mismatch"):
                validate_runtime(config, {"transformers": "5.12.1"})


if __name__ == "__main__":
    unittest.main()
