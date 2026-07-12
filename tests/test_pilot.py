from __future__ import annotations

import copy
import unittest

from rlvr_safety.choice import SYSTEM_PROMPT
from rlvr_safety.pilot import (
    PilotValidationError,
    analyze_endpoint_pilot,
    render_endpoint_pilot_markdown,
)


def fixture():
    config = {
        "experiment_id": "pilot",
        "status": "throughput_and_format_feasibility_only",
        "models": [
            {"id": "m0", "step": 0, "name": "repo/base", "revision": "a" * 40},
            {"id": "m1", "step": 10, "name": "repo/final", "revision": "b" * 40},
        ],
        "prompt_filter": {"wording_ids": ["original"], "option_orders": ["012", "210"]},
        "design": {"conditions_per_checkpoint": 2, "source_items": 1},
        "generation": {
            "attention_mask_mode": "explicit_all_ones",
            "seed": 0,
            "system_prompt": SYSTEM_PROMPT,
        },
        "runtime": {
            "accelerator": "NvidiaTeslaT4",
            "transformers": "4.57.6",
            "accelerate": "1.13.0",
        },
        "gates": {
            "expected_rows": 4,
            "max_gpu_memory_gb": 13,
            "scientific_interpretation_allowed": False,
        },
    }
    rows = []
    specs = {spec["id"]: spec for spec in config["models"]}
    for model in ("m0", "m1"):
        for order, score in (("012", 0), ("210", 1)):
            rows.append(
                {
                    "id": f"item-{order}",
                    "source_id": "source",
                    "model": model,
                    "model_repo": specs[model]["name"],
                    "model_revision_requested": specs[model]["revision"],
                    "model_revision_resolved": specs[model]["revision"],
                    "generation_seed": "0",
                    "wording_id": "original",
                    "option_order": order,
                    "parse_status": "strict",
                    "malformed": "false",
                    "verbosity_confounded": "false",
                    "instrumental_score_0_2": str(score),
                }
            )
    model_metadata = []
    for spec in config["models"]:
        model_metadata.append(
            {
                "model": spec["id"],
                "model_repo": spec["name"],
                "model_revision_requested": spec["revision"],
                "model_revision_resolved": spec["revision"],
                "rows": 2,
                "model_is_quantized": True,
                "attention_mask_mode": "explicit_all_ones",
                "quantization_config_resolved": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "float16",
                },
                "peak_gpu_memory_bytes": 7 * 1024**3,
            }
        )
    summary = {
        "scientific_interpretation_allowed": False,
        "had_error": False,
        "errors": [],
        "rows": 4,
        "expected_rows": 4,
        "completed_utc": "2026-01-01T00:08:00+00:00",
        "models": model_metadata,
    }
    run_metadata = {
        "started_utc": "2026-01-01T00:00:00+00:00",
        "package_versions": {"transformers": "4.57.6", "accelerate": "1.13.0"},
        "gpus": ["Tesla T4", "Tesla T4"],
    }
    return rows, summary, run_metadata, config


class PilotTests(unittest.TestCase):
    def test_passes_strict_feasibility_and_withholds_interpretation(self) -> None:
        metrics = analyze_endpoint_pilot(*fixture())
        self.assertTrue(metrics["feasibility_passed"])
        self.assertFalse(metrics["scientific_interpretation_allowed"])
        self.assertEqual(metrics["wall_clock_seconds"], 480)
        self.assertEqual([model["rows"] for model in metrics["models"]], [2, 2])
        report = render_endpoint_pilot_markdown(metrics)
        self.assertIn("Scientific interpretation: not allowed", report)
        self.assertIn("not reported as evidence", report)

    def test_rejects_missing_attention_mask_provenance(self) -> None:
        rows, summary, run_metadata, config = fixture()
        summary["models"][0].pop("attention_mask_mode")
        with self.assertRaisesRegex(PilotValidationError, "attention-mask mode mismatch"):
            analyze_endpoint_pilot(rows, summary, run_metadata, config)

    def test_rejects_quality_or_memory_failure(self) -> None:
        rows, summary, run_metadata, config = fixture()
        bad_rows = copy.deepcopy(rows)
        bad_rows[0]["malformed"] = "true"
        with self.assertRaisesRegex(PilotValidationError, "malformed row"):
            analyze_endpoint_pilot(bad_rows, summary, run_metadata, config)

        bad_summary = copy.deepcopy(summary)
        bad_summary["models"][1]["peak_gpu_memory_bytes"] = 14 * 1024**3
        with self.assertRaisesRegex(PilotValidationError, "GPU memory gate failed"):
            analyze_endpoint_pilot(rows, bad_summary, run_metadata, config)

    def test_rejects_duplicate_cells(self) -> None:
        rows, summary, run_metadata, config = fixture()
        rows[1] = dict(rows[0])
        with self.assertRaisesRegex(PilotValidationError, "duplicate model/id cells"):
            analyze_endpoint_pilot(rows, summary, run_metadata, config)


if __name__ == "__main__":
    unittest.main()
