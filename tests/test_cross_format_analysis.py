from __future__ import annotations

import unittest

from rlvr_safety.cross_format_analysis import (
    MILESTONES,
    STEPS,
    analyze_capability,
    analyze_freeform,
    apply_claim_gate,
)


class CrossFormatAnalysisTests(unittest.TestCase):
    def test_complete_capability_and_freeform_panels(self) -> None:
        capability = []
        for step in STEPS:
            for index in range(30):
                capability.append(
                    {
                        "checkpoint": step,
                        "id": f"item-{index}",
                        "family": ("gsm", "math", "instruction_following")[index // 10],
                        "correct": step >= 1920 or index % 2 == 0,
                        "parse_status": "strict",
                    }
                )
        cap = analyze_capability(capability, repetitions=20)
        self.assertEqual(len(cap["checkpoints"]), 12)
        self.assertTrue(cap["h7_capability_threshold_met"])

        freeform = []
        for step in MILESTONES:
            for index in range(24):
                freeform.append(
                    {
                        "checkpoint": step,
                        "id": f"free-{index}",
                        "source_id": f"source-{index}",
                        "score_0_2": 1 if step else 0,
                        "judgeable": True,
                    }
                )
        safety = analyze_freeform(freeform, repetitions=20)
        self.assertEqual(len(safety["baseline_contrasts"]), 5)
        self.assertEqual(safety["baseline_contrasts"][0]["delta_vs_step_0"], 1)

    def test_measurement_drift_does_not_depend_on_freeform_agreement(self) -> None:
        structured = {
            "h6": {"supported": False, "persistent_pairs": []},
            "adjacent_contrasts": [
                {
                    "first_step": 0,
                    "second_step": 40,
                    "behavior": {"equivalent_at_95pct_interval": True},
                    "measurement": {
                        "order_range": {
                            "estimate": 0.12,
                            "simultaneous_ci_low": 0.01,
                            "simultaneous_ci_high": 0.20,
                        }
                    },
                }
            ],
            "baseline_contrasts": [
                {"step": 1920, "equivalent_at_95pct_interval": True}
            ],
        }
        capability = {"h7_capability_threshold_met": False}
        freeform = {"baseline_contrasts": []}
        result = apply_claim_gate(structured, capability, freeform, {"passed": False})
        self.assertEqual(result["label"], "measurement_drift")
        self.assertFalse(result["freeform_agreement_gate_passed"])


if __name__ == "__main__":
    unittest.main()
