from __future__ import annotations

import unittest

from rlvr_safety.cross_format_analysis import MILESTONES, STEPS, analyze_capability, analyze_freeform


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


if __name__ == "__main__":
    unittest.main()
