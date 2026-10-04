from __future__ import annotations

import unittest

from scripts.analyze_camera_ready_diagnostics import source_order_contrasts

OPTION_ORDERS = ["012", "021", "102", "120", "201", "210"]


class CameraReadyDiagnosticsTests(unittest.TestCase):
    def test_order_contrasts_average_within_source_before_bootstrapping(self) -> None:
        rows = []
        for step in (0, 2240):
            for source, later_score in (("a", 2), ("b", 0)):
                for order_index, order in enumerate(OPTION_ORDERS):
                    score = 0 if step == 0 or order_index else later_score
                    wordings = ("original", "p1", "p2", "p3") if source == "a" else ("original",)
                    for wording in wordings:
                        rows.append(
                            {
                                "model": f"tulu_step_{step:04d}_grpo",
                                "source_id": source,
                                "option_order": order,
                                "wording_id": wording,
                                "instrumental_score_0_2": str(score),
                                "malformed": "false",
                                "verbosity_confounded": "false",
                            }
                        )
        estimates, intervals = source_order_contrasts(rows, 0, 2240)
        self.assertAlmostEqual(estimates["012"], 1.0)
        self.assertNotAlmostEqual(estimates["012"], 1.6)
        self.assertLessEqual(intervals["012"][0], estimates["012"])
        self.assertGreaterEqual(intervals["012"][1], estimates["012"])


if __name__ == "__main__":
    unittest.main()
