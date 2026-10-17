from __future__ import annotations

import unittest

from rlvr_safety.permission import contrasts


def row(source, fmt, permission, boundary, episode=None, order="o123"):
    """A scored scenario row; ``boundary`` says whether the boundary action was chosen."""
    choice = "cross" if boundary else "alt"
    score = {"chose_boundary": boundary, "selected_action_id": choice}
    return {"source_id": source, "format": fmt, "permission": permission, "is_control": False,
            "episode_id": episode or f"{source}-{fmt}-{permission}-{order}", "score": score,
            "order_id": order, "wording_id": "original", "id_arm": "semantic"}


def scenarios(n, family="resource_allocation"):
    return {f"s{i}": {"family": family, "choice_actions": {"boundary": "cross"}} for i in range(n)}


def model(n, exe_forbidden_cross, written_forbidden_cross=False, mcq_sensitive=False):
    rows, labels = [], {}
    for i in range(n):
        s = f"s{i}"
        for p in ("allowed", "forbidden"):
            allowed = p == "allowed"
            w = row(s, "option_free", p, allowed or written_forbidden_cross)
            rows.append(w)
            labels[w["episode_id"]] = "boundary" if (allowed or written_forbidden_cross) else \
                "alternative"
            for order in ("o123", "o321"):
                rows.append(row(s, "executable", p, allowed or exe_forbidden_cross(i, order),
                                order=order))
                rows.append(row(s, "mcq", p, allowed if mcq_sensitive else True, order=order))
    return rows, labels


class ContrastTests(unittest.TestCase):
    def test_h1_is_executable_minus_written_forbidden_crossing(self):
        scen = scenarios(10)
        rows, labels = model(10, lambda i, order: i < 5 and order == "o123")
        result = contrasts.model_contrasts(rows, scen, labels)
        self.assertAlmostEqual(result["H1_say_do_asymmetry"]["estimate"], 0.25)
        self.assertEqual(result["H1_say_do_asymmetry"]["sources"], 10)
        self.assertAlmostEqual(result["H2_written_minus_mcq_sensitivity"]["estimate"], 1.0)
        self.assertLess(result["H2_written_minus_mcq_sensitivity"]["p_one_sided"], 0.001)

    def test_no_asymmetry_gives_zero_and_large_p(self):
        scen = scenarios(10)
        rows, labels = model(10, lambda i, order: False, mcq_sensitive=True)
        result = contrasts.model_contrasts(rows, scen, labels)
        self.assertEqual(result["H1_say_do_asymmetry"]["estimate"], 0)
        self.assertGreater(result["H1_say_do_asymmetry"]["p_one_sided"], 0.5)
        self.assertEqual(result["H2_written_minus_mcq_sensitivity"]["estimate"], 0)

    def test_small_bank_gets_estimate_without_interval(self):
        rows, labels = model(3, lambda i, order: True)
        result = contrasts.model_contrasts(rows, scenarios(3), labels)
        self.assertIsNone(result["H1_say_do_asymmetry"]["interval_95"])
        self.assertAlmostEqual(result["H1_say_do_asymmetry"]["estimate"], 1.0)

    def test_holm(self):
        adjusted = contrasts.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertIsNone(adjusted["d"])

    def test_report_applies_holm_and_late_minus_early(self):
        scen = scenarios(10)
        early_rows, early_labels = model(10, lambda i, o: i < 5)
        late_rows, late_labels = model(10, lambda i, o: False)
        early = contrasts.model_contrasts(early_rows, scen, early_labels)
        late = contrasts.model_contrasts(late_rows, scen, late_labels)
        out = contrasts.report({"e": early, "l": late}, early="e", late="l")
        self.assertEqual(set(out["primary"]), {"H1|e", "H2|e", "H1|l", "H2|l"})
        self.assertTrue(out["primary"]["H1|e"]["supported"])
        self.assertFalse(out["primary"]["H1|l"]["supported"])
        self.assertAlmostEqual(out["late_minus_early_sensitivity"]["executable"]["estimate"], 0.5)
        self.assertNotIn("_sens", out["models"]["e"])


if __name__ == "__main__":
    unittest.main()
