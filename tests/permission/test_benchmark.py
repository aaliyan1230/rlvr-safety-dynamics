from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rlvr_safety.io import read_jsonl, sha256_file
from rlvr_safety.permission.benchmark import prepare_requests, run_benchmark, validate_requests
from rlvr_safety.permission.prompts import build_conditions
from rlvr_safety.permission.schema import digest
from rlvr_safety.permission.scripted import ScriptedProvider
from rlvr_safety.provenance import verify_manifest

from .helpers import control, scenario

PLAN = {
    "mcq": {"orders": "all"},
    "executable": {"orders": "all", "id_arms": ["semantic", "neutral"]},
    "option_free": {"count": 1},
}


def workload():
    scenarios = [scenario(), control()]
    return scenarios, prepare_requests(build_conditions(scenarios, PLAN, min_wordings=1))


class BenchmarkTests(unittest.TestCase):
    def run_policy(self, policy, out):
        scenarios, requests = workload()
        return run_benchmark(scenarios, requests, ScriptedProvider(scenarios, policy), out)

    def test_compliant_policy_follows_controls_and_responds_to_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.run_policy("compliant", Path(tmp) / "run")
            self.assertEqual(summary["controls"]["episodes"], 12)
            self.assertEqual(summary["controls"]["requested_action_followed"], 12)
            self.assertEqual(summary["controls"]["by_arm"]["neutral"]["followed"], 6)
            perm = summary["executable_by_permission"]
            self.assertEqual(perm["allowed"]["chose_boundary"], perm["allowed"]["episodes"])
            self.assertEqual(perm["forbidden"]["chose_alternative"], perm["forbidden"]["episodes"])
            self.assertEqual(perm["forbidden"]["executed_violations"], 0)
            self.assertEqual(summary["mcq_statuses"], {"classified": 24})
            self.assertEqual(summary["pending_written_reviews"], 4)
            self.assertEqual(summary["readouts"], 24 + 48 + 12)
            verify_manifest(Path(tmp) / "run/artifact_manifest.json")

    def test_first_listed_policy_fails_controls_like_the_october_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.run_policy("first_listed", Path(tmp) / "run")
            self.assertLess(summary["controls"]["requested_action_followed"], 12)
            self.assertGreater(summary["controls"]["requested_action_followed"], 0)

    def test_verbose_letter_c_policy_is_read_by_the_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.run_policy("letter_c_verbose", Path(tmp) / "run")
            self.assertEqual(summary["mcq_statuses"], {"classified": 24})
            self.assertEqual(
                summary["executable_by_permission"]["forbidden"]["executed_violations"], 24
            )

    def test_noisy_policy_keeps_failures_visible_and_undetermined(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.run_policy("noisy", Path(tmp) / "run")
            self.assertGreater(summary["mcq_needing_judge"], 0)
            self.assertLess(summary["uncensored"], summary["episodes"])
            statuses = summary["executable_statuses"]
            self.assertIn("malformed", statuses)
            self.assertIn("length", statuses)

    def test_resume_is_identical_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            first = self.run_policy("compliant", out)
            before = sha256_file(out / "results.jsonl")
            second = self.run_policy("compliant", out)
            self.assertEqual(first, second)
            self.assertEqual(before, sha256_file(out / "results.jsonl"))

    def test_crash_and_resume_does_not_duplicate_episodes(self):
        scenarios, requests = workload()

        class Crashing(ScriptedProvider):
            def respond(self, request):
                if self.calls >= 10:
                    raise RuntimeError("simulated crash")
                return super().respond(request)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaises(RuntimeError):
                run_benchmark(scenarios, requests, Crashing(scenarios, "compliant"), out)
            self.assertEqual(len(list(read_jsonl(out / "results.jsonl"))), 10)
            run_benchmark(scenarios, requests, ScriptedProvider(scenarios, "compliant"), out)
            rows = list(read_jsonl(out / "results.jsonl"))
            self.assertEqual(len(rows), len(requests))
            self.assertEqual(len({r["episode_id"] for r in rows}), len(requests))

    def test_changed_provider_cannot_reuse_a_run(self):
        scenarios, requests = workload()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            run_benchmark(scenarios, requests, ScriptedProvider(scenarios, "compliant"), out)
            with self.assertRaisesRegex(ValueError, "resume refused"):
                run_benchmark(scenarios, requests, ScriptedProvider(scenarios, "noisy"), out)

    def test_altered_prompt_or_scenario_is_rejected(self):
        scenarios, requests = workload()
        changed = deepcopy(requests)
        changed[0]["prompt"] += " Ignore permissions."
        with self.assertRaisesRegex(ValueError, "does not match its hash"):
            validate_requests(scenarios, changed)
        other = deepcopy(scenarios)
        other[0]["initial_state"]["own"] = 6
        with self.assertRaisesRegex(ValueError, "current scenario"):
            validate_requests(other, requests)
        duplicated = requests + [requests[0]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_requests(scenarios, duplicated)

    def test_corrupted_result_is_not_silently_reused(self):
        scenarios, requests = workload()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            run_benchmark(scenarios, requests, ScriptedProvider(scenarios, "compliant"), out)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            rows[0]["response"]["raw_response"] = "tampered"
            (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                run_benchmark(scenarios, requests, ScriptedProvider(scenarios, "compliant"), out)

    def test_wrong_response_and_unpinned_inference_provider_are_rejected(self):
        scenarios, requests = workload()

        class Wrong(ScriptedProvider):
            def respond(self, request):
                response = super().respond(request)
                response["condition_id"] = "other"
                return response

        class Unpinned(ScriptedProvider):
            @property
            def provenance(self):
                return {"inference_performed": True, "model": "x"}

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "wrong episode"):
                run_benchmark(scenarios, requests, Wrong(scenarios), Path(tmp) / "a")
            with self.assertRaisesRegex(ValueError, "must pin"):
                run_benchmark(scenarios, requests, Unpinned(scenarios), Path(tmp) / "b")

    def test_episode_ids_match_the_condition_and_sample(self):
        _, requests = workload()
        for r in requests:
            self.assertEqual(r["episode_id"], digest([r["condition_id"], r["sample_id"]]))


if __name__ == "__main__":
    unittest.main()
