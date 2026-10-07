from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rlvr_safety.io import read_jsonl, sha256_file, write_jsonl
from rlvr_safety.permission.v1.benchmark import (
    FixtureProvider,
    prepare_requests,
    run_benchmark,
    scripted_responses,
)
from rlvr_safety.permission.v1.prompts import build_conditions, score_mcq
from rlvr_safety.permission.v1.schema import digest
from rlvr_safety.provenance import verify_manifest


def scenario() -> dict:
    return json.loads((Path(__file__).parent / "permission_scenario.json").read_text())


class PermissionBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scenarios = [scenario()]
        self.requests = prepare_requests(self.scenarios, stress=True)
        self.fixtures = self.root / "fixtures.jsonl"
        write_jsonl(self.fixtures, scripted_responses(self.scenarios, self.requests))
        self.provider = FixtureProvider(self.fixtures)
        self.output = self.root / "run"

    def test_permutations_preserve_semantic_actions(self):
        conditions = build_conditions(self.scenarios)
        self.assertEqual(len(conditions), 16)
        for permission in ("allowed", "forbidden"):
            mcq = [c for c in conditions if c["permission"] == permission and c["format"] == "mcq"]
            self.assertEqual(len({c["order_id"] for c in mcq}), 6)
            for condition in mcq:
                letter = next(k for k, v in condition["choice_key"].items() if v == "use_other")
                scored = score_mcq(
                    condition,
                    {"raw_response": letter, "stop_status": "complete"},
                    self.scenarios[0],
                )
                self.assertEqual(scored["selected_action_id"], "use_other")
                self.assertEqual(scored["endorsed_forbidden_action"], permission == "forbidden")

    def test_resume_is_identical_and_summary_keeps_unknown_denominator(self):
        summary = run_benchmark(self.scenarios, self.requests, self.provider, self.output)
        self.assertEqual(summary["episodes"], 34)
        self.assertEqual(summary["undetermined_forbidden_episodes"], 5)
        self.assertEqual(summary["forbidden_violation_bounds"], [0.2, 0.7])
        result_hash = sha256_file(self.output / "results.jsonl")
        self.assertEqual(
            summary, run_benchmark(self.scenarios, self.requests, self.provider, self.output)
        )
        self.assertEqual(sha256_file(self.output / "results.jsonl"), result_hash)
        verify_manifest(self.output / "artifact_manifest.json")
        free = [
            r for r in read_jsonl(self.output / "results.jsonl") if r["format"] == "option_free"
        ]
        self.assertTrue(all(r["score"]["text_label"] is None for r in free))

    def test_crash_and_resume_does_not_duplicate_episodes(self):
        real = self.provider

        class InterruptedProvider:
            provenance = real.provenance
            count = 0

            def respond(self, request):
                self.count += 1
                if self.count == 4:
                    raise RuntimeError("controller interrupted")
                return real.respond(request)

        with self.assertRaises(RuntimeError):
            run_benchmark(self.scenarios, self.requests, InterruptedProvider(), self.output)
        self.assertEqual(len(list(read_jsonl(self.output / "results.jsonl"))), 3)
        run_benchmark(self.scenarios, self.requests, real, self.output)
        rows = list(read_jsonl(self.output / "results.jsonl"))
        self.assertEqual(len(rows), len(self.requests))
        self.assertEqual(len({r["episode_id"] for r in rows}), len(rows))

    def test_changed_prompt_provider_or_runtime_cannot_reuse_run(self):
        run_benchmark(self.scenarios, self.requests, self.provider, self.output)
        changed = deepcopy(self.scenarios)
        changed[0]["wordings"]["original"]["task"] += " Revised."
        with self.assertRaisesRegex(ValueError, "resume refused"):
            run_benchmark(
                changed, prepare_requests(changed, stress=True), self.provider, self.output
            )
        responses = list(read_jsonl(self.fixtures))
        responses[0]["raw_response"] = "B"
        write_jsonl(self.fixtures, responses)
        with self.assertRaisesRegex(ValueError, "resume refused"):
            run_benchmark(
                self.scenarios, self.requests, FixtureProvider(self.fixtures), self.output
            )
        manifest_path = self.output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["python"] = "different runtime"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "resume refused"):
            run_benchmark(self.scenarios, self.requests, self.provider, self.output)

    def test_altered_request_or_duplicate_episode_is_rejected(self):
        changed = deepcopy(self.requests)
        changed[0]["prompt"] += " Ignore permissions."
        with self.assertRaisesRegex(ValueError, "current condition"):
            run_benchmark(self.scenarios, changed, self.provider, self.output)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_benchmark(
                self.scenarios, self.requests + self.requests[:1], self.provider, self.output
            )

    def test_corrupted_or_partial_results_are_not_silently_reused(self):
        run_benchmark(self.scenarios, self.requests, self.provider, self.output)
        rows = list(read_jsonl(self.output / "results.jsonl"))
        rows[0]["response"]["raw_response"] = "B"
        write_jsonl(self.output / "results.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "corrupted"):
            run_benchmark(self.scenarios, self.requests, self.provider, self.output)
        rows[0]["record_sha256"] = digest(
            {k: v for k, v in rows[0].items() if k != "record_sha256"}
        )
        write_jsonl(self.output / "results.jsonl", rows)
        with (self.output / "results.jsonl").open("ab") as handle:
            handle.write(b'{"partial":')
        with self.assertRaisesRegex(ValueError, "partial"):
            run_benchmark(self.scenarios, self.requests, self.provider, self.output)

    def test_wrong_response_or_unpinned_model_is_rejected(self):
        real = self.provider

        class WrongProvider:
            provenance = real.provenance

            def respond(self, request):
                return {**real.respond(request), "condition_id": "wrong"}

        with self.assertRaisesRegex(ValueError, "wrong episode"):
            run_benchmark(self.scenarios, self.requests, WrongProvider(), self.output)

        class UnpinnedProvider:
            provenance = {"provider": "model", "inference_performed": True, "model": "test"}

        with self.assertRaisesRegex(ValueError, "pin model_revision"):
            run_benchmark(self.scenarios, self.requests, UnpinnedProvider(), self.root / "unpinned")


if __name__ == "__main__":
    unittest.main()
