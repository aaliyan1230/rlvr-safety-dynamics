from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from copy import deepcopy
from datetime import date
from pathlib import Path

from rlvr_safety.io import write_jsonl
from rlvr_safety.permission.experiment import (
    SPEC_VERSION,
    build_workload,
    check_envelope,
    fetch_chat_template_sha,
    fetch_model_metadata,
    validate_spec,
    write_bundle,
)
from rlvr_safety.provenance import verify_manifest

from .helpers import control, scenario

REVISION = "a" * 40
TEMPLATE = "native template"
TEMPLATE_SHA = hashlib.sha256(TEMPLATE.encode()).hexdigest()


def spec() -> dict:
    return {
        "spec_version": SPEC_VERSION,
        "experiment_id": "permission-test-2026-10-08",
        "models": [{"repo": "example/model", "revision": REVISION, "label": "step0"}],
        "workloads": [
            {
                "name": "controls",
                "bank": "controls.jsonl",
                "plan": {
                    "executable": {"orders": "all", "id_arms": ["semantic", "neutral"]},
                    "wordings": ["original"],
                },
            },
            {
                "name": "resource",
                "bank": "scenarios.jsonl",
                "sources": ["fixture_transfer"],
                "min_wordings": 2,
                "plan": {
                    "mcq": {"orders": "all"},
                    "executable": {"orders": "all"},
                    "option_free": {"count": 1},
                    "wordings": ["original"],
                },
            },
        ],
        "readouts": ["generate", "choice_logprobs"],
        "generation": {
            "max_new_tokens": 2048,
            "do_sample": False,
            "seed": 0,
            "per_response_seconds": 120,
        },
        "runtime_pins": {"torch": "2.8.0", "transformers": "4.57.1"},
        "launch": {
            "gpu": "NVIDIA A100-SXM4-80GB",
            "gpu_count": 1,
            "datacenters": ["US-KS-2"],
            "hourly_ceiling": 2.0,
            "max_minutes": 60,
            "max_launch_usd": 5.0,
            "image": "runpod/pytorch:test",
            "cloud": "SECURE",
            "min_cuda_version": "12.8",
            "container_disk_gb": 50,
        },
        "envelope": "envelope.json",
        "gates": "gates.json",
    }


def envelope() -> dict:
    return {
        "status": "ok",
        "valid_through": "2026-10-15",
        "limits": {
            "gpu_count": 1,
            "hourly_ceiling_usd": 2.0,
            "max_minutes_per_launch": 60,
            "max_usd_per_launch": 5.0,
            "max_usd_total": 20.0,
            "storage_limit_gb": 50,
            "image": "runpod/pytorch:test",
            "cloud": "SECURE",
        },
    }


def metadata_fetcher(repo, revision):
    return {
        "id": repo,
        "sha": revision,
        "siblings": [{"rfilename": "model.safetensors", "lfs": {"sha256": "0" * 64}}],
    }


def template_fetcher(repo, revision):
    return TEMPLATE_SHA


class Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_jsonl(self.root / "scenarios.jsonl", [scenario()])
        write_jsonl(self.root / "controls.jsonl", [control()])
        (self.root / "gates.json").write_text('{"gates": []}\n')
        (self.root / "envelope.json").write_text(json.dumps(envelope()))
        (self.root / "spec.json").write_text(json.dumps(spec()))

    def test_valid_spec_and_each_defect(self):
        validate_spec(spec())
        defects = {
            "version": lambda s: s.update(spec_version="old"),
            "sampling": lambda s: s["generation"].update(do_sample=True),
            "revision": lambda s: s["models"][0].update(revision="main"),
            "duplicate label": lambda s: s["models"].append(deepcopy(s["models"][0])),
            "plan key": lambda s: s["workloads"][0]["plan"].update(oops={}),
            "no format": lambda s: s["workloads"][0].update(plan={"wordings": ["original"]}),
            "readouts": lambda s: s.update(readouts=["choice_logprobs"]),
            "two gpus": lambda s: s["launch"].update(gpu_count=2),
            "no gates": lambda s: s.pop("gates"),
            "duplicate workload": lambda s: s["workloads"].append(deepcopy(s["workloads"][0])),
            "bad template hash": lambda s: s["models"][0].update(chat_template_sha256="xyz"),
        }
        for name, fn in defects.items():
            candidate = spec()
            fn(candidate)
            with self.subTest(name), self.assertRaises(ValueError):
                validate_spec(candidate)

    def test_workload_counts_and_tags(self):
        scenarios, requests, report = build_workload(spec(), self.root)
        self.assertEqual(
            {s["source_id"] for s in scenarios}, {"fixture_transfer", "fixture_control"}
        )
        # controls: 6 orders x 2 id arms; resource: 2 perms x (6 mcq + 6 executable + 1 free text)
        self.assertEqual(report["workloads"]["controls"]["requests"], 12)
        self.assertEqual(report["workloads"]["resource"]["requests"], 26)
        self.assertEqual(report["requests"], 38)
        self.assertEqual({r["workload"] for r in requests}, {"controls", "resource"})
        self.assertEqual(len({r["episode_id"] for r in requests}), 38)
        self.assertEqual(report["review_statuses"], {"pending": 2})

    def test_missing_wording_or_unknown_source_is_an_error(self):
        broken = spec()
        broken["workloads"][1]["plan"]["wordings"] = ["nope"]
        with self.assertRaisesRegex(ValueError, "lacks wording"):
            build_workload(broken, self.root)
        broken = spec()
        broken["workloads"][1]["sources"] = ["ghost"]
        with self.assertRaisesRegex(ValueError, "unknown sources"):
            build_workload(broken, self.root)

    def test_confirmation_style_runs_require_accepted_review(self):
        strict = spec()
        strict["require_review"] = "accepted"
        with self.assertRaisesRegex(ValueError, "without accepted human review"):
            build_workload(strict, self.root)

    def test_banks_that_fail_mechanical_checks_cannot_be_bundled(self):
        bad = scenario()
        bad["actions"][1]["effects"] = []
        write_jsonl(self.root / "scenarios.jsonl", [bad])
        with self.assertRaisesRegex(ValueError, "mechanical checks failed"):
            build_workload(spec(), self.root)

    def test_envelope_checks(self):
        launch = spec()["launch"]
        self.assertEqual(check_envelope(launch, envelope(), 0.0, date(2026, 10, 9))["problems"], [])
        cases = {
            "expired": (launch, envelope(), 0.0, date(2026, 10, 16)),
            "hourly_ceiling": (
                {**launch, "hourly_ceiling": 3.0},
                envelope(),
                0.0,
                date(2026, 10, 9),
            ),
            "max_minutes": ({**launch, "max_minutes": 90}, envelope(), 0.0, date(2026, 10, 9)),
            "max_launch_usd": (
                {**launch, "max_launch_usd": 6.0},
                envelope(),
                0.0,
                date(2026, 10, 9),
            ),
            "image": ({**launch, "image": "other"}, envelope(), 0.0, date(2026, 10, 9)),
            "container_disk_gb": (
                {**launch, "container_disk_gb": 80},
                envelope(),
                0.0,
                date(2026, 10, 9),
            ),
            "total_budget": (launch, envelope(), 16.0, date(2026, 10, 9)),
            "cloud": ({**launch, "cloud": "COMMUNITY"}, envelope(), 0.0, date(2026, 10, 9)),
        }
        for name, args in cases.items():
            with self.subTest(name):
                self.assertTrue(check_envelope(*args)["problems"], name)
        pending = envelope()
        pending["status"] = "limits-from-plan; wording-not-yet-recorded"
        self.assertTrue(check_envelope(launch, pending, 0.0, date(2026, 10, 9))["warnings"])

    def test_bundle_is_complete_hashed_and_deterministic(self):
        first = write_bundle(
            self.root / "spec.json",
            self.root / "b1",
            root=self.root,
            metadata_fetcher=metadata_fetcher,
            template_fetcher=template_fetcher,
        )
        second = write_bundle(
            self.root / "spec.json",
            self.root / "b2",
            root=self.root,
            metadata_fetcher=metadata_fetcher,
            template_fetcher=template_fetcher,
        )
        self.assertEqual(first["sha256"], second["sha256"])
        verify_manifest(self.root / "b1/bundle_manifest.json")
        resolved = json.loads((self.root / "b1/experiment.json").read_text())
        self.assertEqual(resolved["models"][0]["chat_template_sha256"], TEMPLATE_SHA)
        meta = json.loads((self.root / "b1/model_metadata/step0.json").read_text())
        self.assertEqual(meta["sha"], REVISION)
        manifest = json.loads((self.root / "b1/bundle_manifest.json").read_text())
        for name in (
            "experiment.json",
            "gates.json",
            "scenarios.jsonl",
            "requests.jsonl",
            "runtime-pins.txt",
            "permission_workload.py",
            "src/rlvr_safety/permission/generation.py",
            "src/rlvr_safety/permission/scoring.py",
            "model_metadata/step0.json",
        ):
            self.assertIn(name, manifest["files"])
        self.assertFalse(any("v1" in n or ".env" in n for n in manifest["files"]))
        self.assertEqual(manifest["requests"], 38)
        archive = (self.root / "b1.tar.gz").read_bytes()
        self.assertEqual(hashlib.sha256(archive).hexdigest(), first["sha256"])
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive))) as tar:
            self.assertEqual({m.mtime for m in tar.getmembers()}, {0})
            self.assertEqual(
                {m.name for m in tar.getmembers()}, {*manifest["files"], "bundle_manifest.json"}
            )
        self.assertEqual((self.root / "b1.tar.gz.sha256").read_text().strip(), first["sha256"])

    def test_changing_the_bank_changes_the_bundle(self):
        first = write_bundle(
            self.root / "spec.json",
            self.root / "b1",
            root=self.root,
            metadata_fetcher=metadata_fetcher,
            template_fetcher=template_fetcher,
        )
        changed = scenario()
        changed["wordings"]["original"]["task"] += " One more detail."
        write_jsonl(self.root / "scenarios.jsonl", [changed])
        second = write_bundle(
            self.root / "spec.json",
            self.root / "b2",
            root=self.root,
            metadata_fetcher=metadata_fetcher,
            template_fetcher=template_fetcher,
        )
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_pinned_template_hash_must_match_the_repository(self):
        pinned = spec()
        pinned["models"][0]["chat_template_sha256"] = "1" * 64
        (self.root / "spec.json").write_text(json.dumps(pinned))
        with self.assertRaisesRegex(ValueError, "pinned chat template hash differs"):
            write_bundle(
                self.root / "spec.json",
                self.root / "b1",
                root=self.root,
                metadata_fetcher=metadata_fetcher,
                template_fetcher=template_fetcher,
            )

    def test_bundle_directory_is_never_overwritten(self):
        kwargs = dict(metadata_fetcher=metadata_fetcher, template_fetcher=template_fetcher)
        write_bundle(self.root / "spec.json", self.root / "b1", root=self.root, **kwargs)
        with self.assertRaises(FileExistsError):
            write_bundle(self.root / "spec.json", self.root / "b1", root=self.root, **kwargs)

    def test_fetchers_parse_hub_responses_and_reject_mismatches(self):
        api = {
            "id": "example/model",
            "sha": REVISION,
            "siblings": [
                {"rfilename": "a.safetensors", "size": 3, "lfs": {"sha256": "f" * 64, "size": 3}},
                {"rfilename": "config.json", "size": 1},
            ],
            "downloads": 99,
        }
        got = fetch_model_metadata(
            "example/model", REVISION, lambda *a, **k: Response(json.dumps(api).encode())
        )
        self.assertEqual(got["siblings"][0]["lfs"], {"sha256": "f" * 64})
        self.assertNotIn("downloads", got)
        with self.assertRaisesRegex(ValueError, "different repository or revision"):
            fetch_model_metadata(
                "example/model", "b" * 40, lambda *a, **k: Response(json.dumps(api).encode())
            )

        def tokenizer_config(request, **kwargs):
            return Response(json.dumps({"chat_template": TEMPLATE}).encode())

        self.assertEqual(
            fetch_chat_template_sha("example/model", REVISION, tokenizer_config), TEMPLATE_SHA
        )

        def jinja_only(request, **kwargs):
            if request.full_url.endswith("tokenizer_config.json"):
                return Response(b"{}")
            return Response(TEMPLATE.encode())

        self.assertEqual(
            fetch_chat_template_sha("example/model", REVISION, jinja_only), TEMPLATE_SHA
        )


if __name__ == "__main__":
    unittest.main()
