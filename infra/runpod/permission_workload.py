"""Run a prepared experiment bundle on one CUDA GPU: every model, every request, resumable.

The controller uploads the bundle, installs the pinned packages and calls this script. It only
generates, scores and records; frozen gates are evaluated locally after retrieval.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from pathlib import Path

from rlvr_safety.io import read_jsonl
from rlvr_safety.permission.benchmark import run_benchmark
from rlvr_safety.permission.generation import HFPermissionProvider, save_runtime
from rlvr_safety.provenance import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    verify_manifest(root / "bundle_manifest.json")
    spec = json.loads((root / "experiment.json").read_text())
    scenarios = list(read_jsonl(root / "scenarios.jsonl"))
    requests = list(read_jsonl(root / "requests.jsonl"))
    output = root / "out"
    output.mkdir(exist_ok=True)
    phase = {"stage": "starting", "model": None, "completed": 0, "total": len(requests)}
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            record = {"time": time.time(), **phase}
            with (output / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            done.wait(10)

    guard = threading.Thread(target=heartbeat, daemon=True)
    guard.start()
    summary = {"experiment_id": spec["experiment_id"], "models": {}, "completed": False}
    try:
        for model in spec["models"]:
            label = model["label"]
            phase.update(stage="model_load", model=label, completed=0)
            config = {
                "model": model,
                "generation": spec["generation"],
                "runtime_pins": spec["runtime_pins"],
                "launch": spec["launch"],
                "readouts": spec["readouts"],
            }
            metadata = json.loads((root / "model_metadata" / f"{label}.json").read_text())
            provider = HFPermissionProvider(config, metadata, root.parent / "hf_cache")
            model_out = output / label
            model_out.mkdir(exist_ok=True)
            save_runtime(model_out / "runtime.json", provider)

            class ProgressProvider:
                provenance = provider.provenance

                def respond(self, request, provider=provider):
                    phase.update(stage="inference", episode_id=request["episode_id"])
                    response = provider.respond(request)
                    phase["completed"] += 1
                    return response

            run_benchmark(scenarios, requests, ProgressProvider(), model_out / "benchmark")
            rows = list(read_jsonl(model_out / "benchmark/results.jsonl"))
            summary["models"][label] = {
                "episodes": len(rows),
                "requested": len(requests),
                "complete": len(rows) == len(requests),
            }
            provider.close()
            del provider
            shutil.rmtree(root.parent / "hf_cache", ignore_errors=True)
        summary["completed"] = all(m["complete"] for m in summary["models"].values())
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        phase["stage"] = "complete" if summary["completed"] else "incomplete"
        print(json.dumps(summary), flush=True)
        if not summary["completed"]:
            raise RuntimeError("not every model completed every request")
    finally:
        done.set()
        guard.join()


if __name__ == "__main__":
    main()
