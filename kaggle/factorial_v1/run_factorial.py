#!/usr/bin/env python3
"""Kaggle T4x2 runner for the counterbalanced wording × option-order study."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
OUT_DIR = WORKING / "wording_position_v2_legacy_quant"
PINNED_DEPENDENCIES = [
    "transformers==4.57.6",
    "accelerate==1.13.0",
    "bitsandbytes==0.49.2",
    "sentencepiece==0.2.1",
    "protobuf==5.29.5",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def ensure_dependencies() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *PINNED_DEPENDENCIES]
    )


def find_one(pattern: str) -> Path:
    matches = sorted(INPUT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {INPUT}, found {matches}")
    return matches[0]


def configure_package_path() -> Path:
    package_init = find_one("**/src/rlvr_safety/__init__.py")
    source_root = package_init.parents[1]
    sys.path.insert(0, str(source_root))
    return source_root


def gpu_names() -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def package_versions() -> dict[str, str]:
    names = ["torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "protobuf"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def run_worker(model_index: int, gpu_id: int, prompts_path: Path, config_path: Path) -> None:
    configure_package_path()
    from rlvr_safety.hf_generation import generate_model_rows
    from rlvr_safety.io import read_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = config["models"][model_index]
    model_name = model_spec["name"]
    revision = model_spec.get("revision") or None
    output_path = OUT_DIR / f"model_{model_index}_generations.jsonl"
    rows, metadata = generate_model_rows(
        model_name,
        list(read_jsonl(prompts_path)),
        model_id=model_spec.get("id"),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        model_revision=revision,
        seed=int(config["generation"]["seed"]),
        trust_remote_code=bool(config["generation"].get("trust_remote_code", False)),
        quantization_mode=str(config["generation"].get("quantization_mode", "nf4_double")),
        checkpoint_path=output_path,
    )
    write_json(
        OUT_DIR / f"model_{model_index}_metadata.json",
        {**metadata, "gpu_slot": gpu_id, "completed_utc": utc_now()},
    )
    print(f"Worker {model_index} completed {len(rows)} rows on GPU slot {gpu_id}", flush=True)


def launch_workers(prompts_path: Path, config_path: Path, model_count: int, gpu_count: int) -> list[dict]:
    errors = []
    for batch_start in range(0, model_count, gpu_count):
        processes = []
        for slot, model_index in enumerate(range(batch_start, min(batch_start + gpu_count, model_count))):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
            env["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-model-index",
                str(model_index),
                "--gpu-id",
                str(slot),
                "--prompts",
                str(prompts_path),
                "--config",
                str(config_path),
            ]
            processes.append((model_index, subprocess.Popen(command, env=env)))
        for model_index, process in processes:
            return_code = process.wait()
            if return_code != 0:
                errors.append({"model_index": model_index, "return_code": return_code})
    return errors


def finalize(prompts_path: Path, config_path: Path, errors: list[dict]) -> None:
    configure_package_path()
    from rlvr_safety.choice import SCORE_FIELDS, choice_summary_markdown, score_generation
    from rlvr_safety.io import read_jsonl, sha256_file, write_csv, write_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_rows = list(read_jsonl(prompts_path))
    generations = []
    model_metadata = []
    for model_index in range(len(config["models"])):
        output_path = OUT_DIR / f"model_{model_index}_generations.jsonl"
        if output_path.exists():
            generations.extend(read_jsonl(output_path))
        metadata_path = OUT_DIR / f"model_{model_index}_metadata.json"
        if metadata_path.exists():
            model_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    expected_rows = len(prompt_rows) * len(config["models"])
    if len(generations) != expected_rows:
        errors.append({"expected_rows": expected_rows, "observed_rows": len(generations)})

    combined_path = OUT_DIR / "choice_generations_combined.jsonl"
    write_jsonl(combined_path, generations)
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    scored = [score_generation(row, max_new_tokens=max_new_tokens) for row in generations]
    write_csv(OUT_DIR / "choice_scores.csv", scored, SCORE_FIELDS)
    (OUT_DIR / "choice_score_summary.md").write_text(
        choice_summary_markdown(scored), encoding="utf-8"
    )

    malformed = sum(row["malformed"] == "true" for row in scored)
    capped = sum(row["verbosity_confounded"] == "true" for row in scored)
    if malformed or capped:
        errors.append({"malformed": malformed, "token_capped": capped})

    write_json(
        OUT_DIR / "run_summary.json",
        {
            "schema_version": 1,
            "completed_utc": utc_now(),
            "had_error": bool(errors),
            "errors": errors,
            "rows": len(generations),
            "expected_rows": expected_rows,
            "malformed": malformed,
            "token_capped": capped,
            "prompt_sha256": sha256_file(prompts_path),
            "config_sha256": sha256_file(config_path),
            "models": model_metadata,
        },
    )
    if errors:
        raise SystemExit(f"Run failed strict completion gate: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-model-index", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    if args.worker_model_index is not None:
        if args.prompts is None or args.config is None:
            raise SystemExit("worker requires --prompts and --config")
        run_worker(args.worker_model_index, args.gpu_id, args.prompts, args.config)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT_DIR / "run_metadata.json",
        {
            "schema_version": 1,
            "started_utc": utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "requested_dependencies": PINNED_DEPENDENCIES,
        },
    )
    ensure_dependencies()
    source_root = configure_package_path()
    prompts_path = find_one("**/choice_factorial_v1.jsonl")
    config_path = find_one("**/wording_position_v2_legacy_quant.json")
    names = gpu_names()
    if not names or any("T4" not in name for name in names):
        raise SystemExit(
            f"This run requires T4 hardware; observed {names}. Push with "
            "--accelerator NvidiaTeslaT4."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata_path = OUT_DIR / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "package_source": str(source_root),
            "package_versions": package_versions(),
            "gpus": names,
            "prompt_path": str(prompts_path),
            "config_path": str(config_path),
        }
    )
    write_json(metadata_path, metadata)

    errors = launch_workers(
        prompts_path,
        config_path,
        len(config["models"]),
        len(names),
    )
    finalize(prompts_path, config_path, errors)
    print("Counterbalanced factorial run completed successfully.", flush=True)


if __name__ == "__main__":
    main()
