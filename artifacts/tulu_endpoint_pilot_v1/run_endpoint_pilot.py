#!/usr/bin/env python3
"""Kaggle T4x2 feasibility runner for pinned Tulu DPO/GRPO endpoints."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
OUT_DIR = WORKING / "tulu_grpo_endpoint_pilot_v1"
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


def filtered_prompts(source_path: Path, config: dict) -> tuple[list[dict], Path]:
    configure_package_path()
    from rlvr_safety.io import read_jsonl, write_jsonl

    rows = list(read_jsonl(source_path))
    selection = config["prompt_filter"]
    wordings = set(selection["wording_ids"])
    orders = set(selection["option_orders"])
    selected = [
        row
        for row in rows
        if row.get("wording_id") in wordings and row.get("option_order") in orders
    ]
    expected = int(config["design"]["conditions_per_checkpoint"])
    if len(selected) != expected:
        raise RuntimeError(f"prompt filter produced {len(selected)} rows; expected {expected}")
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise RuntimeError("prompt filter produced duplicate ids")
    if len({str(row["source_id"]) for row in selected}) != int(
        config["design"]["source_items"]
    ):
        raise RuntimeError("prompt filter does not cover every declared source item")
    output_path = OUT_DIR / "pilot_prompts.jsonl"
    write_jsonl(output_path, selected)
    return selected, output_path


def run_worker(model_index: int, gpu_id: int, prompts_path: Path, config_path: Path) -> None:
    configure_package_path()
    from rlvr_safety.hf_generation import generate_model_rows
    from rlvr_safety.io import read_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = config["models"][model_index]
    output_path = OUT_DIR / f"model_{model_index}_generations.jsonl"
    rows, metadata = generate_model_rows(
        model_spec["name"],
        list(read_jsonl(prompts_path)),
        model_id=model_spec["id"],
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        model_revision=model_spec["revision"],
        seed=int(config["generation"]["seed"]),
        trust_remote_code=bool(config["generation"].get("trust_remote_code", False)),
        quantization_mode=str(config["generation"]["quantization_mode"]),
        attention_mask_mode=str(config["generation"]["attention_mask_mode"]),
        checkpoint_path=output_path,
    )
    write_json(
        OUT_DIR / f"model_{model_index}_metadata.json",
        {
            **metadata,
            "step": model_spec["step"],
            "gpu_slot": gpu_id,
            "completed_utc": utc_now(),
        },
    )
    print(f"Worker {model_index} completed {len(rows)} rows on GPU slot {gpu_id}", flush=True)


def launch_workers(
    prompts_path: Path,
    config_path: Path,
    model_count: int,
    gpu_count: int,
) -> list[dict]:
    errors = []
    for batch_start in range(0, model_count, gpu_count):
        processes = []
        model_indexes = range(batch_start, min(batch_start + gpu_count, model_count))
        for slot, model_index in enumerate(model_indexes):
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


def finalize(
    source_prompts_path: Path,
    prompts_path: Path,
    config_path: Path,
    errors: list[dict],
) -> None:
    configure_package_path()
    from rlvr_safety.choice import SCORE_FIELDS, choice_summary_markdown, score_generation
    from rlvr_safety.io import read_jsonl, sha256_file, write_csv, write_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    generations = []
    model_metadata = []
    for model_index in range(len(config["models"])):
        output_path = OUT_DIR / f"model_{model_index}_generations.jsonl"
        if output_path.exists():
            generations.extend(read_jsonl(output_path))
        metadata_path = OUT_DIR / f"model_{model_index}_metadata.json"
        if metadata_path.exists():
            model_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    expected_rows = int(config["gates"]["expected_rows"])
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

    quality_by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "malformed": 0, "token_capped": 0}
    )
    for row in scored:
        quality = quality_by_model[str(row["model"])]
        quality["rows"] += 1
        quality["malformed"] += row["malformed"] == "true"
        quality["token_capped"] += row["verbosity_confounded"] == "true"
    maximum_rate = float(config["gates"]["max_malformed_or_capped_rate"])
    for model, quality in quality_by_model.items():
        bad = quality["malformed"] + quality["token_capped"]
        if quality["rows"] and bad / quality["rows"] > maximum_rate:
            errors.append({"model": model, "quality_gate": quality})

    memory_limit = int(float(config["gates"]["max_gpu_memory_gb"]) * 1024**3)
    for metadata in model_metadata:
        if metadata.get("model_is_quantized") is not True:
            errors.append(
                {
                    "model": metadata.get("model"),
                    "model_is_quantized": metadata.get("model_is_quantized"),
                }
            )
        peak = metadata.get("peak_gpu_memory_bytes")
        if peak is not None and int(peak) > memory_limit:
            errors.append(
                {
                    "model": metadata["model"],
                    "peak_gpu_memory_bytes": peak,
                    "limit_bytes": memory_limit,
                }
            )

    write_json(
        OUT_DIR / "run_summary.json",
        {
            "schema_version": 1,
            "completed_utc": utc_now(),
            "scientific_interpretation_allowed": False,
            "had_error": bool(errors),
            "errors": errors,
            "rows": len(generations),
            "expected_rows": expected_rows,
            "quality_by_model": quality_by_model,
            "source_prompt_sha256": sha256_file(source_prompts_path),
            "filtered_prompt_sha256": sha256_file(prompts_path),
            "config_sha256": sha256_file(config_path),
            "models": model_metadata,
        },
    )
    if errors:
        raise SystemExit(f"Pilot failed strict completion gate: {errors}")


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
    source_prompts_path = find_one("**/choice_factorial_v1.jsonl")
    config_path = find_one("**/tulu_grpo_endpoint_pilot_v1.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_rows, prompts_path = filtered_prompts(source_prompts_path, config)
    names = gpu_names()
    if len(names) < 2 or any("T4" not in name for name in names):
        raise SystemExit(f"This pilot requires T4x2 hardware; observed {names}")

    metadata_path = OUT_DIR / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "package_source": str(source_root),
            "package_versions": package_versions(),
            "gpus": names,
            "source_prompt_path": str(source_prompts_path),
            "filtered_prompt_path": str(prompts_path),
            "filtered_prompt_rows": len(prompt_rows),
            "prompt_filter": config["prompt_filter"],
            "config_path": str(config_path),
        }
    )
    write_json(metadata_path, metadata)

    errors = launch_workers(prompts_path, config_path, len(config["models"]), len(names))
    finalize(source_prompts_path, prompts_path, config_path, errors)
    print("Tulu endpoint feasibility pilot completed successfully.", flush=True)


if __name__ == "__main__":
    main()
