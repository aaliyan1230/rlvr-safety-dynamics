"""Fail-closed Kaggle T4x2 runner for one pinned Tülu trajectory wave."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
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
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return source_root


def gpu_names() -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
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


def validate_wave_config(config: dict[str, Any]) -> None:
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("a trajectory wave must contain exactly two checkpoints")
    if len({int(model["step"]) for model in models}) != 2:
        raise ValueError("trajectory wave steps must be unique")
    if any(len(str(model.get("revision", ""))) != 40 for model in models):
        raise ValueError("every checkpoint must use a full 40-character revision")
    design = config["design"]
    expected = int(design["conditions_per_checkpoint"]) * len(models)
    if expected != int(design["total_generations"]):
        raise ValueError("trajectory wave total_generations is inconsistent")
    if expected != int(config["gates"]["expected_rows"]):
        raise ValueError("trajectory wave expected_rows is inconsistent")
    if config["gates"].get("scientific_interpretation_allowed") is not False:
        raise ValueError("individual trajectory waves must prohibit scientific interpretation")


def validate_prompt_pack(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    expected = int(config["design"]["conditions_per_checkpoint"])
    if len(rows) != expected or len({str(row["id"]) for row in rows}) != expected:
        raise ValueError(f"expected {expected} unique structured prompts, observed {len(rows)}")
    cells = {
        (str(row["source_id"]), str(row["wording_id"]), str(row["option_order"]))
        for row in rows
    }
    if len(cells) != expected:
        raise ValueError("structured prompt pack has duplicate factorial cells")
    if len({row["source_id"] for row in rows}) != int(config["design"]["source_items"]):
        raise ValueError("structured prompt pack does not cover all source items")


def run_worker(
    model_index: int,
    gpu_id: int,
    prompts_path: Path,
    config_path: Path,
    out_dir: Path,
) -> None:
    configure_package_path()
    from rlvr_safety.hf_generation import generate_model_rows
    from rlvr_safety.io import read_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_wave_config(config)
    model_spec = config["models"][model_index]
    output_path = out_dir / f"model_{model_index}_generations.jsonl"
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
        system_prompt=str(config["generation"]["system_prompt"]),
        checkpoint_path=output_path,
    )
    write_json(
        out_dir / f"model_{model_index}_metadata.json",
        {**metadata, "step": model_spec["step"], "gpu_slot": gpu_id, "completed_utc": utc_now()},
    )
    cache_path = os.environ.get("HF_HOME")
    if cache_path:
        shutil.rmtree(cache_path, ignore_errors=True)
    print(f"Worker {model_index} completed {len(rows)} rows on GPU slot {gpu_id}", flush=True)


def launch_workers(
    wrapper_path: Path,
    prompts_path: Path,
    config_path: Path,
    out_dir: Path,
    model_count: int,
    gpu_count: int,
) -> list[dict[str, Any]]:
    errors = []
    processes = []
    for model_index in range(model_count):
        slot = model_index % gpu_count
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HOME"] = str(out_dir / f"hf_cache_model_{model_index}")
        command = [
            sys.executable,
            str(wrapper_path),
            "--worker-model-index",
            str(model_index),
            "--gpu-id",
            str(slot),
            "--prompts",
            str(prompts_path),
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
        ]
        processes.append((model_index, subprocess.Popen(command, env=env)))
    for model_index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            errors.append({"model_index": model_index, "return_code": return_code})
    return errors


def finalize(
    source_prompts_path: Path,
    config_path: Path,
    out_dir: Path,
    errors: list[dict[str, Any]],
) -> None:
    configure_package_path()
    from rlvr_safety.choice import SCORE_FIELDS, choice_summary_markdown, score_generation
    from rlvr_safety.io import read_jsonl, sha256_file, write_csv, write_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    generations = []
    model_metadata = []
    for model_index in range(len(config["models"])):
        output_path = out_dir / f"model_{model_index}_generations.jsonl"
        if output_path.exists():
            generations.extend(read_jsonl(output_path))
        metadata_path = out_dir / f"model_{model_index}_metadata.json"
        if metadata_path.exists():
            model_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    expected_rows = int(config["gates"]["expected_rows"])
    if len(generations) != expected_rows:
        errors.append({"expected_rows": expected_rows, "observed_rows": len(generations)})
    unique_cells = {(str(row.get("model")), str(row.get("id"))) for row in generations}
    if len(unique_cells) != len(generations):
        errors.append({"duplicate_generation_cells": len(generations) - len(unique_cells)})
    combined_path = out_dir / "choice_generations_combined.jsonl"
    write_jsonl(combined_path, generations)
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    scored = [score_generation(row, max_new_tokens=max_new_tokens) for row in generations]
    write_csv(out_dir / "choice_scores.csv", scored, SCORE_FIELDS)
    (out_dir / "choice_score_summary.md").write_text(
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
    expected_per_model = int(config["gates"]["expected_rows_per_checkpoint"])
    maximum_rate = float(config["gates"]["max_malformed_or_capped_rate"])
    for model_spec in config["models"]:
        quality = quality_by_model.get(model_spec["id"], {"rows": 0, "malformed": 0, "token_capped": 0})
        if quality["rows"] != expected_per_model:
            errors.append({"model": model_spec["id"], "expected_rows": expected_per_model, "quality": quality})
        bad = quality["malformed"] + quality["token_capped"]
        if quality["rows"] and bad / quality["rows"] > maximum_rate:
            errors.append({"model": model_spec["id"], "quality_gate": quality})
    metadata_by_model = {str(row.get("model")): row for row in model_metadata}
    memory_limit = int(float(config["gates"]["max_gpu_memory_gb"]) * 1024**3)
    for model_spec in config["models"]:
        metadata = metadata_by_model.get(model_spec["id"])
        if metadata is None:
            errors.append({"model": model_spec["id"], "missing_metadata": True})
            continue
        if metadata.get("model_revision_resolved") != model_spec["revision"]:
            errors.append(
                {
                    "model": model_spec["id"],
                    "requested_revision": model_spec["revision"],
                    "resolved_revision": metadata.get("model_revision_resolved"),
                }
            )
        if metadata.get("model_is_quantized") is not True:
            errors.append({"model": model_spec["id"], "model_is_quantized": False})
        peak = metadata.get("peak_gpu_memory_bytes")
        if peak is not None and int(peak) > memory_limit:
            errors.append({"model": model_spec["id"], "peak_gpu_memory_bytes": peak})
    write_json(
        out_dir / "run_summary.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "completed_utc": utc_now(),
            "scientific_interpretation_allowed": False,
            "had_error": bool(errors),
            "errors": errors,
            "rows": len(generations),
            "expected_rows": expected_rows,
            "unique_cells": len(unique_cells),
            "quality_by_model": quality_by_model,
            "source_prompt_sha256": sha256_file(source_prompts_path),
            "config_sha256": sha256_file(config_path),
            "models": model_metadata,
        },
    )
    if errors:
        raise SystemExit(f"Trajectory wave failed strict integrity gate: {errors}")


def main(wrapper_path: Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-model-index", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.worker_model_index is not None:
        if args.prompts is None or args.config is None or args.out_dir is None:
            raise SystemExit("worker requires --prompts, --config, and --out-dir")
        run_worker(args.worker_model_index, args.gpu_id, args.prompts, args.config, args.out_dir)
        return
    ensure_dependencies()
    source_root = configure_package_path()
    source_prompts_path = find_one("**/choice_factorial_v1.jsonl")
    config_path = find_one("**/tulu_grpo_trajectory_wave_01.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_wave_config(config)
    out_dir = WORKING / config["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    from rlvr_safety.io import read_jsonl

    prompt_rows = list(read_jsonl(source_prompts_path))
    validate_prompt_pack(prompt_rows, config)
    names = gpu_names()
    if len(names) != 2 or any("T4" not in name for name in names):
        raise SystemExit(f"This trajectory wave requires exactly T4x2; observed {names}")
    write_json(
        out_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "started_utc": utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "requested_dependencies": PINNED_DEPENDENCIES,
            "package_source": str(source_root),
            "package_versions": package_versions(),
            "gpus": names,
            "source_prompt_path": str(source_prompts_path),
            "source_prompt_rows": len(prompt_rows),
            "config_path": str(config_path),
        },
    )
    actual_wrapper = wrapper_path or Path(sys.argv[0]).resolve()
    errors = launch_workers(
        actual_wrapper,
        source_prompts_path,
        config_path,
        out_dir,
        len(config["models"]),
        len(names),
    )
    finalize(source_prompts_path, config_path, out_dir, errors)
    print("Tülu trajectory wave 01 completed its integrity gate.", flush=True)
