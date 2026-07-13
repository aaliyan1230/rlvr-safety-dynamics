"""Fail-closed Kaggle T4x2 runner for frozen Tülu cross-format anchors."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .kaggle_trajectory import (
    INPUT,
    PINNED_DEPENDENCIES,
    WORKING,
    configure_package_path,
    ensure_dependencies,
    find_one,
    gpu_names,
    package_versions,
    utc_now,
    write_json,
)


def validate_config(config: dict[str, Any]) -> None:
    models = config.get("models", [])
    if len(models) != 2 or len({int(model["step"]) for model in models}) != 2:
        raise ValueError("cross-format waves require two distinct checkpoints")
    if any(len(str(model.get("revision", ""))) != 40 for model in models):
        raise ValueError("every checkpoint must use a full revision")
    expected_by_step = config["design"]["expected_rows_by_checkpoint"]
    if set(expected_by_step) != {str(model["step"]) for model in models}:
        raise ValueError("expected row map does not match checkpoints")
    if sum(int(value) for value in expected_by_step.values()) != int(
        config["gates"]["expected_rows"]
    ):
        raise ValueError("cross-format expected rows are inconsistent")
    if config["gates"].get("scientific_interpretation_allowed") is not False:
        raise ValueError("individual cross-format waves must prohibit interpretation")


def panel_prompts(
    capability_rows: list[dict[str, Any]],
    freeform_rows: list[dict[str, Any]],
    config: dict[str, Any],
    step: int,
) -> list[dict[str, Any]]:
    prompts = []
    for panel, rows in (("capability", capability_rows), ("freeform", freeform_rows)):
        panel_config = config["panels"][panel]
        if panel == "freeform" and step not in panel_config["milestone_steps"]:
            continue
        for row in rows:
            prompts.append(
                {
                    **row,
                    "panel": panel,
                    "checkpoint": step,
                    "row_system_prompt": panel_config["system_prompt"],
                    "row_max_new_tokens": int(panel_config["max_new_tokens"]),
                }
            )
    expected = int(config["design"]["expected_rows_by_checkpoint"][str(step)])
    if len(prompts) != expected or len({(row["panel"], row["id"]) for row in prompts}) != expected:
        raise ValueError(f"step {step}: expected {expected} unique panel rows, got {len(prompts)}")
    return prompts


def run_worker(
    model_index: int,
    gpu_id: int,
    capability_path: Path,
    freeform_path: Path,
    config_path: Path,
    out_dir: Path,
) -> None:
    configure_package_path()
    from rlvr_safety.anchors import validate_capability_pack
    from rlvr_safety.hf_generation import generate_model_rows
    from rlvr_safety.io import read_jsonl

    config = json.loads(config_path.read_text())
    validate_config(config)
    capability = validate_capability_pack(read_jsonl(capability_path))
    freeform = list(read_jsonl(freeform_path))
    if len(freeform) != 24 or len({row["id"] for row in freeform}) != 24:
        raise ValueError("expected 24 unique frozen free-form anchors")
    model_spec = config["models"][model_index]
    prompts = panel_prompts(capability, freeform, config, int(model_spec["step"]))
    output_path = out_dir / f"model_{model_index}_generations.jsonl"
    rows, metadata = generate_model_rows(
        model_spec["name"],
        prompts,
        model_id=model_spec["id"],
        max_new_tokens=256,
        model_revision=model_spec["revision"],
        seed=int(config["generation"]["seed"]),
        trust_remote_code=bool(config["generation"]["trust_remote_code"]),
        quantization_mode=str(config["generation"]["quantization_mode"]),
        attention_mask_mode=str(config["generation"]["attention_mask_mode"]),
        system_prompt_field="row_system_prompt",
        max_new_tokens_field="row_max_new_tokens",
        checkpoint_path=output_path,
    )
    write_json(
        out_dir / f"model_{model_index}_metadata.json",
        {
            **metadata,
            "step": model_spec["step"],
            "gpu_slot": gpu_id,
            "panel_rows": dict(Counter(row["panel"] for row in rows)),
            "completed_utc": utc_now(),
        },
    )


def launch_workers(
    wrapper: Path,
    capability_path: Path,
    freeform_path: Path,
    config_path: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    processes = []
    for index in range(2):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(index)
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_DISABLE_XET"] = "1"
        command = [
            sys.executable,
            str(wrapper),
            "--worker-model-index",
            str(index),
            "--gpu-id",
            str(index),
            "--capability",
            str(capability_path),
            "--freeform",
            str(freeform_path),
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
        ]
        processes.append((index, subprocess.Popen(command, env=env)))
    errors = []
    for index, process in processes:
        if process.wait() != 0:
            errors.append({"model_index": index, "return_code": process.returncode})
    shutil.rmtree(Path.home() / ".cache/huggingface", ignore_errors=True)
    return errors


def finalize(config_path: Path, out_dir: Path, errors: list[dict[str, Any]]) -> None:
    configure_package_path()
    from rlvr_safety.io import read_jsonl, sha256_file, write_jsonl

    config = json.loads(config_path.read_text())
    generations = []
    metadata = []
    for index in range(2):
        generations_path = out_dir / f"model_{index}_generations.jsonl"
        metadata_path = out_dir / f"model_{index}_metadata.json"
        if generations_path.exists():
            generations.extend(read_jsonl(generations_path))
        if metadata_path.exists():
            metadata.append(json.loads(metadata_path.read_text()))
    expected = int(config["gates"]["expected_rows"])
    cells = {(str(row.get("model")), str(row.get("panel")), str(row.get("id"))) for row in generations}
    if len(generations) != expected or len(cells) != expected:
        errors.append({"expected_rows": expected, "rows": len(generations), "unique_cells": len(cells)})
    capped = [
        row for row in generations if int(row.get("generated_tokens", 0)) >= int(row["row_max_new_tokens"])
    ]
    if generations and len(capped) / len(generations) > float(config["gates"]["max_token_capped_rate"]):
        errors.append({"token_capped": len(capped), "rows": len(generations)})
    metadata_by_model = {row.get("model"): row for row in metadata}
    memory_limit = int(float(config["gates"]["max_gpu_memory_gb"]) * 1024**3)
    for model in config["models"]:
        observed = metadata_by_model.get(model["id"])
        if not observed:
            errors.append({"model": model["id"], "missing_metadata": True})
        elif observed.get("model_revision_resolved") != model["revision"]:
            errors.append({"model": model["id"], "revision_mismatch": True})
        elif observed.get("model_is_quantized") is not True:
            errors.append({"model": model["id"], "model_is_quantized": False})
        elif int(observed.get("peak_gpu_memory_bytes") or 0) > memory_limit:
            errors.append({"model": model["id"], "memory_gate": observed["peak_gpu_memory_bytes"]})
    write_jsonl(out_dir / "cross_format_generations.jsonl", generations)
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
            "expected_rows": expected,
            "unique_cells": len(cells),
            "token_capped": len(capped),
            "panel_counts": dict(Counter(row.get("panel") for row in generations)),
            "config_sha256": sha256_file(config_path),
            "models": metadata,
        },
    )
    if errors:
        raise SystemExit(f"Cross-format wave failed integrity gate: {errors}")


def main(wrapper_path: Path | None = None, config_name: str = "tulu_cross_format_wave_01.json") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-model-index", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--capability", type=Path)
    parser.add_argument("--freeform", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.worker_model_index is not None:
        if not all((args.capability, args.freeform, args.config, args.out_dir)):
            raise SystemExit("worker requires both packs, config, and output directory")
        run_worker(args.worker_model_index, args.gpu_id, args.capability, args.freeform, args.config, args.out_dir)
        return
    ensure_dependencies()
    source_root = configure_package_path()
    capability_path = find_one("**/tulu_capability_anchor_v1.jsonl")
    freeform_path = find_one("**/tulu_freeform_anchor_v1.jsonl")
    config_path = find_one(f"**/{config_name}")
    config = json.loads(config_path.read_text())
    validate_config(config)
    names = gpu_names()
    if len(names) != 2 or any("T4" not in name for name in names):
        raise SystemExit(f"This cross-format wave requires exactly T4x2; observed {names}")
    out_dir = WORKING / config["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
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
            "huggingface_cache": "shared_default_deleted_after_both_workers",
            "hf_hub_disable_xet": True,
            "gpus": names,
            "capability_path": str(capability_path),
            "freeform_path": str(freeform_path),
            "config_path": str(config_path),
        },
    )
    errors = launch_workers(
        wrapper_path or Path(sys.argv[0]).resolve(),
        capability_path,
        freeform_path,
        config_path,
        out_dir,
    )
    finalize(config_path, out_dir, errors)
    print(f"{config['experiment_id']} completed its integrity gate.", flush=True)
