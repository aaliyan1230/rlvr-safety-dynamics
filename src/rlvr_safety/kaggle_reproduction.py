"""Restartable Kaggle execution for protocol-specific historical reproductions."""

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
from typing import Any

PROFILE_CONFIGS = {
    "stage": "historical_stage_reproduction_v1.json",
    "paraphrase": "historical_paraphrase_reproduction_v1.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def select_prompt_rows(
    rows: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the declared historical-layout filter without inspecting outcomes."""

    wordings = set(selection.get("wording_ids", []))
    orders = set(selection.get("option_orders", []))
    matches_source_order = selection.get("matches_source_order")
    layout_mode = selection.get("layout_mode")
    selected = []
    for row in rows:
        if wordings and str(row.get("wording_id")) not in wordings:
            continue
        if layout_mode == "historical_pack_layouts":
            wording = str(row.get("wording_id"))
            source_order = str(row.get("matches_source_order", "")).lower() == "true"
            candidate_order = str(row.get("option_order")) == "012"
            if not (
                (wording == "original" and source_order)
                or (wording in {"p1", "p2"} and candidate_order)
            ):
                continue
        elif layout_mode is not None:
            raise ValueError(f"unknown layout_mode: {layout_mode}")
        if orders and str(row.get("option_order")) not in orders:
            continue
        if matches_source_order is not None:
            actual = str(row.get("matches_source_order", "")).strip().lower() == "true"
            if actual is not bool(matches_source_order):
                continue
        selected.append(row)
    return selected


def find_one(input_dir: Path, pattern: str) -> Path:
    matches = sorted(input_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {input_dir}, found {matches}")
    return matches[0]


def ensure_dependencies(config: dict[str, Any]) -> None:
    command = [sys.executable, "-m", "pip", "install", "-q"]
    if config.get("upgrade_dependencies"):
        command.append("-U")
    command.extend(str(value) for value in config["dependency_install"])
    subprocess.check_call(command)


def package_versions() -> dict[str, str]:
    names = [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "tokenizers",
        "huggingface-hub",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def validate_runtime(config: dict[str, Any], versions: dict[str, str]) -> None:
    expected = config["expected_runtime"]
    if platform.python_version() != expected["python"]:
        raise RuntimeError(
            f"Python runtime mismatch: expected {expected['python']}, "
            f"observed {platform.python_version()}"
        )
    mismatches = {
        name: {"expected": version, "observed": versions.get(name, "missing")}
        for name, version in expected["packages"].items()
        if versions.get(name) != version
    }
    if mismatches:
        raise RuntimeError(f"historical package runtime mismatch: {mismatches}")


def gpu_names() -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def run_worker(
    *,
    model_index: int,
    gpu_id: int,
    prompts_path: Path,
    config_path: Path,
    out_dir: Path,
) -> None:
    from .hf_generation import generate_model_rows
    from .io import read_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = config["models"][model_index]
    generation = config["generation"]
    output_path = out_dir / f"model_{model_index}_generations.jsonl"
    rows, metadata = generate_model_rows(
        model_spec["name"],
        list(read_jsonl(prompts_path)),
        max_new_tokens=int(generation["max_new_tokens"]),
        model_revision=model_spec["revision"],
        seed=int(generation["seed"]),
        trust_remote_code=bool(generation["trust_remote_code"]),
        quantization_mode=str(generation["quantization_mode"]),
        inference_context=str(generation["inference_context"]),
        checkpoint_path=output_path,
    )
    write_json(
        out_dir / f"model_{model_index}_metadata.json",
        {**metadata, "gpu_slot": gpu_id, "completed_utc": utc_now()},
    )
    print(f"Worker {model_index} completed {len(rows)} rows on GPU slot {gpu_id}", flush=True)


def launch_workers(
    *,
    script_path: Path,
    prompts_path: Path,
    config_path: Path,
    model_count: int,
    gpu_count: int,
) -> list[dict[str, Any]]:
    errors = []
    for batch_start in range(0, model_count, gpu_count):
        processes = []
        batch_stop = min(batch_start + gpu_count, model_count)
        for slot, model_index in enumerate(range(batch_start, batch_stop)):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
            env["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable,
                str(script_path),
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
    *,
    source_prompts_path: Path,
    prompts_path: Path,
    config_path: Path,
    out_dir: Path,
    errors: list[dict[str, Any]],
) -> None:
    from .choice import SCORE_FIELDS, choice_summary_markdown, score_generation
    from .io import read_jsonl, sha256_file, write_csv, write_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    generations = []
    model_metadata = []
    for model_index in range(len(config["models"])):
        generation_path = out_dir / f"model_{model_index}_generations.jsonl"
        if generation_path.exists():
            generations.extend(read_jsonl(generation_path))
        metadata_path = out_dir / f"model_{model_index}_metadata.json"
        if metadata_path.exists():
            model_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    expected_rows = int(config["expected_generation_rows"])
    if len(generations) != expected_rows:
        errors.append({"expected_rows": expected_rows, "observed_rows": len(generations)})
    combined_path = out_dir / "choice_generations_combined.jsonl"
    write_jsonl(combined_path, generations)
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    scored = [score_generation(row, max_new_tokens=max_new_tokens) for row in generations]
    write_csv(out_dir / "choice_scores.csv", scored, SCORE_FIELDS)
    (out_dir / "choice_score_summary.md").write_text(
        choice_summary_markdown(scored), encoding="utf-8"
    )
    malformed = sum(str(row["malformed"]).lower() == "true" for row in scored)
    capped = sum(str(row["verbosity_confounded"]).lower() == "true" for row in scored)
    if malformed or capped:
        errors.append({"malformed": malformed, "token_capped": capped})
    for metadata in model_metadata:
        if metadata.get("model_is_quantized") is not True:
            errors.append(
                {
                    "model": metadata.get("model"),
                    "model_is_quantized": metadata.get("model_is_quantized"),
                }
            )

    write_json(
        out_dir / "run_summary.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "completed_utc": utc_now(),
            "had_error": bool(errors),
            "errors": errors,
            "rows": len(generations),
            "expected_rows": expected_rows,
            "malformed": malformed,
            "token_capped": capped,
            "source_prompt_sha256": sha256_file(source_prompts_path),
            "filtered_prompt_sha256": sha256_file(prompts_path),
            "config_sha256": sha256_file(config_path),
            "models": model_metadata,
        },
    )
    if errors:
        raise SystemExit(f"Historical reproduction failed strict gate: {errors}")


def main(
    *,
    profile: str,
    script_path: Path,
    package_source: Path,
    input_dir: Path = Path("/kaggle/input"),
    working_dir: Path = Path("/kaggle/working"),
    argv: list[str] | None = None,
) -> None:
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"unknown historical reproduction profile: {profile}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-model-index", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)

    config_filename = PROFILE_CONFIGS[profile]
    out_dir = working_dir / Path(config_filename).stem
    if args.worker_model_index is not None:
        if args.prompts is None or args.config is None:
            raise SystemExit("worker requires --prompts and --config")
        run_worker(
            model_index=args.worker_model_index,
            gpu_id=args.gpu_id,
            prompts_path=args.prompts,
            config_path=args.config,
            out_dir=out_dir,
        )
        return

    from .io import read_jsonl, sha256_file, write_jsonl

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = find_one(input_dir, f"**/{config_filename}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    write_json(
        out_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "started_utc": utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "requested_docker_image": config["docker_image"],
            "historical_dependency_install": config["historical_dependency_install"],
            "dependency_install": config["dependency_install"],
            "upgrade_dependencies": config["upgrade_dependencies"],
        },
    )
    ensure_dependencies(config)
    versions = package_versions()
    validate_runtime(config, versions)
    source_prompts_path = find_one(input_dir, "**/choice_factorial_v1.jsonl")
    source_rows = list(read_jsonl(source_prompts_path))
    selected = select_prompt_rows(source_rows, config["prompt_filter"])
    if len(selected) != int(config["expected_prompt_rows"]):
        raise RuntimeError(
            f"historical prompt filter produced {len(selected)} rows; "
            f"expected {config['expected_prompt_rows']}"
        )
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise RuntimeError("historical prompt filter produced duplicate ids")
    prompts_path = out_dir / "filtered_prompts.jsonl"
    write_jsonl(prompts_path, selected)
    names = gpu_names()
    if not names or any("T4" not in name for name in names):
        raise SystemExit(f"Historical reproduction requires T4 hardware; observed {names}")

    metadata_path = out_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "package_source": str(package_source),
            "package_versions": versions,
            "gpus": names,
            "source_prompt_path": str(source_prompts_path),
            "source_prompt_sha256": sha256_file(source_prompts_path),
            "filtered_prompt_path": str(prompts_path),
            "filtered_prompt_sha256": sha256_file(prompts_path),
            "filtered_prompt_rows": len(selected),
            "prompt_filter": config["prompt_filter"],
            "config_path": str(config_path),
        }
    )
    write_json(metadata_path, metadata)
    errors = launch_workers(
        script_path=script_path,
        prompts_path=prompts_path,
        config_path=config_path,
        model_count=len(config["models"]),
        gpu_count=len(names),
    )
    finalize(
        source_prompts_path=source_prompts_path,
        prompts_path=prompts_path,
        config_path=config_path,
        out_dir=out_dir,
        errors=errors,
    )
    print(f"Historical {profile} reproduction completed successfully.", flush=True)
