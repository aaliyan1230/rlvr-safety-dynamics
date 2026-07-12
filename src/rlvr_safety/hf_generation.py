"""Hugging Face generation primitives shared by local and Kaggle runners."""

from __future__ import annotations

import gc
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .choice import SYSTEM_PROMPT


def encode_prompt(tokenizer: Any, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, dict):
            return encoded["input_ids"]
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids
        return encoded
    return tokenizer(f"{system_prompt}\n\n{prompt}", return_tensors="pt").input_ids


def generate_model_rows(
    model_name: str,
    prompts: Iterable[Mapping[str, Any]],
    *,
    model_id: str | None = None,
    max_new_tokens: int = 96,
    model_revision: str | None = None,
    seed: int = 0,
    trust_remote_code: bool = False,
    quantization_mode: str = "nf4_double",
    inference_context: str = "inference_mode",
    attention_mask_mode: str = "omitted",
    system_prompt: str = SYSTEM_PROMPT,
    checkpoint_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate deterministic choice responses while preserving every source field."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

    if inference_context == "inference_mode":
        inference_guard = torch.inference_mode
    elif inference_context == "no_grad":
        inference_guard = torch.no_grad
    else:
        raise ValueError(f"unknown inference_context: {inference_context}")
    if attention_mask_mode not in {"omitted", "explicit_all_ones"}:
        raise ValueError(f"unknown attention_mask_mode: {attention_mask_mode}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    if quantization_mode == "legacy_default":
        quantization = BitsAndBytesConfig(load_in_4bit=True)
        device_map: Any = "auto"
        dtype_kwargs = {}
    elif quantization_mode == "nf4_double":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        device_map = {"": 0}
        dtype_kwargs = {"torch_dtype": torch.float16}
    else:
        raise ValueError(f"unknown quantization_mode: {quantization_mode}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization,
        **dtype_kwargs,
    )
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    model_is_quantized = bool(getattr(model, "is_quantized", False))
    resolved_quantization = getattr(model.config, "quantization_config", None)
    if hasattr(resolved_quantization, "to_dict"):
        resolved_quantization = resolved_quantization.to_dict()
    elif resolved_quantization is not None and not isinstance(
        resolved_quantization, (dict, list, str, int, float, bool)
    ):
        resolved_quantization = repr(resolved_quantization)

    analysis_model_id = model_id or model_name
    rows: list[dict[str, Any]] = []
    peak_gpu_memory_bytes: int | None = None
    try:
        prompt_rows = list(prompts)
        for index, row in enumerate(prompt_rows, start=1):
            print(f"[{model_name}] {index}/{len(prompt_rows)} {row['id']}", flush=True)
            encoded = encode_prompt(tokenizer, str(row["prompt"]), system_prompt=system_prompt).to(
                model.device
            )
            generation_kwargs: dict[str, Any] = {}
            if attention_mask_mode == "explicit_all_ones":
                generation_kwargs["attention_mask"] = torch.ones_like(encoded)
            with inference_guard():
                output_ids = model.generate(
                    encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                )
            generated_ids = output_ids[0, encoded.shape[-1] :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            rows.append(
                {
                    **row,
                    "model": analysis_model_id,
                    "model_repo": model_name,
                    "model_revision_requested": model_revision or "main",
                    "model_revision_resolved": resolved_revision or "unknown",
                    "response": response,
                    "raw_response": response,
                    "response_chars": len(response),
                    "raw_response_chars": len(response),
                    "generated_tokens": int(generated_ids.shape[-1]),
                    "strip_thinking_applied": False,
                    "thinking_trace_exposed": "<think>" in response or "</think>" in response,
                    "generation_seed": seed,
                }
            )
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("w", encoding="utf-8") as handle:
                    for output_row in rows:
                        handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    finally:
        if torch.cuda.is_available() and hasattr(torch.cuda, "max_memory_allocated"):
            peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "model": analysis_model_id,
        "model_repo": model_name,
        "model_revision_requested": model_revision or "main",
        "model_revision_resolved": resolved_revision or "unknown",
        "rows": len(rows),
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "quantization_mode": quantization_mode,
        "trust_remote_code": trust_remote_code,
        "inference_context": inference_context,
        "attention_mask_mode": attention_mask_mode,
        "system_prompt": system_prompt,
        "model_is_quantized": model_is_quantized,
        "quantization_config_resolved": resolved_quantization,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }
    return rows, metadata
