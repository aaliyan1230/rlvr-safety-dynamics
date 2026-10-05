"""Bounded BF16 development inference; no provider credentials enter this module."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from .io import sha256_file
from .permission_schema import digest

USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


def verify_weights(snapshot: Path, metadata: dict) -> dict:
    weights = [s for s in metadata["siblings"] if s["rfilename"].endswith(".safetensors")]
    if not weights:
        raise ValueError("pinned metadata contains no safetensors weights")
    hashes = {}
    for weight in weights:
        filename = weight["rfilename"]
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("invalid model weight path")
        expected = weight.get("lfs", {}).get("sha256")
        if not expected or sha256_file(snapshot / filename) != expected:
            raise ValueError(f"model weight checksum mismatch: {filename}")
        hashes[filename] = expected
    return hashes


def classify_stop(token_ids: list[int], eos_ids: list[int], timed_out: bool, cap: int) -> str:
    if token_ids and token_ids[-1] in eos_ids:
        return "complete"
    if timed_out:
        return "timeout"
    if len(token_ids) >= cap:
        return "length"
    return "error"


def configure_downloads() -> None:
    import requests
    from huggingface_hub import configure_http_backend

    class Downloader(requests.Session):
        def send(self, request, **kwargs):
            request.headers["User-Agent"] = USER_AGENT
            return super().send(request, **kwargs)

    configure_http_backend(backend_factory=Downloader)


class HFPermissionProvider:
    def __init__(self, config: dict, metadata: dict, cache_dir: Path):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("this smoke workload requires exactly one CUDA GPU")
        self.config = config
        self.generation = config["generation"]
        model_spec = config["model"]
        if metadata.get("sha") != model_spec["revision"]:
            raise ValueError("model metadata does not match the pinned revision")
        if metadata.get("id") != model_spec["repo"]:
            raise ValueError("model metadata does not match the requested repository")
        packages = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "huggingface-hub",
                "accelerate",
                "safetensors",
                "tokenizers",
                "sentencepiece",
                "protobuf",
            )
        }
        for name, expected in config["runtime_pins"].items():
            if packages.get(name, "").split("+")[0] != expected:
                raise ValueError(f"runtime pin mismatch: {name}")
        configure_downloads()
        started = time.monotonic()
        self.snapshot = Path(
            snapshot_download(
                repo_id=model_spec["repo"],
                revision=model_spec["revision"],
                cache_dir=str(cache_dir),
                token=False,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"],
            )
        )
        weight_hashes = verify_weights(self.snapshot, metadata)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        template = self.tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError("native string chat template required; no wrapper substitution")
        template_hash = hashlib.sha256(template.encode()).hexdigest()
        if template_hash != model_spec["chat_template_sha256"]:
            raise ValueError("native chat template changed")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.snapshot,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        self.model.eval()
        if getattr(self.model, "is_quantized", False):
            raise ValueError("BF16 smoke model must not be quantized")
        self.load_seconds = time.monotonic() - started
        eos = self.model.generation_config.eos_token_id
        eos = self.tokenizer.eos_token_id if eos is None else eos
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        if not self.eos_ids or any(type(i) is not int for i in self.eos_ids):
            raise ValueError("explicit native EOS token IDs required")
        self._provenance = {
            "provider": "huggingface_bf16_development",
            "inference_performed": True,
            "model": model_spec["repo"],
            "model_revision": model_spec["revision"],
            "tokenizer_revision": model_spec["revision"],
            "chat_template_sha256": template_hash,
            "weight_sha256": weight_hashes,
            "config_sha256": digest(config),
            "generation_settings": self.generation,
            "eos_token_ids": self.eos_ids,
            "packages": packages,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "attention_implementation": "eager",
            "precision": "bf16",
            "quantized": False,
            "trust_remote_code": False,
        }

    @property
    def provenance(self) -> dict:
        return self._provenance

    def respond(self, request: dict) -> dict:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        started = time.monotonic()
        limit = self.generation["per_response_seconds"]

        class Deadline(StoppingCriteria):
            triggered = False

            def __call__(self, input_ids, scores, **kwargs):
                self.triggered = time.monotonic() - started >= limit
                return self.triggered

        deadline = Deadline()
        torch.manual_seed(self.generation["seed"])
        torch.cuda.manual_seed_all(self.generation["seed"])
        rendered = self.tokenizer.apply_chat_template(
            request["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to("cuda")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        cap = self.generation["max_new_tokens"]
        if prompt_tokens + cap > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus generation cap exceeds model context")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=cap,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([deadline]),
            )
        torch.cuda.synchronize()
        token_ids = generated[0, prompt_tokens:].tolist()
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        final = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {
            "condition_id": request["condition_id"],
            "sample_id": request["sample_id"],
            "raw_response": final,
            "raw_response_with_special_tokens": raw,
            "generated_token_ids": token_ids,
            "generated_tokens": len(token_ids),
            "stop_status": classify_stop(token_ids, self.eos_ids, deadline.triggered, cap),
            "rendered_prompt": rendered,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "prompt_tokens": prompt_tokens,
            "inference_seconds": time.monotonic() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        }


def save_runtime(path: Path, provider: HFPermissionProvider) -> None:
    import subprocess
    import sys

    path.write_text(
        json.dumps(
            {
                **provider.provenance,
                "download_hash_and_load_seconds": provider.load_seconds,
                "pip_freeze": subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"],
                    text=True,
                ).splitlines(),
            },
            indent=2,
        )
        + "\n"
    )
