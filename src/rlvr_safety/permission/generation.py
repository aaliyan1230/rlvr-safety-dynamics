"""Pinned BF16 Hugging Face provider: greedy generation plus answer log-probability readouts.

The provider records everything needed to reproduce a response: model/tokenizer file hashes, the
inherited and effective generation configuration, runtime versions and the GPU. It refuses any
generation configuration that is not plain greedy decoding after our explicit overrides.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

from ..io import sha256_file
from .schema import digest

USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
MIN_LOGPROB = -1e4
SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
)
# Overrides that turn any inherited configuration into plain greedy decoding.
GREEDY_OVERRIDES = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "typical_p": None,
    "epsilon_cutoff": None,
    "eta_cutoff": None,
    "num_beams": 1,
    "num_beam_groups": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "min_length": 0,
    "min_new_tokens": None,
    "length_penalty": 1.0,
    "diversity_penalty": 0.0,
    "bad_words_ids": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "renormalize_logits": False,
    "use_cache": True,
}
GREEDY_REQUIRED = {
    "do_sample": False,
    "num_beams": 1,
    "num_beam_groups": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
    "diversity_penalty": 0.0,
    "renormalize_logits": False,
}


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


def snapshot_file_hashes(snapshot: Path) -> dict:
    return {n: sha256_file(snapshot / n) for n in SNAPSHOT_FILES if (snapshot / n).is_file()}


def classify_stop(token_ids: list[int], eos_ids: list[int], timed_out: bool, cap: int) -> str:
    if token_ids and token_ids[-1] in eos_ids:
        return "complete"
    if timed_out:
        return "timeout"
    if len(token_ids) >= cap:
        return "length"
    return "error"


def greedy_violations(effective: dict) -> dict:
    """Fields of an effective generation config that are not plain greedy decoding."""
    return {
        key: effective[key]
        for key, required in GREEDY_REQUIRED.items()
        if key in effective and effective[key] != required
    }


def build_effective_config(inherited, max_new_tokens: int, pad_token_id: int):
    """Merge our explicit greedy overrides into the inherited config and verify the result.

    Returns ``(config, config_as_dict, unused_override_names)``. Raises if the merged
    configuration is not plain greedy decoding.
    """
    effective = copy.deepcopy(inherited)
    unused = effective.update(
        **GREEDY_OVERRIDES, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id
    )
    as_dict = effective.to_dict()
    violations = greedy_violations(as_dict)
    if violations:
        raise ValueError(f"effective generation config is not greedy: {violations}")
    return effective, as_dict, sorted(unused) if unused else []


def sequence_logprob(rows: list[list[float]], token_ids: list[int]) -> float:
    """Sum of log P(token_i given prefix); ``rows[i]`` is the log-prob vector for token i."""
    if len(rows) != len(token_ids) or not token_ids:
        raise ValueError("one log-probability row per candidate token is required")
    return max(sum(row[token] for row, token in zip(rows, token_ids, strict=True)), MIN_LOGPROB)


def configure_downloads() -> None:
    import requests
    from huggingface_hub import configure_http_backend

    class Downloader(requests.Session):
        def send(self, request, **kwargs):
            request.headers["User-Agent"] = USER_AGENT
            return super().send(request, **kwargs)

    configure_http_backend(backend_factory=Downloader)


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


class HFPermissionProvider:
    def __init__(self, config: dict, metadata: dict, cache_dir: Path):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("this workload requires exactly one CUDA GPU")
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
            self.snapshot, local_files_only=True, trust_remote_code=False
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
            raise ValueError("BF16 workload must not be quantized")
        self.load_seconds = time.monotonic() - started
        eos = self.model.generation_config.eos_token_id
        eos = self.tokenizer.eos_token_id if eos is None else eos
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        if not self.eos_ids or any(type(i) is not int for i in self.eos_ids):
            raise ValueError("explicit native EOS token IDs required")
        inherited = self.model.generation_config
        self.effective, effective_dict, unused = build_effective_config(
            inherited, self.generation["max_new_tokens"], self.tokenizer.eos_token_id
        )
        self.readouts = list(config.get("readouts", ["generate"]))
        launch = config.get("launch", {})
        self._provenance = {
            "provider": "huggingface_bf16",
            "inference_performed": True,
            "model": model_spec["repo"],
            "model_revision": model_spec["revision"],
            "tokenizer_revision": model_spec["revision"],
            "label": model_spec.get("label"),
            "chat_template_sha256": template_hash,
            "weight_sha256": weight_hashes,
            "snapshot_file_sha256": snapshot_file_hashes(self.snapshot),
            "config_sha256": digest(config),
            "generation_settings": self.generation,
            "inherited_generation_config": inherited.to_dict(),
            "effective_generation_config": effective_dict,
            "unused_generation_overrides": unused,
            "eos_token_ids": self.eos_ids,
            "readouts": self.readouts,
            "packages": packages,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "nvidia_driver": _driver_version(),
            "image": launch.get("image"),
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "attention_implementation": "eager",
            "precision": "bf16",
            "quantized": False,
            "trust_remote_code": False,
        }

    @property
    def provenance(self) -> dict:
        return self._provenance

    def close(self) -> None:
        import torch

        del self.model
        torch.cuda.empty_cache()

    def _first_token_readout(self, prompt_ids, candidate_ids: dict[str, list[int]]):
        import torch

        with torch.inference_mode():
            logits = self.model(input_ids=prompt_ids).logits[0, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        top = torch.topk(logp, 5)
        top5 = [
            {"token": self.tokenizer.decode([int(i)]), "prob": float(v.exp())}
            for v, i in zip(top.values, top.indices, strict=True)
        ]
        single = {
            text: max(float(logp[ids[0]]), MIN_LOGPROB)
            for text, ids in candidate_ids.items()
            if len(ids) == 1
        }
        return single, top5

    def _answer_logprobs(self, prompt_ids, candidates: dict[str, str]) -> tuple[dict, list]:
        import torch

        candidate_ids = {
            text: self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in candidates
        }
        if any(not ids for ids in candidate_ids.values()):
            raise ValueError("candidate answer tokenized to nothing")
        single, top5 = self._first_token_readout(prompt_ids, candidate_ids)
        result = dict(single)
        start = prompt_ids.shape[-1]
        for text, ids in candidate_ids.items():
            if text in result:
                continue
            full = torch.cat([prompt_ids, torch.tensor([ids], device=prompt_ids.device)], dim=-1)
            with torch.inference_mode():
                logits = self.model(input_ids=full).logits[0, start - 1 : -1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            rows = logp.tolist()
            result[text] = sequence_logprob(rows, ids)
        return result, top5

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
            request["messages"], tokenize=False, add_generation_prompt=True
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
                generation_config=self.effective,
                stopping_criteria=StoppingCriteriaList([deadline]),
            )
        torch.cuda.synchronize()
        token_ids = generated[0, prompt_tokens:].tolist()
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        final = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        response = {
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
        }
        candidates = request.get("answer_candidates") or {}
        if candidates and "choice_logprobs" in self.readouts:
            logprobs, top5 = self._answer_logprobs(inputs["input_ids"], candidates)
            response["answer_logprobs"] = logprobs
            response["first_token_top5"] = top5
        response["inference_seconds"] = time.monotonic() - started
        response["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        return response


def save_runtime(path: Path, provider: HFPermissionProvider) -> None:
    import sys

    path.write_text(
        json.dumps(
            {
                **provider.provenance,
                "download_hash_and_load_seconds": provider.load_seconds,
                "pip_freeze": subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"], text=True
                ).splitlines(),
            },
            indent=2,
        )
        + "\n"
    )
