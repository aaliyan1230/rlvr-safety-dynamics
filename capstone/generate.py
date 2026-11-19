"""Greedy vLLM generation over the frozen confirmation requests, for the parent or a LoRA adapter.

Prompts are rendered with the pinned chat template and passed as token IDs without a BOS token,
which reproduces the October runner's rendered_prompt_sha256 and prompt_tokens exactly.
Output: one response row per request in the schema validate_response() accepts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="/workspace/models/tulu-dpo")
    p.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="label=path to a LoRA adapter; repeat for several. Omit for the parent only.",
    )
    p.add_argument("--parent", action="store_true", help="also generate for the parent")
    p.add_argument("--system-prompt", help="replace the system message (nudge check only)")
    p.add_argument("--formats", default="executable,mcq,option_free")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-lora-rank", type=int, default=64)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    # vLLM 0.11 under transformers 5 silently drops rope_theta (500000 -> 10000).
    if AutoConfig.from_pretrained(a.model).rope_theta != 500000.0:
        raise SystemExit("rope_theta is not 500000: pin transformers==4.57.1")
    formats = set(a.formats.split(","))
    requests = [json.loads(line) for line in a.requests.open()]
    requests = [r for r in requests if r["format"] in formats]
    prompts, rendered = [], []
    for r in requests:
        messages = r["messages"]
        if a.system_prompt is not None:
            messages = [{"role": "system", "content": a.system_prompt}] + [
                m for m in messages if m["role"] != "system"
            ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered.append(text)
        prompts.append({"prompt_token_ids": tok(text, add_special_tokens=False)["input_ids"]})

    adapters = dict(item.split("=", 1) for item in a.adapter)
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        enable_lora=bool(adapters),
        skip_tokenizer_init=True,
        max_lora_rank=a.max_lora_rank,
        max_model_len=8192,
        seed=0,
        gpu_memory_utilization=0.85,
    )
    # vLLM's tokenizer wrapper breaks under transformers 5, so decode with the pinned HF tokenizer.
    params = SamplingParams(
        temperature=0.0,
        max_tokens=a.max_tokens,
        seed=0,
        detokenize=False,
        stop_token_ids=[tok.eos_token_id],
    )
    targets = ([("parent", None)] if a.parent or not adapters else []) + [
        (label, LoRARequest(label, i + 1, path)) for i, (label, path) in enumerate(adapters.items())
    ]

    a.out.mkdir(parents=True, exist_ok=True)
    for label, lora in targets:
        started = time.time()
        outputs = llm.generate(prompts, params, lora_request=lora)
        path = a.out / f"{label}.responses.jsonl"
        with path.open("w") as f:
            for r, text, out in zip(requests, rendered, outputs, strict=True):
                c = out.outputs[0]
                f.write(
                    json.dumps(
                        {
                            "episode_id": r["episode_id"],
                            "condition_id": r["condition_id"],
                            "sample_id": r.get("sample_id", "0"),
                            "raw_response": tok.decode(c.token_ids, skip_special_tokens=True),
                            "generated_tokens": len(c.token_ids),
                            "stop_status": "complete" if c.finish_reason == "stop" else "length",
                            "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "prompt_tokens": len(out.prompt_token_ids),
                            "system_prompt_override": a.system_prompt,
                        }
                    )
                    + "\n"
                )
        print(
            json.dumps(
                {
                    "label": label,
                    "rows": len(outputs),
                    "seconds": round(time.time() - started, 1),
                    "path": str(path),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
