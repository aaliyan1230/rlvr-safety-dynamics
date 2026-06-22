#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def strip_thinking_trace(text: str) -> str:
    """Best-effort removal for models that expose <think>...</think> blocks."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts_seed.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--strip-thinking", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install torch and transformers on the GPU machine."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f:
        for row in load_prompts(args.prompts):
            messages = [{"role": "user", "content": row["prompt"]}]
            if hasattr(tokenizer, "apply_chat_template"):
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(row["prompt"], return_tensors="pt").input_ids

            encoded = encoded.to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0, encoded.shape[-1] :]
            raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            response = strip_thinking_trace(raw_response) if args.strip_thinking else raw_response

            out = {
                **row,
                "model": args.model,
                "response": response,
                "raw_response": raw_response,
                "response_chars": len(response),
                "raw_response_chars": len(raw_response),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote generations to {args.out}")


if __name__ == "__main__":
    main()
