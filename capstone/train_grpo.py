"""GRPO + LoRA training of the Tülu DPO parent on permission situations (one arm, one seed).

Plan of record: local/plans/2026-11-17-capstone-plan.md. Rewards come from
rlvr_safety.permission.rl so they stay identical to the tested reward tables.
Every reward call appends one line to <out>/rollouts.jsonl (roles, permissions, rewards).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rlvr_safety.io import read_jsonl
from rlvr_safety.permission import rl

MODEL = "allenai/Llama-3.1-Tulu-3-8B-DPO"
REVISION = "a7beb67e33ffd01cc87ac3b46cadc1000985b8db"


def make_reward(arm: str, seed: int, log_path: Path):
    def reward(prompts, completions, roles, permission, trainer_state=None, **_):
        step = trainer_state.global_step if trainer_state is not None else -1
        texts = [rl.completion_text(c) for c in completions]
        role_maps = [json.loads(r) for r in roles]
        rewards, chosen, coins = rl.batch_rewards(
            arm, texts, role_maps, list(permission), seed=seed, step=step
        )
        by_perm = {
            p: dict(Counter(c for c, q in zip(chosen, permission, strict=True) if q == p))
            for p in ("allowed", "forbidden")
        }
        with log_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "t": time.time(),
                        "step": step,
                        "n": len(texts),
                        "mean_reward": sum(rewards) / len(rewards),
                        "roles": by_perm,
                        "coins": sum(coins),
                    }
                )
                + "\n"
            )
        return rewards

    reward.__name__ = f"{arm}_reward"
    return reward


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=rl.ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--train", type=Path, required=True, help="JSONL from rl.training_rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--max-completion", type=int, default=48)
    p.add_argument("--save-steps", type=int, default=30)
    p.add_argument("--vllm-mem", type=float, default=0.3)
    p.add_argument(
        "--model",
        default="/workspace/models/tulu-dpo",
        help="local snapshot of MODEL at REVISION, so trainer and vLLM load identical weights",
    )
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(a.train))
    keep = ("prompt", "roles", "permission", "condition_id")
    dataset = Dataset.from_list([{k: r[k] for k in keep} for r in rows])
    completions = a.prompts_per_step * a.num_generations
    config = GRPOConfig(
        output_dir=str(a.out),
        seed=a.seed,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        per_device_train_batch_size=completions // 4,
        gradient_accumulation_steps=4,
        num_generations=a.num_generations,
        max_completion_length=a.max_completion,
        temperature=a.temperature,
        top_p=1.0,
        beta=a.beta,
        loss_type="dr_grpo",
        scale_rewards="none",
        mask_truncated_completions=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_mem,
        vllm_max_model_length=2048,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=a.save_steps,
        save_only_model=True,
        report_to="none",
        log_completions=False,
        model_init_kwargs={"dtype": "bfloat16"},
    )
    lora = LoraConfig(
        r=a.lora_r,
        lora_alpha=2 * a.lora_r,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=a.model,
        reward_funcs=make_reward(a.arm, a.seed, a.out / "rollouts.jsonl"),
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )
    (a.out / "run_config.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "arm": a.arm,
                "seed": a.seed,
                "train": str(a.train),
                "train_rows": len(rows),
                **{k: str(v) for k, v in vars(a).items()},
            },
            indent=1,
        )
        + "\n"
    )
    trainer.train()
    trainer.save_model(str(a.out / "final"))
    (a.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
