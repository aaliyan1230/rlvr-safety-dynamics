"""Exercise the log-probability readout against a tiny random Llama on CPU.

Skipped when torch/transformers are not installed (they are GPU-extra dependencies).
"""

from __future__ import annotations

import json
import math
import unittest

try:
    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
except ImportError:  # pragma: no cover
    torch = None

from rlvr_safety.permission.generation import (
    HFPermissionProvider,
    build_effective_config,
)
from rlvr_safety.permission.scoring import score_readout


def make_tokenizer():
    chars = [chr(i) for i in range(32, 127)]
    vocab = {"<unk>": 0, "<eos>": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TinyModelReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tokenizer = make_tokenizer()
        config = LlamaConfig(
            vocab_size=len(cls.tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        cls.model = LlamaForCausalLM(config).eval()

    def provider(self):
        provider = object.__new__(HFPermissionProvider)
        provider.model = self.model
        provider.tokenizer = self.tokenizer
        return provider

    def prompt_ids(self, text="Task: pick one. Answer:"):
        return torch.tensor([self.tokenizer(text, add_special_tokens=False)["input_ids"]])

    def stepwise_reference(self, prompt, candidate):
        ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
        total = 0.0
        for i, token in enumerate(ids):
            prefix = torch.cat([prompt, torch.tensor([ids[:i]], dtype=prompt.dtype)], dim=-1)
            with torch.no_grad():
                logits = self.model(input_ids=prefix).logits[0, -1].float()
            total += float(torch.log_softmax(logits, dim=-1)[token])
        return total

    def test_single_token_letters_match_a_step_by_step_reference(self):
        prompt = self.prompt_ids()
        logprobs, top5 = self.provider()._answer_logprobs(prompt, {"A": "x", "B": "y", "C": "z"})
        for letter in "ABC":
            self.assertAlmostEqual(logprobs[letter], self.stepwise_reference(prompt, letter), 4)
        self.assertEqual(len(top5), 5)
        self.assertAlmostEqual(sum(t["prob"] for t in top5) <= 1.0 + 1e-6, True)

    def test_multi_token_json_answers_match_a_step_by_step_reference(self):
        prompt = self.prompt_ids()
        candidates = {json.dumps({"action": name}): name for name in ("op_k", "op_q", "op_t")}
        logprobs, _ = self.provider()._answer_logprobs(prompt, candidates)
        for text in candidates:
            self.assertAlmostEqual(logprobs[text], self.stepwise_reference(prompt, text), 3)

    def test_readout_feeds_the_scorer_and_probabilities_sum_to_one(self):
        prompt = self.prompt_ids()
        candidates = {"A": "use_own", "B": "use_other", "C": "ask"}
        logprobs, _ = self.provider()._answer_logprobs(prompt, candidates)
        condition = {"answer_candidates": candidates}
        scored = score_readout(condition, logprobs)
        self.assertAlmostEqual(sum(scored["action_probs"].values()), 1.0)
        self.assertLessEqual(scored["coverage"], 1.0)
        self.assertAlmostEqual(scored["coverage"], sum(math.exp(v) for v in logprobs.values()))

    def test_effective_config_from_real_generation_config_is_greedy(self):
        from transformers import GenerationConfig

        inherited = GenerationConfig(
            do_sample=True, temperature=0.6, top_p=0.9, num_beams=2, repetition_penalty=1.2
        )
        _, as_dict, _ = build_effective_config(inherited, 64, 1)
        self.assertFalse(as_dict["do_sample"])
        self.assertEqual(as_dict["num_beams"], 1)
        self.assertEqual(as_dict["repetition_penalty"], 1.0)
        self.assertEqual(as_dict["max_new_tokens"], 64)
        self.assertEqual(inherited.num_beams, 2)

    def test_greedy_generation_with_the_effective_config_is_deterministic(self):
        from transformers import GenerationConfig

        effective, _, _ = build_effective_config(
            GenerationConfig(do_sample=True, temperature=0.7, num_beams=3), 8, 1
        )
        prompt = self.prompt_ids()
        runs = [
            self.model.generate(input_ids=prompt, generation_config=effective)[0].tolist()
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])


if __name__ == "__main__":
    unittest.main()
