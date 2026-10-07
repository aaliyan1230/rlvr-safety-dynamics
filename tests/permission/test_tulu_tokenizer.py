"""Opt-in check of letter tokenization against the real Tülu tokenizer.

Set RLVR_TULU_TOKENIZER_DIR to a snapshot directory holding tokenizer.json and
tokenizer_config.json for allenai/Llama-3.1-Tulu-3-8B-DPO @ a7beb67e. Verified on 2026-10-07:
the prompt ends with "<|assistant|>\\n", and A/B/C are single tokens 32/33/34, the same first
tokens the model generated for every multiple-choice reply in the 5 October smoke.
"""

from __future__ import annotations

import os
import unittest

DIRECTORY = os.environ.get("RLVR_TULU_TOKENIZER_DIR")


@unittest.skipUnless(DIRECTORY, "RLVR_TULU_TOKENIZER_DIR not set")
class TuluTokenizerTests(unittest.TestCase):
    def test_letters_are_single_first_tokens_after_the_assistant_header(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DIRECTORY)
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertTrue(rendered.endswith("<|assistant|>\n"))
        ids = {x: tokenizer(x, add_special_tokens=False)["input_ids"] for x in "ABC"}
        self.assertEqual(ids, {"A": [32], "B": [33], "C": [34]})


if __name__ == "__main__":
    unittest.main()
