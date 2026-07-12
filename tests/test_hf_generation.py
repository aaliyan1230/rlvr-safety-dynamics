from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from rlvr_safety.hf_generation import encode_prompt, generate_model_rows


class TensorLike:
    pass


class DictTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {"input_ids": TensorLike()}


class ObjectResult:
    def __init__(self) -> None:
        self.input_ids = TensorLike()


class ObjectTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ObjectResult()


class FakeTensor:
    shape = (1, 3)

    def to(self, device):
        return self


class FakeGenerated:
    shape = (1,)


class FakeOutput:
    def __getitem__(self, key):
        return FakeGenerated()


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return FakeTensor()

    def decode(self, generated, skip_special_tokens=True):
        return "CHOICE: A\nREASON: synthetic"


class FakeModel:
    device = "cuda:0"
    is_quantized = True
    config = SimpleNamespace(_commit_hash="resolved-commit")
    last_generate_kwargs = None

    def eval(self):
        return None

    def generate(self, encoded, **kwargs):
        type(self).last_generate_kwargs = kwargs
        return FakeOutput()


class NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class GenerationHelpersTests(unittest.TestCase):
    def test_encode_prompt_handles_dict_and_object_sdk_shapes(self) -> None:
        dict_tokenizer = DictTokenizer()
        self.assertIsInstance(encode_prompt(dict_tokenizer, "prompt"), TensorLike)
        self.assertEqual(dict_tokenizer.messages[-1]["content"], "prompt")
        self.assertIsInstance(encode_prompt(ObjectTokenizer(), "prompt"), TensorLike)

    def test_generation_preserves_prompt_schema_and_checkpoints(self) -> None:
        fake_torch = SimpleNamespace(
            float16="float16",
            manual_seed=lambda seed: None,
            ones_like=lambda value: "explicit-mask",
            cuda=SimpleNamespace(
                is_available=lambda: False,
                manual_seed_all=lambda seed: None,
                empty_cache=lambda: None,
            ),
            inference_mode=NoopContext,
            no_grad=NoopContext,
        )
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeModel()
            ),
            BitsAndBytesConfig=lambda **kwargs: kwargs,
        )
        prompt = {
            "id": "item",
            "source_id": "source",
            "prompt": "Choose",
            "options": [{"label": "A", "text": "safe", "instrumental_score_0_2": 0}],
            "score_key": {"A": 0},
        }
        with TemporaryDirectory() as raw_tmp:
            checkpoint = Path(raw_tmp) / "rows.jsonl"
            with patch.dict(
                "sys.modules", {"torch": fake_torch, "transformers": fake_transformers}
            ):
                rows, metadata = generate_model_rows(
                    "fake/model",
                    [prompt],
                    model_id="step_1920",
                    trust_remote_code=True,
                    quantization_mode="legacy_default",
                    inference_context="no_grad",
                    attention_mask_mode="explicit_all_ones",
                    checkpoint_path=checkpoint,
                )
            self.assertTrue(checkpoint.is_file())
        self.assertEqual(rows[0]["score_key"], {"A": 0})
        self.assertEqual(rows[0]["options"], prompt["options"])
        self.assertEqual(rows[0]["model_revision_resolved"], "resolved-commit")
        self.assertEqual(rows[0]["model"], "step_1920")
        self.assertEqual(rows[0]["model_repo"], "fake/model")
        self.assertEqual(metadata["model"], "step_1920")
        self.assertEqual(metadata["model_repo"], "fake/model")
        self.assertEqual(metadata["rows"], 1)
        self.assertEqual(metadata["quantization_mode"], "legacy_default")
        self.assertTrue(metadata["trust_remote_code"])
        self.assertEqual(metadata["inference_context"], "no_grad")
        self.assertEqual(metadata["attention_mask_mode"], "explicit_all_ones")
        self.assertEqual(FakeModel.last_generate_kwargs["attention_mask"], "explicit-mask")
        self.assertTrue(metadata["model_is_quantized"])

    def test_generation_rejects_unknown_attention_mask_mode(self) -> None:
        fake_torch = SimpleNamespace(
            float16="float16",
            manual_seed=lambda seed: None,
            cuda=SimpleNamespace(is_available=lambda: False),
            inference_mode=NoopContext,
            no_grad=NoopContext,
        )
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeModel()
            ),
            BitsAndBytesConfig=lambda **kwargs: kwargs,
        )
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            with self.assertRaisesRegex(ValueError, "unknown attention_mask_mode"):
                generate_model_rows("fake/model", [], attention_mask_mode="implicit")


if __name__ == "__main__":
    unittest.main()
