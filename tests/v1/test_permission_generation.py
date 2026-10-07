from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlvr_safety.permission.v1.generation import HFPermissionProvider, classify_stop, verify_weights


class Inputs(dict):
    def to(self, device):
        return self


class Output:
    def __getitem__(self, key):
        return SimpleNamespace(tolist=lambda: [7, 2])


class Tokenizer:
    chat_template = "native template"
    eos_token_id = 2
    last_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        return json.dumps(messages)

    def __call__(self, text, **kwargs):
        self.tokenize_kwargs = kwargs
        return Inputs(input_ids=SimpleNamespace(shape=(1, 3)), attention_mask="mask")

    def decode(self, token_ids, skip_special_tokens):
        return '{"actions": []}' + ("" if skip_special_tokens else "<eos>")


class Model:
    is_quantized = False
    generation_config = SimpleNamespace(eos_token_id=[2, 3])
    config = SimpleNamespace(max_position_embeddings=8192)

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return Output()


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class PermissionGenerationTests(unittest.TestCase):
    def test_stop_metadata_distinguishes_eos_cap_timeout_and_failure(self):
        self.assertEqual(classify_stop([2], [2], False, 1), "complete")
        self.assertEqual(classify_stop([7], [2], False, 1), "length")
        self.assertEqual(classify_stop([7], [2], True, 8), "timeout")
        self.assertEqual(classify_stop([], [2], False, 8), "error")

    def test_weight_verification_rejects_changed_content_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            expected = hashlib.sha256(b"weights").hexdigest()
            metadata = {
                "siblings": [{"rfilename": "model.safetensors", "lfs": {"sha256": expected}}]
            }
            self.assertEqual(verify_weights(root, metadata), {"model.safetensors": expected})
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_weights(root, metadata)
            metadata["siblings"][0]["rfilename"] = "../model.safetensors"
            with self.assertRaisesRegex(ValueError, "path"):
                verify_weights(root, metadata)

    def test_real_adapter_contract_pins_native_wrapper_and_preserves_stop_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            tokenizer, model = Tokenizer(), Model()
            metadata = {
                "id": "example/model",
                "sha": "a" * 40,
                "siblings": [
                    {
                        "rfilename": "model.safetensors",
                        "lfs": {"sha256": hashlib.sha256(b"weights").hexdigest()},
                    }
                ],
            }
            config = {
                "model": {
                    "repo": "example/model",
                    "revision": "a" * 40,
                    "chat_template_sha256": hashlib.sha256(
                        tokenizer.chat_template.encode()
                    ).hexdigest(),
                },
                "generation": {"max_new_tokens": 2048, "seed": 0, "per_response_seconds": 120},
                "runtime_pins": {"torch": "2.8.0"},
            }
            torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    device_count=lambda: 1,
                    get_device_name=lambda n: "test GPU",
                    manual_seed_all=lambda s: None,
                    reset_peak_memory_stats=lambda: None,
                    synchronize=lambda: None,
                    max_memory_allocated=lambda: 100,
                ),
                bfloat16="bf16",
                version=SimpleNamespace(cuda="12.8"),
                manual_seed=lambda s: None,
                inference_mode=Context,
            )
            hub = SimpleNamespace(snapshot_download=lambda **kwargs: str(root))
            transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
                StoppingCriteria=object,
                StoppingCriteriaList=list,
            )
            with (
                patch.dict(
                    "sys.modules",
                    {"torch": torch, "huggingface_hub": hub, "transformers": transformers},
                ),
                patch("rlvr_safety.permission.v1.generation.configure_downloads"),
                patch(
                    "rlvr_safety.permission.v1.generation.importlib.metadata.version",
                    return_value="2.8.0",
                ),
            ):
                provider = HFPermissionProvider(config, metadata, root)
                messages = [{"role": "user", "content": "decide"}]
                response = provider.respond(
                    {
                        "messages": messages,
                        "condition_id": "condition",
                        "sample_id": "sample",
                        "hidden_label": "forbidden",
                    }
                )
                self.assertEqual(tokenizer.last_messages, messages)
                self.assertNotIn("hidden_label", response["rendered_prompt"])
                self.assertEqual(response["stop_status"], "complete")
                self.assertEqual(response["generated_token_ids"], [7, 2])
                self.assertEqual(
                    response["raw_response_with_special_tokens"], '{"actions": []}<eos>'
                )
                self.assertEqual(response["raw_response"], '{"actions": []}')
                self.assertFalse(model.kwargs["do_sample"])
                self.assertEqual(model.kwargs["attention_mask"], "mask")
                self.assertFalse(tokenizer.tokenize_kwargs["add_special_tokens"])
                self.assertEqual(provider.provenance["model_revision"], "a" * 40)
                self.assertEqual(provider.provenance["eos_token_ids"], [2, 3])
                changed = {**config, "model": {**config["model"], "chat_template_sha256": "bad"}}
                with self.assertRaisesRegex(ValueError, "template changed"):
                    HFPermissionProvider(changed, metadata, root)
                changed = {**config, "runtime_pins": {"torch": "bad"}}
                with self.assertRaisesRegex(ValueError, "runtime pin"):
                    HFPermissionProvider(changed, metadata, root)


if __name__ == "__main__":
    unittest.main()
