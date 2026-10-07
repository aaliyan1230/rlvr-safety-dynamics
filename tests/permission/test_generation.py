from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlvr_safety.permission.generation import (
    GREEDY_OVERRIDES,
    HFPermissionProvider,
    build_effective_config,
    classify_stop,
    greedy_violations,
    sequence_logprob,
    snapshot_file_hashes,
    verify_weights,
)


class FakeGenerationConfig:
    """Mimics transformers.GenerationConfig.update/to_dict for configs we do not control."""

    def __init__(self, **values):
        self.__dict__.update(values)
        self.eos_token_id = values.get("eos_token_id", [2, 3])

    def update(self, **kwargs):
        unused = {}
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                unused[key] = value
        return unused

    def to_dict(self):
        return dict(self.__dict__)


SAMPLING = dict(
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    num_beams=2,
    num_beam_groups=1,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    length_penalty=1.2,
    diversity_penalty=0.0,
    renormalize_logits=False,
    min_length=0,
    min_new_tokens=None,
    max_new_tokens=None,
    pad_token_id=None,
    use_cache=True,
)


class PureFunctionTests(unittest.TestCase):
    def test_stop_status(self):
        self.assertEqual(classify_stop([2], [2], False, 1), "complete")
        self.assertEqual(classify_stop([7], [2], False, 1), "length")
        self.assertEqual(classify_stop([7], [2], True, 8), "timeout")
        self.assertEqual(classify_stop([], [2], False, 8), "error")

    def test_inherited_sampling_settings_are_overridden_to_greedy(self):
        inherited = FakeGenerationConfig(**SAMPLING)
        effective, as_dict, unused = build_effective_config(inherited, 2048, 2)
        self.assertEqual(greedy_violations(as_dict), {})
        self.assertEqual(
            (as_dict["num_beams"], as_dict["repetition_penalty"], as_dict["no_repeat_ngram_size"]),
            (1, 1.0, 0),
        )
        self.assertIsNone(as_dict["temperature"])
        self.assertEqual((as_dict["max_new_tokens"], as_dict["pad_token_id"]), (2048, 2))
        self.assertTrue(set(unused) <= set(GREEDY_OVERRIDES))
        # the inherited configuration is left untouched so it can be recorded
        self.assertTrue(inherited.do_sample)
        self.assertEqual(inherited.num_beams, 2)

    def test_non_greedy_result_is_refused(self):
        class Stubborn(FakeGenerationConfig):
            def update(self, **kwargs):
                kwargs.pop("num_beams")
                return super().update(**kwargs)

        with self.assertRaisesRegex(ValueError, "not greedy"):
            build_effective_config(Stubborn(**SAMPLING), 2048, 2)

    def test_greedy_violations_names_each_offender(self):
        self.assertEqual(
            greedy_violations({"num_beams": 4, "do_sample": False, "repetition_penalty": 1.3}),
            {"num_beams": 4, "repetition_penalty": 1.3},
        )

    def test_sequence_logprob_sums_token_log_probabilities(self):
        rows = [[math.log(0.5), math.log(0.5)], [math.log(0.25), math.log(0.75)]]
        self.assertAlmostEqual(sequence_logprob(rows, [0, 1]), math.log(0.5 * 0.75))
        with self.assertRaises(ValueError):
            sequence_logprob(rows, [0])
        with self.assertRaises(ValueError):
            sequence_logprob([], [])
        self.assertEqual(sequence_logprob([[-1e9]], [0]), -1e4)

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
            bad = {"siblings": [{"rfilename": "../x.safetensors", "lfs": {"sha256": expected}}]}
            with self.assertRaisesRegex(ValueError, "invalid model weight path"):
                verify_weights(root, bad)
            with self.assertRaisesRegex(ValueError, "no safetensors"):
                verify_weights(root, {"siblings": []})

    def test_snapshot_file_hashes_cover_present_files_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generation_config.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            hashes = snapshot_file_hashes(root)
            self.assertEqual(set(hashes), {"generation_config.json", "tokenizer_config.json"})


class Inputs(dict):
    def to(self, device):
        return self


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
        return '{"action": "a"}' + ("" if skip_special_tokens else "<eos>")


class Output:
    def __getitem__(self, key):
        return SimpleNamespace(tolist=lambda: [7, 2])


class Model:
    is_quantized = False
    generation_config = FakeGenerationConfig(**SAMPLING)
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


class ProviderWiringTests(unittest.TestCase):
    def test_provider_records_full_provenance_and_generates_with_the_effective_config(self):
        tokenizer, model = Tokenizer(), Model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "generation_config.json").write_text('{"temperature": 0.6}')
            expected = hashlib.sha256(b"weights").hexdigest()
            metadata = {
                "id": "example/model",
                "sha": "a" * 40,
                "siblings": [{"rfilename": "model.safetensors", "lfs": {"sha256": expected}}],
            }
            config = {
                "model": {
                    "repo": "example/model",
                    "revision": "a" * 40,
                    "label": "step0",
                    "chat_template_sha256": hashlib.sha256(
                        tokenizer.chat_template.encode()
                    ).hexdigest(),
                },
                "generation": {"max_new_tokens": 2048, "seed": 0, "per_response_seconds": 120},
                "runtime_pins": {"torch": "2.8.0"},
                "launch": {"image": "runpod/pytorch:test"},
                "readouts": ["generate"],
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
                patch("rlvr_safety.permission.generation.configure_downloads"),
                patch(
                    "rlvr_safety.permission.generation.importlib.metadata.version",
                    return_value="2.8.0",
                ),
                patch("rlvr_safety.permission.generation._driver_version", return_value="570.1"),
            ):
                provider = HFPermissionProvider(config, metadata, root)
                messages = [{"role": "user", "content": "decide"}]
                response = provider.respond(
                    {
                        "messages": messages,
                        "condition_id": "condition",
                        "sample_id": "0",
                        "answer_candidates": {},
                        "hidden_label": "forbidden",
                    }
                )
                prov = provider.provenance
                self.assertEqual(prov["inherited_generation_config"]["temperature"], 0.6)
                self.assertIsNone(prov["effective_generation_config"]["temperature"])
                self.assertEqual(prov["effective_generation_config"]["num_beams"], 1)
                self.assertEqual(prov["effective_generation_config"]["max_new_tokens"], 2048)
                self.assertEqual(prov["image"], "runpod/pytorch:test")
                self.assertEqual(prov["nvidia_driver"], "570.1")
                self.assertEqual(prov["label"], "step0")
                self.assertIn("generation_config.json", prov["snapshot_file_sha256"])
                self.assertEqual(prov["eos_token_ids"], [2, 3])
                self.assertIs(model.kwargs["generation_config"], provider.effective)
                self.assertEqual(model.kwargs["attention_mask"], "mask")
                self.assertFalse(tokenizer.tokenize_kwargs["add_special_tokens"])
                self.assertNotIn("hidden_label", response["rendered_prompt"])
                self.assertEqual(response["stop_status"], "complete")
                self.assertEqual(response["generated_token_ids"], [7, 2])
                self.assertNotIn("answer_logprobs", response)
                changed = {**config, "model": {**config["model"], "chat_template_sha256": "bad"}}
                with self.assertRaisesRegex(ValueError, "template changed"):
                    HFPermissionProvider(changed, metadata, root)
                changed = {**config, "runtime_pins": {"torch": "bad"}}
                with self.assertRaisesRegex(ValueError, "runtime pin"):
                    HFPermissionProvider(changed, metadata, root)


if __name__ == "__main__":
    unittest.main()
