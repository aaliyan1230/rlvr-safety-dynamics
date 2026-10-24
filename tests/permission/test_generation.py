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
    build_sampling_config,
    classify_stop,
    condition_seed,
    greedy_violations,
    sequence_logprob,
    snapshot_file_hashes,
    split_sequences,
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

    def test_inherited_logit_modifiers_and_early_stops_are_cleared(self):
        inherited = FakeGenerationConfig(
            **SAMPLING,
            sequence_bias={(7,): 10.0},
            max_time=0.01,
            stop_strings=["stop"],
            encoder_repetition_penalty=1.4,
            num_return_sequences=3,
            guidance_scale=2.0,
            constraints=["forced"],
            force_words_ids=[[7]],
            penalty_alpha=0.5,
            token_healing=True,
            watermarking_config={"bias": 2.0},
        )
        _, effective, _ = build_effective_config(inherited, 64, 2)
        for name in (
            "sequence_bias",
            "max_time",
            "stop_strings",
            "guidance_scale",
            "constraints",
            "force_words_ids",
            "penalty_alpha",
            "watermarking_config",
        ):
            self.assertIsNone(effective[name], name)
        self.assertEqual(effective["num_return_sequences"], 1)
        self.assertEqual(effective["encoder_repetition_penalty"], 1.0)
        self.assertFalse(effective["token_healing"])
        self.assertEqual(inherited.max_time, 0.01)

    def test_native_eos_fallback_is_present_in_the_config_passed_to_generate(self):
        inherited = FakeGenerationConfig(**SAMPLING, eos_token_id=None)
        _, effective, _ = build_effective_config(inherited, 64, 2, eos_token_ids=[2])
        self.assertEqual(effective["eos_token_id"], [2])
        self.assertIsNone(inherited.eos_token_id)

    def test_non_greedy_result_is_refused(self):
        class Stubborn(FakeGenerationConfig):
            def update(self, **kwargs):
                kwargs.pop("num_beams")
                return super().update(**kwargs)

        with self.assertRaisesRegex(ValueError, "not greedy"):
            build_effective_config(Stubborn(**SAMPLING), 2048, 2)

    def test_stubborn_inherited_stopping_fields_are_refused(self):
        class Stubborn(FakeGenerationConfig):
            def update(self, **kwargs):
                kwargs.pop("stop_strings")
                return super().update(**kwargs)

        with self.assertRaisesRegex(ValueError, "stop_strings"):
            build_effective_config(Stubborn(**SAMPLING, stop_strings=["early"]), 64, 2)


    def test_sampling_config_pins_temperature_and_top_p_and_disables_top_k(self):
        inherited = FakeGenerationConfig(**dict(SAMPLING, top_k=50))
        _, as_dict, _ = build_sampling_config(inherited, 2048, 2, 0.6, 0.9)
        self.assertEqual((as_dict["do_sample"], as_dict["temperature"], as_dict["top_p"]),
                         (True, 0.6, 0.9))
        self.assertIsNone(as_dict["top_k"])
        self.assertEqual((as_dict["num_beams"], as_dict["repetition_penalty"]), (1, 1.0))

    def test_sampling_preserves_native_eos_and_clears_inherited_early_stop(self):
        inherited = FakeGenerationConfig(
            **SAMPLING, eos_token_id=None, stop_strings=["early"], max_time=0.01
        )
        _, effective, _ = build_sampling_config(
            inherited, 64, 2, 0.6, 0.9, eos_token_ids=[2]
        )
        self.assertEqual(effective["eos_token_id"], [2])
        self.assertIsNone(effective["stop_strings"])
        self.assertIsNone(effective["max_time"])
        self.assertEqual(inherited.stop_strings, ["early"])

    def test_condition_seeds_differ_and_are_stable(self):
        self.assertEqual(condition_seed(0, "a"), condition_seed(0, "a"))
        self.assertNotEqual(condition_seed(0, "a"), condition_seed(0, "b"))

    def test_batched_sequences_are_trimmed_after_eos(self):
        rows = [[5, 6, 2, 0, 0], [7, 2, 0, 0, 0], [8, 9, 9, 9, 9]]
        self.assertEqual(split_sequences(rows, [2], 0), [[5, 6, 2], [7, 2], [8, 9, 9, 9, 9]])
        self.assertEqual(split_sequences([[8, 0, 0]], [2], 0), [[8]])

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
                "launch": {
                    "image": "runpod/pytorch:test",
                    "image_digest": "sha256:" + "d" * 64,
                },
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
                self.assertEqual(prov["image_digest"], "sha256:" + "d" * 64)
                self.assertEqual(
                    prov["stopping_criteria"],
                    {"type": "wall_clock_deadline", "seconds": 120},
                )
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


class BatchOutput:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        index, cut = key
        return SimpleNamespace(tolist=lambda: self.rows[index][cut])


class SamplingModel(Model):
    calls = 0

    def generate(self, **kwargs):
        self.kwargs = kwargs
        self.calls += 1
        n = kwargs["num_return_sequences"]
        rows = [[0, 0, 0, 10 + i, 2, 0] for i in range(n)]
        return BatchOutput(rows)


class SamplingProviderTests(unittest.TestCase):
    def test_draws_come_from_one_batched_call_with_a_per_condition_seed(self):
        tokenizer, model, seeds = Tokenizer(), SamplingModel(), []
        tokenizer.pad_token_id = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            metadata = {"id": "example/model", "sha": "a" * 40, "siblings": [
                {"rfilename": "model.safetensors",
                 "lfs": {"sha256": hashlib.sha256(b"weights").hexdigest()}}]}
            config = {
                "model": {"repo": "example/model", "revision": "a" * 40, "label": "s",
                          "chat_template_sha256": hashlib.sha256(
                              tokenizer.chat_template.encode()).hexdigest()},
                "generation": {"max_new_tokens": 64, "seed": 0, "per_response_seconds": 120,
                               "do_sample": True, "temperature": 0.6, "top_p": 0.9,
                               "samples": {"option_free": 3}},
                "runtime_pins": {"torch": "2.8.0"},
                "readouts": ["generate"],
            }
            torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                     get_device_name=lambda n: "gpu",
                                     manual_seed_all=lambda s: None,
                                     reset_peak_memory_stats=lambda: None,
                                     synchronize=lambda: None, max_memory_allocated=lambda: 1),
                bfloat16="bf16", version=SimpleNamespace(cuda="12.8"),
                manual_seed=seeds.append, inference_mode=Context)
            transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
                StoppingCriteria=object, StoppingCriteriaList=list)
            hub = SimpleNamespace(snapshot_download=lambda **kwargs: str(root))
            with (
                patch.dict("sys.modules",
                           {"torch": torch, "huggingface_hub": hub, "transformers": transformers}),
                patch("rlvr_safety.permission.generation.configure_downloads"),
                patch("rlvr_safety.permission.generation.importlib.metadata.version",
                      return_value="2.8.0"),
                patch("rlvr_safety.permission.generation._driver_version", return_value="1"),
            ):
                provider = HFPermissionProvider(config, metadata, root)
                eff = provider.provenance["effective_generation_config"]
                self.assertEqual((eff["do_sample"], eff["temperature"], eff["top_p"]),
                                 (True, 0.6, 0.9))
                base = {"messages": [{"role": "user", "content": "x"}], "sample_count": 3,
                        "answer_candidates": {"a": "b"}}
                out = [provider.respond({**base, "condition_id": "c1", "sample_id": str(i)})
                       for i in range(3)]
                self.assertEqual(model.calls, 1)
                self.assertEqual(model.kwargs["num_return_sequences"], 3)
                self.assertEqual([o["generated_token_ids"] for o in out],
                                 [[10, 2], [11, 2], [12, 2]])
                self.assertEqual([o["sample_id"] for o in out], ["0", "1", "2"])
                self.assertTrue(all(o["stop_status"] == "complete" for o in out))
                self.assertNotIn("answer_logprobs", out[0])
                provider.respond({**base, "condition_id": "c2", "sample_id": "0"})
                self.assertEqual(model.calls, 2)
                self.assertEqual(seeds, [condition_seed(0, "c1"), condition_seed(0, "c2")])



if __name__ == "__main__":
    unittest.main()
