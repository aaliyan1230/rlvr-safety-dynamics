from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from rlvr_safety.agent_tasks import openrouter
from rlvr_safety.agent_tasks.core import make_task

SECRET = "sk-or-test-not-real"


def task():
    return make_task("demo", "demo-rubric-1", "Add one.", {"x": 1}, {"answer": "int"})


def reply(content='{"answer": 2}'):
    return {"id": "gen-1", "model": "openai/gpt-x", "provider": "OpenAI",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.0001}}


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": SECRET})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_envelope_records_settings_and_provider_but_never_the_key(self):
        seen = {}

        def post(body, key, timeout):
            seen.update(body=body, key=key)
            return reply()

        backend = openrouter.OpenRouterBackend("openai/gpt-x", seed=7, post=post)
        envelope = backend.answer(task())
        self.assertEqual(seen["key"], SECRET)
        self.assertEqual(envelope["payload"], {"answer": 2})
        self.assertEqual(envelope["model_reported"], "openai/gpt-x")
        self.assertEqual(envelope["backend"]["generation_id"], "gen-1")
        self.assertEqual(envelope["backend"]["settings"]["seed"], 7)
        self.assertEqual(seen["body"]["provider"], {"only": ["openai"], "allow_fallbacks": False})
        self.assertNotIn(SECRET, json.dumps(envelope))
        self.assertIn("Input:", seen["body"]["messages"][0]["content"])

    def test_code_fences_are_stripped(self):
        backend = openrouter.OpenRouterBackend(
            "m", post=lambda b, k, t: reply('```json\n{"answer": 2}\n```'))
        self.assertEqual(backend.answer(task())["payload"], {"answer": 2})

    def test_retries_transient_errors_then_succeeds(self):
        calls = []

        def post(body, key, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError("u", 429, "busy", {}, io.BytesIO())
            return reply()

        backend = openrouter.OpenRouterBackend("m", post=post, sleep=lambda s: None)
        self.assertEqual(backend.answer(task())["payload"], {"answer": 2})
        self.assertEqual(len(calls), 3)

    def test_permanent_error_raises_without_body_or_key(self):
        def post(body, key, timeout):
            raise urllib.error.HTTPError("u", 401, "nope", {}, io.BytesIO(SECRET.encode()))

        backend = openrouter.OpenRouterBackend("m", post=post, sleep=lambda s: None)
        with self.assertRaises(RuntimeError) as caught:
            backend.answer(task())
        self.assertNotIn(SECRET, str(caught.exception))

    def test_env_file_must_be_owner_only(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            env = Path(tmp) / ".env"
            env.write_text(f"export OPENROUTER_API_KEY='{SECRET}'\n")
            env.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "chmod 600"):
                openrouter.api_key(env)
            env.chmod(0o600)
            self.assertEqual(openrouter.api_key(env), SECRET)


if __name__ == "__main__":
    unittest.main()
