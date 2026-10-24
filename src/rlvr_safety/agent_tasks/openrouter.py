"""OpenRouter chat-completions backend for the agent-task pipeline (stdlib HTTP only)."""

from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .core import digest

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
KEY_NAME = "OPENROUTER_API_KEY"
PROJECT_ENV = Path(__file__).resolve().parents[3] / ".env"
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def api_key(path: Path = PROJECT_ENV) -> str:
    key = os.environ.get(KEY_NAME)
    if not key and path.exists():
        if path.stat().st_mode & 0o077:
            raise ValueError("Restrict the project .env to owner access with chmod 600")
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == KEY_NAME:
                parts = shlex.split(value, comments=True)
                if len(parts) != 1:
                    raise ValueError(f"{KEY_NAME} must be one quoted or unquoted value")
                key = parts[0]
    if not key:
        raise ValueError(f"Set {KEY_NAME} in the environment or owner-only project .env")
    if any(character in key for character in "\r\n\x00"):
        raise ValueError(f"Invalid control character in {KEY_NAME}")
    return key


def task_prompt(task: dict) -> str:
    return (
        task["instructions"]
        + "\n\nInput:\n" + json.dumps(task["input"], ensure_ascii=False, indent=1)
        + "\n\nresponse_schema:\n"
        + json.dumps(task["response_schema"], ensure_ascii=False, indent=1)
        + "\n\nReturn only the JSON payload."
    )


def _post(body: dict, key: str, timeout: int) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "rlvr-safety-dynamics"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class OpenRouterBackend:
    """Answer tasks through OpenRouter; settings and provider metadata go into the envelope."""

    def __init__(self, model: str, *, temperature: float = 0.0, seed: int = 0,
                 reasoning_effort: str | None = "medium", provider: str | None = "openai",
                 max_tokens: int = 16000, timeout: int = 600, retries: int = 4,
                 agent: str = "openrouter", post: Callable[[dict, str, int], dict] = _post,
                 sleep: Callable[[float], None] = time.sleep):
        self.model, self.agent, self.post, self.sleep = model, agent, post, sleep
        self.timeout, self.retries = timeout, retries
        self.settings = {"model": model, "temperature": temperature, "seed": seed,
                         "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
        if reasoning_effort:
            self.settings["reasoning"] = {"effort": reasoning_effort}
        if provider:
            self.settings["provider"] = {"only": [provider], "allow_fallbacks": False}

    def complete(self, task: dict) -> dict:
        body = dict(self.settings, messages=[{"role": "user", "content": task_prompt(task)}])
        key = api_key()
        for attempt in range(self.retries + 1):
            try:
                reply = self.post(body, key, self.timeout)
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == self.retries:
                    raise RuntimeError(f"OpenRouter HTTP {error.code}") from None
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.retries:
                    raise RuntimeError(f"OpenRouter unreachable: {type(error).__name__}") from None
            else:
                if "error" in reply and not reply.get("choices"):
                    message = str(reply["error"].get("message", ""))[:200]
                    if attempt == self.retries:
                        raise RuntimeError(f"OpenRouter error: {message}")
                else:
                    return reply
            self.sleep(min(60, 5 * 2**attempt))
        raise AssertionError("unreachable")

    def answer(self, task: dict) -> dict:
        reply = self.complete(task)
        text = reply["choices"][0]["message"].get("content") or ""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").removeprefix("json").strip()
        return {
            "task_id": task["task_id"],
            "task_sha256": task["task_sha256"],
            "agent": self.agent,
            "model_reported": reply.get("model") or self.model,
            "backend": {
                "type": "openrouter",
                "endpoint": ENDPOINT,
                "settings": self.settings,
                "settings_sha256": digest(self.settings),
                "generation_id": reply.get("id"),
                "provider": reply.get("provider"),
                "finish_reason": reply["choices"][0].get("finish_reason"),
                "usage": reply.get("usage"),
                "raw_text_sha256": digest(text),
            },
            "payload": json.loads(stripped),
        }
