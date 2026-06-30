#!/usr/bin/env python3
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]


class GeminiError(RuntimeError):
    pass


def get_api_key(env_name: str = "GEMINI_API_KEY") -> str:
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise GeminiError(f"{env_name} is not set")
    return api_key


def _request_json(url: str, payload: dict | None = None, retries: int = 3) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    last_error = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload else "GET")
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise GeminiError(f"Gemini API request failed: {last_error}") from last_error


def list_models(api_key: str) -> list[dict]:
    query = urllib.parse.urlencode({"key": api_key})
    response = _request_json(f"{API_BASE}/models?{query}", payload=None)
    return response.get("models", [])


def select_flash_lite_model(api_key: str, requested: str | None = None) -> str:
    if requested:
        return requested.removeprefix("models/")

    models = list_models(api_key)
    names = [model.get("name", "").removeprefix("models/") for model in models]
    supported = {
        name
        for model, name in zip(models, names)
        if "generateContent" in model.get("supportedGenerationMethods", [])
    }
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate in supported:
            return candidate

    flash_lite = sorted(name for name in supported if "flash-lite" in name)
    if flash_lite:
        return flash_lite[-1]
    raise GeminiError("No Gemini Flash-Lite generateContent model found")


def generate_json(api_key: str, model: str, prompt: str, schema: dict, temperature: float = 0.0) -> dict:
    query = urllib.parse.urlencode({"key": api_key})
    url = f"{API_BASE}/models/{model}:generateContent?{query}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    response = _request_json(url, payload=payload)
    candidates = response.get("candidates", [])
    if not candidates:
        raise GeminiError(f"No candidates returned: {response}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Gemini returned invalid JSON: {text[:500]}") from exc


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
