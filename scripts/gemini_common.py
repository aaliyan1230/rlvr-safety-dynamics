#!/usr/bin/env python3
import json
import os
import time


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


def get_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiError(
            "Missing official Gemini SDK. Install it with: python3 -m pip install google-genai"
        ) from exc
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ.pop("GEMINI_API_KEY", None)
    return genai.Client(api_key=api_key)


def _model_name(model) -> str:
    name = getattr(model, "name", "") or ""
    return str(name).removeprefix("models/")


def _supported_methods(model) -> set[str]:
    for attr in ("supported_actions", "supported_generation_methods", "supportedGenerationMethods"):
        value = getattr(model, attr, None)
        if value:
            return {str(item) for item in value}
    if isinstance(model, dict):
        value = (
            model.get("supported_actions")
            or model.get("supported_generation_methods")
            or model.get("supportedGenerationMethods")
        )
        if value:
            return {str(item) for item in value}
    return set()


def list_models(api_key: str):
    client = get_client(api_key)
    return list(client.models.list())


def select_flash_lite_model(api_key: str, requested: str | None = None) -> str:
    if requested:
        return requested.removeprefix("models/")

    models = list_models(api_key)
    supported = set()
    fallback_names = set()
    for model in models:
        name = _model_name(model)
        if not name:
            continue
        fallback_names.add(name)
        methods = _supported_methods(model)
        if not methods or "generateContent" in methods:
            supported.add(name)

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate in supported:
            return candidate

    flash_lite = sorted(name for name in supported if "flash-lite" in name)
    if flash_lite:
        return flash_lite[-1]

    fallback_flash_lite = sorted(name for name in fallback_names if "flash-lite" in name)
    if fallback_flash_lite:
        return fallback_flash_lite[-1]
    raise GeminiError("No Gemini Flash-Lite model found through the official SDK")


def _parse_response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    candidates = getattr(response, "candidates", None) or []
    chunks = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        for part in parts or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "".join(chunks).strip()


def generate_json(api_key: str, model: str, prompt: str, schema: dict, temperature: float = 0.0) -> dict:
    client = get_client(api_key)
    config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = _parse_response_text(response)
            return json.loads(text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < 3:
                time.sleep(2**attempt)
    raise GeminiError(f"Gemini SDK generate_content failed: {last_error}") from last_error


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
