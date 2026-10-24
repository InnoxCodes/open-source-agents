"""Thin model layer. Everything above it only sees complete_json()."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

DEFAULT_MODEL = "gemini-flash-latest"
RETRY_CODES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    """Parse a model reply into a JSON object, tolerating ``` fences and stray prose."""
    if text is None:
        raise LLMError("model returned an empty reply")
    cleaned = text.strip()
    fence = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some replies wrap the object in a sentence. Try the outermost braces.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"model reply is not JSON: {cleaned[:200]!r}")
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model reply is not valid JSON ({exc}): {cleaned[:200]!r}")
    if not isinstance(value, dict):
        raise LLMError(f"expected a JSON object, got {type(value).__name__}")
    return value


class LLM:
    name = "llm"

    def complete_json(self, system: str, messages: list[dict]) -> dict:
        raise NotImplementedError


class GeminiLLM(LLM):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 retries: int = 2, backoff: float = 2.0, client: Any = None):
        self.model = model or os.environ.get("FIRSTRUN_MODEL") or DEFAULT_MODEL
        self.name = f"gemini:{self.model}"
        self.retries = retries
        self.backoff = backoff
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMError("no Gemini API key: set GEMINI_API_KEY or GOOGLE_API_KEY")
            from google import genai

            self._client = genai.Client(api_key=key)

    def _contents(self, messages: list[dict]):
        from google.genai import types

        contents = []
        for m in messages:
            role = "model" if m.get("role") == "model" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["text"])]))
        return contents

    def complete_json(self, system: str, messages: list[dict]) -> dict:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0.2,
        )
        contents = self._contents(messages)
        attempt = 0
        while True:
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                break
            except errors.APIError as exc:
                if exc.code in RETRY_CODES and attempt < self.retries:
                    attempt += 1
                    time.sleep(self.backoff * attempt)
                    continue
                raise LLMError(f"Gemini API error {exc.code}: {exc.message or exc}") from exc
            except Exception as exc:  # network errors, bad model names, ...
                raise LLMError(f"Gemini request failed: {exc}") from exc
        return parse_json(getattr(resp, "text", None))


class FakeLLM(LLM):
    """Replays canned replies. Each item is a dict, a raw string, or an Exception to raise."""

    name = "fake"

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_json(self, system: str, messages: list[dict]) -> dict:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        if not self.responses:
            raise LLMError("FakeLLM ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return parse_json(item)
        return dict(item)


def get_llm(offline: bool = False) -> Optional[LLM]:
    if offline:
        return None
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return None
    return GeminiLLM()
