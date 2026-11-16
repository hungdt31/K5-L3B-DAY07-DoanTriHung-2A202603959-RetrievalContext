"""Thin Gemini client: retries on transient errors, counts calls/tokens/time. Never mocks."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import errors, types

log = logging.getLogger(__name__)
RETRYABLE = {429, 500, 502, 503, 504}


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    retries: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def add(self, other: "Usage") -> None:
        for k in ("calls", "prompt_tokens", "output_tokens", "thought_tokens", "retries", "seconds"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.errors += other.errors

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "thought_tokens": self.thought_tokens, "retries": self.retries,
                "seconds": round(self.seconds, 2), "errors": self.errors}


_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not set (.env)")
        _client = genai.Client()
    return _client


def generate(model: str, contents, config: types.GenerateContentConfig, usage: Usage,
             max_retries: int = 6) -> types.GenerateContentResponse:
    """generate_content with exponential backoff on 429/5xx; other errors propagate."""
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            response = client().models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            usage.seconds += time.perf_counter() - started
            if exc.code not in RETRYABLE or attempt == max_retries:
                usage.errors.append(f"{exc.code}: {str(exc.message)[:200]}")
                raise
            wait = min(2 ** attempt * 2, 60)
            usage.retries += 1
            log.warning("%s %s, retry %d/%d in %ds", model, exc.code, attempt + 1, max_retries, wait)
            time.sleep(wait)
            continue
        usage.seconds += time.perf_counter() - started
        usage.calls += 1
        meta = response.usage_metadata
        if meta:
            usage.prompt_tokens += meta.prompt_token_count or 0
            usage.output_tokens += meta.candidates_token_count or 0
            usage.thought_tokens += meta.thoughts_token_count or 0
        return response
    raise AssertionError("unreachable")
