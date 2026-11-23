"""Thin wrapper around Google Gemini.

The rest of the system only depends on the `Generator` protocol (a callable that
takes a prompt and returns text), so tests can swap in a fake generator and the
app can run without a network connection.
"""

import logging
import os
import time
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
MAX_ATTEMPTS = 3
RETRYABLE_CODES = {429, 500, 503}   # rate limit / temporary server problems
RETRY_DELAY_SECONDS = 2.0


class LLMUnavailable(Exception):
    """Raised when the LLM cannot be used (no key, network error, bad response)."""


class Generator(Protocol):
    def __call__(self, prompt: str) -> str: ...


class GeminiGenerator:
    """Calls Gemini and returns the raw JSON text of its answer."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        if not self.api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not set")

        # Imported lazily so the app (and tests) work without the SDK configured.
        from google import genai

        self._client = genai.Client(api_key=self.api_key)

    def __call__(self, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.2,  # low temperature -> more consistent answers
            response_mime_type="application/json",
        )
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                break
            except Exception as exc:  # network, auth, quota, unknown model, ...
                code = getattr(exc, "code", None)
                # Full error goes to the log file; the exception message
                # stays short because it is shown to the listener.
                logger.warning(
                    "Gemini attempt %d/%d failed (model=%s): %s",
                    attempt, MAX_ATTEMPTS, self.model, exc,
                )
                if code in RETRYABLE_CODES and attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)
                    continue
                raise LLMUnavailable(
                    f"Gemini request failed ({code or type(exc).__name__})"
                ) from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMUnavailable("Gemini returned an empty response")
        return text


def make_default_generator() -> Optional[Generator]:
    """Return a Gemini generator, or None (and a log line) if it can't be built."""
    try:
        return GeminiGenerator()
    except LLMUnavailable as exc:
        logger.warning("LLM disabled, using rule-based fallback: %s", exc)
        return None
