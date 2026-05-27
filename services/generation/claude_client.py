"""
services/generation/claude_client.py

Anthropic Claude implementation of LLMClient.
Instantiate once and inject into ReviewLoop (and agent modules).
"""

import logging
import os
import random
import threading
import time

import anthropic

from config import MAX_TOKENS, MODEL
from services.generation.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting + retry config
# ---------------------------------------------------------------------------
# Two layers of protection:
#   1. Proactive: global token-bucket per model — serialize calls with a
#      minimum interval, so we stay under the per-minute request quota
#      even when 4 parallel families fire calls at the same time.
#   2. Reactive: exponential backoff with jitter when a 429 still slips through
#      (input-token limits, brief bursts, etc.).
#
# Haiku free tier:    5 req/min  → 12.5s/call minimum  →  use 13s
# Sonnet free tier:  50 req/min  →  1.2s/call minimum  →  use 1.5s

_MIN_INTERVALS = {
    "claude-haiku-4-5-20251001": 13.0,
    "claude-sonnet-4-6":          1.5,
}
_DEFAULT_MIN_INTERVAL = 3.0

_rate_lock = threading.Lock()
_last_call_time: dict[str, float] = {}


def _wait_for_rate_limit(model: str) -> None:
    """Sleep just enough so that successive calls respect the per-model interval."""
    interval = _MIN_INTERVALS.get(model, _DEFAULT_MIN_INTERVAL)
    with _rate_lock:
        last = _last_call_time.get(model, 0.0)
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _last_call_time[model] = time.monotonic()


# Retry on 429 even after the rate limiter — protects against input-token bursts.
_MAX_RETRIES = 4
_BASE_DELAY  = 15.0
_MAX_DELAY   = 45.0


def _backoff_delay(attempt: int) -> float:
    """Full-jitter exponential backoff. attempt is 0-indexed."""
    cap = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt))
    return random.uniform(cap / 2, cap)   # keep some floor so we don't spin


class ClaudeClient(LLMClient):
    """
    Wraps the Anthropic Messages API.

    Usage:
        client = ClaudeClient()                      # reads ANTHROPIC_API_KEY from env
        client = ClaudeClient(model="claude-opus-4-7")  # override model
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        timeout: float = 120.0,
    ) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"],
            timeout=timeout,
        )
        self.model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = MAX_TOKENS,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        t0 = time.monotonic()
        logger.info("LLM call start model=%s max_tokens=%d", self.model, max_tokens)

        for attempt in range(_MAX_RETRIES):
            _wait_for_rate_limit(self.model)
            try:
                msg = self._client.messages.create(**kwargs)
                break
            except anthropic.RateLimitError as exc:
                if attempt == _MAX_RETRIES - 1:
                    ms = int((time.monotonic() - t0) * 1000)
                    logger.error("LLM call failed after %dms — %s", ms, exc)
                    raise
                delay = _backoff_delay(attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
            except Exception as exc:
                ms = int((time.monotonic() - t0) * 1000)
                logger.error("LLM call failed after %dms — %s", ms, exc)
                raise

        ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "model=%s in_tokens=%d out_tokens=%d dur=%dms",
            self.model, msg.usage.input_tokens, msg.usage.output_tokens, ms,
        )
        return LLMResponse(
            text=msg.content[0].text,
            model=self.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
