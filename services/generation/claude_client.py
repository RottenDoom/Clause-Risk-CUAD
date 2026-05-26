"""
services/generation/claude_client.py

Anthropic Claude implementation of LLMClient.
Instantiate once and inject into ReviewLoop (and agent modules).
"""

import logging
import os
import time

import anthropic

from config import MAX_TOKENS, MODEL
from services.generation.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
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
        try:
            msg = self._client.messages.create(**kwargs)
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
