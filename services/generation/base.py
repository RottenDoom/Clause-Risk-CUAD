"""
services/generation/base.py

Abstract LLM client interface. Every LLM call in the agent pipeline goes
through this contract — never through a provider SDK directly.

To add a new provider: subclass LLMClient, implement generate(), pass the
instance into ReviewLoop. No other file needs to change.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(ABC):
    """
    Provider-agnostic LLM interface.

    Implementors must define generate(). generate_json() is provided here
    because the fence-stripping + parse pattern is identical for every caller.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a prompt and return the raw text response."""

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
    ) -> dict:
        """
        Generate and parse a JSON response.

        Strips markdown code fences (```json ... ```) before parsing so that
        models which ignore "respond only in JSON" instructions still work.
        Raises ValueError if the result is not valid JSON after stripping.
        """
        response = self.generate(prompt, system=system, max_tokens=max_tokens)
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", response.text).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM response was not valid JSON after fence-stripping.\n"
                f"Raw text (first 500 chars): {response.text[:500]}\n"
                f"Parse error: {e}"
            ) from e
