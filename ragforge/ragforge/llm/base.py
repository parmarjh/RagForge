"""RAGForge — abstract base LLM provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Async version of generate."""
        ...
