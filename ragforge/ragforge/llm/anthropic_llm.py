"""RAGForge — Anthropic Claude LLM provider."""

from __future__ import annotations

import os

from ragforge.llm.base import BaseLLM, LLMResponse


class AnthropicLLM(BaseLLM):
    """Generate responses using Anthropic's Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError("anthropic is required. Install with: pip install anthropic")

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        import anthropic as anth
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anth.Anthropic(**kwargs)
        self._aclient = anth.AsyncAnthropic(**kwargs)

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return LLMResponse(
            text=text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await self._aclient.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return LLMResponse(
            text=text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
