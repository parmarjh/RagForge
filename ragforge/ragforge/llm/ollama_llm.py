"""RAGForge — Ollama local LLM provider."""

from __future__ import annotations

import aiohttp

from ragforge.llm.base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    """Generate responses using a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._base_url = (base_url or "http://localhost:11434").rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        import requests

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if system:
            payload["system"] = system

        response = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data.get("response", ""),
            model=self._model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )

    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        return LLMResponse(
            text=data.get("response", ""),
            model=self._model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
