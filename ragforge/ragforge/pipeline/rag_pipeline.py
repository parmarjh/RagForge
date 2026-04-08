"""RAGForge — full RAG pipeline: retrieve → rerank → generate."""

from __future__ import annotations

import logging
from typing import Any

from ragforge.core.types import GenerationResult, SearchResult
from ragforge.llm.base import BaseLLM
from ragforge.pipeline.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT, format_context
from ragforge.retrieval.reranker import Reranker
from ragforge.retrieval.retriever import Retriever

logger = logging.getLogger("ragforge.pipeline")


class RAGPipeline:
    """Orchestrate the full RAG flow: retrieve, optionally rerank, then generate."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        rerank: bool = False,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        max_context_chars: int = 8000,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._rerank = rerank
        self._reranker = Reranker(llm) if rerank else None
        self._system_prompt = system_prompt or QA_SYSTEM_PROMPT
        self._user_template = user_prompt_template or QA_USER_PROMPT
        self._max_context = max_context_chars

    def query(self, question: str, top_k: int | None = None, **kwargs: Any) -> GenerationResult:
        """Synchronous RAG query."""
        # 1. Retrieve
        results = self._retriever.retrieve(question, top_k=top_k)

        if not results:
            return GenerationResult(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0,
            )

        # 2. Rerank (optional)
        if self._rerank and self._reranker:
            results = self._reranker.rerank(question, results, top_k=top_k or 5)

        # 3. Generate
        return self._generate(question, results, **kwargs)

    async def aquery(self, question: str, top_k: int | None = None, **kwargs: Any) -> GenerationResult:
        """Async RAG query."""
        # Retrieve (sync — embedding call is fast)
        results = self._retriever.retrieve(question, top_k=top_k)

        if not results:
            return GenerationResult(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0,
            )

        # Rerank
        if self._rerank and self._reranker:
            results = self._reranker.rerank(question, results, top_k=top_k or 5)

        # Generate (async)
        return await self._agenerate(question, results, **kwargs)

    def _generate(self, question: str, results: list[SearchResult], **kwargs: Any) -> GenerationResult:
        context_texts = [r.chunk.text for r in results]
        context = format_context(context_texts, max_chars=self._max_context)

        prompt = self._user_template.format(context=context, question=question)
        system = kwargs.get("system_prompt", self._system_prompt)

        response = self._llm.generate(prompt, system=system)

        # Compute average confidence from retrieval scores
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        return GenerationResult(
            answer=response.text,
            sources=results,
            confidence=avg_score,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    async def _agenerate(self, question: str, results: list[SearchResult], **kwargs: Any) -> GenerationResult:
        context_texts = [r.chunk.text for r in results]
        context = format_context(context_texts, max_chars=self._max_context)

        prompt = self._user_template.format(context=context, question=question)
        system = kwargs.get("system_prompt", self._system_prompt)

        response = await self._llm.agenerate(prompt, system=system)

        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        return GenerationResult(
            answer=response.text,
            sources=results,
            confidence=avg_score,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
