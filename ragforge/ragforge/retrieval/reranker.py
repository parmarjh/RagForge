"""RAGForge — result reranking."""

from __future__ import annotations

import logging

from ragforge.core.types import SearchResult
from ragforge.llm.base import BaseLLM

logger = logging.getLogger("ragforge.reranker")


class Reranker:
    """Rerank search results using an LLM for improved relevance."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        """Rerank results by asking the LLM to score relevance."""
        if not results:
            return []

        # Build a scoring prompt
        chunks_text = ""
        for i, r in enumerate(results):
            chunks_text += f"\n[{i}] {r.chunk.text[:300]}\n"

        prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"Rank the following text passages by relevance (most relevant first). "
            f"Return ONLY a comma-separated list of indices, e.g.: 2,0,4,1,3\n"
            f"{chunks_text}"
        )

        try:
            response = self._llm.generate(prompt)
            # Parse the ranking
            indices_str = response.text.strip().split("\n")[0]
            indices = []
            for part in indices_str.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(results):
                        indices.append(idx)

            # Reorder results
            if indices:
                reranked = []
                seen = set()
                for idx in indices:
                    if idx not in seen:
                        reranked.append(results[idx])
                        seen.add(idx)
                # Append any not mentioned
                for i, r in enumerate(results):
                    if i not in seen:
                        reranked.append(r)

                # Update ranks
                for rank, r in enumerate(reranked):
                    r.rank = rank

                return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")

        return results[:top_k]
