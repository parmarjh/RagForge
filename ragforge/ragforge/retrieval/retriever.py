"""RAGForge — retriever with vector search."""

from __future__ import annotations

import logging

from ragforge.core.types import SearchResult
from ragforge.embeddings.base import BaseEmbedding
from ragforge.vectorstore.base import BaseVectorStore

logger = logging.getLogger("ragforge.retriever")


class Retriever:
    """Retrieve relevant chunks from the vector store."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding: BaseEmbedding,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        self._store = vector_store
        self._embedding = embedding
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve the most relevant chunks for a query."""
        k = top_k or self._top_k
        query_embedding = self._embedding.embed(query)
        results = self._store.search(
            embedding=query_embedding,
            top_k=k,
            score_threshold=self._score_threshold,
        )
        logger.debug(f"Retrieved {len(results)} results for query (top_k={k})")
        return results
