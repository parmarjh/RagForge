"""RAGForge — Qdrant vector store implementation."""

from __future__ import annotations

import logging
from typing import Any

from ragforge.core.types import Chunk, SearchResult
from ragforge.vectorstore.base import BaseVectorStore

logger = logging.getLogger("ragforge.qdrant")


class QdrantStore(BaseVectorStore):
    """Vector store backed by Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "ragforge",
        api_key: str | None = None,
        dimensions: int = 1536,
        on_disk: bool = False,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self._collection = collection
        self._dimensions = dimensions

        # Support both remote and in-memory modes
        if url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=url, api_key=api_key)

        # Ensure collection exists
        collections = [c.name for c in self._client.get_collections().collections]
        if collection not in collections:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Distance.COSINE,
                    on_disk=on_disk,
                ),
            )
            logger.info(f"Created Qdrant collection '{collection}' (dim={dimensions})")

    def upsert(self, chunks: list[Chunk]) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload={
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                )
            )

        # Batch upsert (Qdrant handles batching internally)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=self._collection, points=batch)

        logger.debug(f"Upserted {len(points)} points to '{self._collection}'")

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
        )

        search_results = []
        for rank, hit in enumerate(response.points):
            payload = hit.payload or {}
            chunk = Chunk(
                text=payload.get("text", ""),
                doc_id=payload.get("doc_id", ""),
                chunk_id=str(hit.id),
                chunk_index=payload.get("chunk_index", 0),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in ("text", "doc_id", "chunk_index")
                },
            )
            search_results.append(
                SearchResult(chunk=chunk, score=hit.score, rank=rank)
            )

        return search_results

    def delete(self, chunk_ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=chunk_ids),
        )

    def count(self) -> int:
        result = self._client.count(collection_name=self._collection)
        return result.count

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)
        logger.info(f"Deleted Qdrant collection '{self._collection}'")
