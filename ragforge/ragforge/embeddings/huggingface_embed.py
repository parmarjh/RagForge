"""RAGForge — HuggingFace sentence-transformers embedding provider."""

from __future__ import annotations

from ragforge.embeddings.base import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):
    """Generate embeddings using local HuggingFace sentence-transformers."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model
        self._model = SentenceTransformer(model, device=device)
        self._dims = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        return self._dims
