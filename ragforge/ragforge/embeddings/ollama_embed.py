"""RAGForge — Ollama local embedding provider."""

from __future__ import annotations

from ragforge.embeddings.base import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    """Generate embeddings using a local Ollama server."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        dimensions: int = 768,
    ) -> None:
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError("requests is required. Install with: pip install requests")

        self._model = model
        self._base_url = (base_url or "http://localhost:11434").rstrip("/")
        self._dims = dimensions

    def embed(self, text: str) -> list[float]:
        import requests

        response = requests.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding", [])
        if embedding and self._dims != len(embedding):
            self._dims = len(embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims
