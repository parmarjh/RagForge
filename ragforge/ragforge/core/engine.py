"""RAGForge — RAGEngine main orchestrator (skeleton, wired in step 7)."""

from __future__ import annotations

import asyncio
import glob as globmod
import logging
from pathlib import Path
from typing import Any

from ragforge.core.config import (
    ChunkConfig,
    EmbeddingConfig,
    LLMConfig,
    RAGConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from ragforge.core.types import Document, GenerationResult, SourceType

logger = logging.getLogger("ragforge")


class RAGEngine:
    """High-level orchestrator for the RAG pipeline.

    Usage::

        engine = RAGEngine(
            embedding_provider="openai",
            llm_provider="anthropic",
            qdrant_url="http://localhost:6333",
        )
        engine.ingest_file("docs/manual.pdf")
        result = engine.query("How does X work?")
    """

    def __init__(
        self,
        *,
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        vector_store: str = "qdrant",
        qdrant_url: str = "http://localhost:6333",
        collection: str = "ragforge",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        rerank: bool = False,
        config: RAGConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config:
            self._config = config
        else:
            self._config = RAGConfig(
                embedding=EmbeddingConfig(
                    provider=embedding_provider,
                    model=embedding_model or self._default_embed_model(embedding_provider),
                    api_key=kwargs.get("embedding_api_key"),
                    base_url=kwargs.get("embedding_base_url"),
                ),
                llm=LLMConfig(
                    provider=llm_provider,
                    model=llm_model or self._default_llm_model(llm_provider),
                    api_key=kwargs.get("llm_api_key"),
                    base_url=kwargs.get("llm_base_url"),
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 2048),
                ),
                vector_store=VectorStoreConfig(
                    provider=vector_store,
                    url=qdrant_url,
                    collection=collection,
                    api_key=kwargs.get("vectorstore_api_key"),
                ),
                chunk=ChunkConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                retrieval=RetrievalConfig(
                    top_k=top_k,
                    rerank=rerank,
                ),
            )

        # Lazy-initialized components
        self._embedding = None
        self._llm = None
        self._vector_store = None
        self._pipeline = None
        self._chunker = None

    # --- Provider defaults ---

    @staticmethod
    def _default_embed_model(provider: str) -> str:
        return {
            "openai": "text-embedding-3-small",
            "huggingface": "all-MiniLM-L6-v2",
            "ollama": "nomic-embed-text",
        }.get(provider, "text-embedding-3-small")

    @staticmethod
    def _default_llm_model(provider: str) -> str:
        return {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "ollama": "llama3",
        }.get(provider, "gpt-4o")

    # --- Lazy component initialization ---

    def _get_embedding(self):
        if self._embedding is None:
            self._embedding = _build_embedding(self._config.embedding)
        return self._embedding

    def _get_llm(self):
        if self._llm is None:
            self._llm = _build_llm(self._config.llm)
        return self._llm

    def _get_vector_store(self):
        if self._vector_store is None:
            self._vector_store = _build_vector_store(
                self._config.vector_store, self._config.embedding.dimensions
            )
        return self._vector_store

    def _get_chunker(self):
        if self._chunker is None:
            from ragforge.ingest.chunker import TextChunker
            self._chunker = TextChunker(
                strategy=self._config.chunk.strategy,
                chunk_size=self._config.chunk.chunk_size,
                chunk_overlap=self._config.chunk.chunk_overlap,
                separators=self._config.chunk.separators,
            )
        return self._chunker

    def _get_pipeline(self):
        if self._pipeline is None:
            from ragforge.pipeline.rag_pipeline import RAGPipeline
            from ragforge.retrieval.retriever import Retriever
            retriever = Retriever(
                vector_store=self._get_vector_store(),
                embedding=self._get_embedding(),
                top_k=self._config.retrieval.top_k,
                score_threshold=self._config.retrieval.score_threshold,
            )
            self._pipeline = RAGPipeline(
                retriever=retriever,
                llm=self._get_llm(),
                rerank=self._config.retrieval.rerank,
            )
        return self._pipeline

    # --- Ingestion methods ---

    def ingest_file(self, path: str, **metadata: Any) -> int:
        """Ingest a single file (PDF, DOCX, TXT, MD, CSV). Returns chunk count."""
        docs = self._load_file(path, metadata)
        return self._ingest_documents(docs)

    def ingest_directory(
        self, path: str, glob: str = "**/*.*", **metadata: Any
    ) -> int:
        """Ingest all matching files in a directory. Returns total chunk count."""
        base = Path(path)
        files = list(base.glob(glob))
        total = 0
        for f in files:
            if f.is_file():
                try:
                    total += self.ingest_file(str(f), **metadata)
                except Exception as e:
                    logger.warning(f"Skipping {f}: {e}")
        return total

    def ingest_url(self, url: str, **metadata: Any) -> int:
        """Ingest content from a web URL. Returns chunk count."""
        from ragforge.ingest.web_loader import WebLoader
        loader = WebLoader()
        docs = loader.load(url, metadata=metadata)
        return self._ingest_documents(docs)

    def ingest_text(self, text: str, source: str = "inline", **metadata: Any) -> int:
        """Ingest raw text directly. Returns chunk count."""
        doc = Document(
            content=text,
            source=source,
            source_type=SourceType.TEXT,
            metadata=metadata,
        )
        return self._ingest_documents([doc])

    def ingest_api(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict | None = None,
        params: dict | None = None,
        json_path: str = "",
        text_field: str = "content",
        **metadata: Any,
    ) -> int:
        """Ingest content from a REST API endpoint. Returns chunk count."""
        from ragforge.ingest.api_loader import APILoader
        loader = APILoader()
        docs = loader.load(
            url,
            method=method,
            headers=headers,
            params=params,
            json_path=json_path,
            text_field=text_field,
            metadata=metadata,
        )
        return self._ingest_documents(docs)

    def ingest_database(
        self, connection_string: str, query: str, text_column: str = "content", **metadata: Any
    ) -> int:
        """Ingest results from a SQL query. Returns chunk count."""
        from ragforge.ingest.db_loader import DatabaseLoader
        loader = DatabaseLoader()
        docs = loader.load(
            connection_string, query=query, text_column=text_column, metadata=metadata
        )
        return self._ingest_documents(docs)

    # --- Query ---

    def query(self, question: str, **kwargs: Any) -> GenerationResult:
        """Run a RAG query: retrieve relevant chunks → generate answer."""
        pipeline = self._get_pipeline()
        return asyncio.get_event_loop().run_until_complete(
            pipeline.aquery(question, **kwargs)
        ) if not asyncio.get_event_loop().is_running() else self._sync_query(question, **kwargs)

    async def aquery(self, question: str, **kwargs: Any) -> GenerationResult:
        """Async version of query."""
        pipeline = self._get_pipeline()
        return await pipeline.aquery(question, **kwargs)

    def _sync_query(self, question: str, **kwargs: Any) -> GenerationResult:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.aquery(question, **kwargs))
        finally:
            loop.close()

    # --- Internal helpers ---

    def _load_file(self, path: str, metadata: dict) -> list[Document]:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            from ragforge.ingest.pdf_loader import PDFLoader
            return PDFLoader().load(path, metadata=metadata)
        elif ext in (".docx", ".doc"):
            from ragforge.ingest.docx_loader import DocxLoader
            return DocxLoader().load(path, metadata=metadata)
        else:
            from ragforge.ingest.text_loader import TextLoader
            return TextLoader().load(path, metadata=metadata)

    def _ingest_documents(self, docs: list[Document]) -> int:
        chunker = self._get_chunker()
        embedding = self._get_embedding()
        store = self._get_vector_store()

        all_chunks = []
        for doc in docs:
            chunks = chunker.split(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        texts = [c.text for c in all_chunks]
        embeddings = embedding.embed_batch(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        store.upsert(all_chunks)
        logger.info(f"Ingested {len(all_chunks)} chunks from {len(docs)} document(s)")
        return len(all_chunks)

    @property
    def config(self) -> RAGConfig:
        return self._config


# --- Factory functions ---

def _build_embedding(cfg: EmbeddingConfig):
    if cfg.provider == "openai":
        from ragforge.embeddings.openai_embed import OpenAIEmbedding
        return OpenAIEmbedding(model=cfg.model, api_key=cfg.api_key, dimensions=cfg.dimensions)
    elif cfg.provider == "huggingface":
        from ragforge.embeddings.huggingface_embed import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model=cfg.model)
    elif cfg.provider == "ollama":
        from ragforge.embeddings.ollama_embed import OllamaEmbedding
        return OllamaEmbedding(model=cfg.model, base_url=cfg.base_url)
    else:
        raise ValueError(f"Unknown embedding provider: {cfg.provider}")


def _build_llm(cfg: LLMConfig):
    if cfg.provider == "openai":
        from ragforge.llm.openai_llm import OpenAILLM
        return OpenAILLM(model=cfg.model, api_key=cfg.api_key, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    elif cfg.provider == "anthropic":
        from ragforge.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(model=cfg.model, api_key=cfg.api_key, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    elif cfg.provider == "ollama":
        from ragforge.llm.ollama_llm import OllamaLLM
        return OllamaLLM(model=cfg.model, base_url=cfg.base_url, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {cfg.provider}")


def _build_vector_store(cfg: VectorStoreConfig, dimensions: int):
    if cfg.provider == "qdrant":
        from ragforge.vectorstore.qdrant_store import QdrantStore
        return QdrantStore(url=cfg.url, collection=cfg.collection, api_key=cfg.api_key, dimensions=dimensions)
    else:
        raise ValueError(f"Unknown vector store provider: {cfg.provider}")
