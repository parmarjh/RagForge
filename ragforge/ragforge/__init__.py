"""
RAGForge — Multi-Source RAG SDK for Python.

A production-grade library for building Retrieval-Augmented Generation
pipelines with support for multiple document sources, embedding providers,
vector stores, and LLM backends.

Quick start::

    from ragforge import RAGEngine

    engine = RAGEngine(
        embedding_provider="openai",
        llm_provider="anthropic",
        qdrant_url="http://localhost:6333",
    )
    engine.ingest_file("docs/manual.pdf")
    result = engine.query("How does X work?")
    print(result.answer)
"""

__version__ = "0.1.0"

from ragforge.core.config import (
    ChunkConfig,
    EmbeddingConfig,
    LLMConfig,
    RAGConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from ragforge.core.engine import RAGEngine
from ragforge.core.types import (
    Chunk,
    ChunkStrategy,
    Document,
    GenerationResult,
    SearchResult,
    SourceType,
)
from ragforge.embeddings.base import BaseEmbedding
from ragforge.embeddings.huggingface_embed import HuggingFaceEmbedding
from ragforge.embeddings.ollama_embed import OllamaEmbedding
from ragforge.embeddings.openai_embed import OpenAIEmbedding
from ragforge.ingest.base import BaseLoader
from ragforge.ingest.chunker import TextChunker
from ragforge.llm.anthropic_llm import AnthropicLLM
from ragforge.llm.base import BaseLLM, LLMResponse
from ragforge.llm.ollama_llm import OllamaLLM
from ragforge.llm.openai_llm import OpenAILLM
from ragforge.pipeline.rag_pipeline import RAGPipeline
from ragforge.retrieval.retriever import Retriever
from ragforge.vectorstore.base import BaseVectorStore
from ragforge.vectorstore.qdrant_store import QdrantStore

__all__ = [
    # Engine
    "RAGEngine",
    # Config
    "RAGConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "VectorStoreConfig",
    "ChunkConfig",
    "RetrievalConfig",
    # Types
    "Document",
    "Chunk",
    "SearchResult",
    "GenerationResult",
    "SourceType",
    "ChunkStrategy",
    # Embeddings
    "BaseEmbedding",
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "OllamaEmbedding",
    # Vector Store
    "BaseVectorStore",
    "QdrantStore",
    # LLM
    "BaseLLM",
    "LLMResponse",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    # Pipeline
    "RAGPipeline",
    "Retriever",
    # Ingest
    "BaseLoader",
    "TextChunker",
]
