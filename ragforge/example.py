#!/usr/bin/env python3
"""RAGForge — Example: in-memory demo with Qdrant and mock embeddings.

This example demonstrates the full RAG pipeline without requiring
external API keys by using a mock embedding provider and Qdrant's
in-memory mode.
"""

from ragforge.core.config import RAGConfig, EmbeddingConfig, LLMConfig, VectorStoreConfig, ChunkConfig, RetrievalConfig
from ragforge.core.types import Document, Chunk, SourceType
from ragforge.embeddings.base import BaseEmbedding
from ragforge.llm.base import BaseLLM, LLMResponse
from ragforge.vectorstore.qdrant_store import QdrantStore
from ragforge.ingest.chunker import TextChunker
from ragforge.retrieval.retriever import Retriever
from ragforge.pipeline.rag_pipeline import RAGPipeline

import hashlib
import struct


# --- Mock providers for demo (no API keys needed) ---

class MockEmbedding(BaseEmbedding):
    """Deterministic hash-based embeddings for testing."""

    def __init__(self, dims: int = 128):
        self._dims = dims

    def embed(self, text: str) -> list[float]:
        # Generate a deterministic pseudo-embedding from text hash
        h = hashlib.sha256(text.lower().encode()).digest()
        # Repeat hash bytes to fill dimensions
        raw = (h * ((self._dims * 4 // len(h)) + 1))[:self._dims * 4]
        values = list(struct.unpack(f'{self._dims}f', raw[:self._dims * 4]))
        # Normalize
        norm = max(sum(v * v for v in values) ** 0.5, 1e-10)
        return [v / norm for v in values]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims


class MockLLM(BaseLLM):
    """Mock LLM that summarizes context for testing."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        # Extract the question from the prompt
        if "Question:" in prompt:
            question = prompt.split("Question:")[-1].strip().split("\n")[0]
        else:
            question = "the query"

        # Extract context snippets
        sources = []
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith("[Source"):
                continue
            if line and not line.startswith("Context:") and not line.startswith("Answer"):
                if len(line) > 20:
                    sources.append(line[:100])

        answer = f"Based on the provided context, here is what I found about {question}:\n\n"
        if sources:
            for i, s in enumerate(sources[:3]):
                answer += f"- {s}...\n"
        else:
            answer += "No specific details were found in the provided context."

        return LLMResponse(
            text=answer,
            model="mock-llm",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
        )

    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        return self.generate(prompt, system)


def main():
    print("=" * 60)
    print("RAGForge — In-Memory Demo")
    print("=" * 60)

    # 1. Set up components
    embedding = MockEmbedding(dims=128)
    store = QdrantStore(url=":memory:", collection="demo", dimensions=128)
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    llm = MockLLM()

    # 2. Create sample documents
    documents = [
        Document(
            content=(
                "Python is a high-level programming language known for its readability. "
                "It supports multiple paradigms including procedural, object-oriented, "
                "and functional programming. Python uses dynamic typing and garbage collection. "
                "It was created by Guido van Rossum and first released in 1991."
            ),
            source="python-overview.txt",
            source_type=SourceType.FILE,
            metadata={"topic": "python"},
        ),
        Document(
            content=(
                "FastAPI is a modern web framework for building APIs with Python. "
                "It is based on standard Python type hints and provides automatic "
                "documentation, validation, and serialization. FastAPI uses Starlette "
                "for the web parts and Pydantic for data validation. It supports "
                "async/await natively and is one of the fastest Python frameworks."
            ),
            source="fastapi-docs.md",
            source_type=SourceType.FILE,
            metadata={"topic": "fastapi"},
        ),
        Document(
            content=(
                "Qdrant is a vector similarity search engine and database. "
                "It provides a production-ready service with a convenient API to store, "
                "search, and manage vectors with additional payload. Qdrant supports "
                "filtering, is written in Rust, and offers both gRPC and REST interfaces. "
                "It is designed for extended filtering support and neural network matching."
            ),
            source="qdrant-intro.md",
            source_type=SourceType.FILE,
            metadata={"topic": "qdrant"},
        ),
    ]

    # 3. Chunk and embed documents
    print("\n[1] Ingesting documents...")
    all_chunks = []
    for doc in documents:
        chunks = chunker.split(doc)
        all_chunks.extend(chunks)
        print(f"    {doc.source}: {len(chunks)} chunks")

    texts = [c.text for c in all_chunks]
    embeddings = embedding.embed_batch(texts)
    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb

    store.upsert(all_chunks)
    print(f"    Total: {store.count()} chunks in vector store")

    # 4. Build retriever and pipeline
    retriever = Retriever(vector_store=store, embedding=embedding, top_k=3)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    # 5. Run queries
    queries = [
        "What is FastAPI and what makes it fast?",
        "Tell me about Qdrant vector database",
        "Who created Python?",
    ]

    print("\n[2] Running queries...\n")
    for q in queries:
        print(f"Q: {q}")
        result = pipeline.query(q)
        print(f"A: {result.answer}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Sources: {len(result.sources)} chunks")
        for s in result.sources:
            print(f"     - [{s.score:.3f}] {s.chunk.metadata.get('source', 'unknown')}")
        print()

    print("=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
