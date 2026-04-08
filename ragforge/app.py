#!/usr/bin/env python3
"""RAGForge — Web Q&A Application.

A Flask-based web interface for the RAG pipeline.
Uses mock providers by default (no API keys needed) with in-memory Qdrant.
"""

import hashlib
import struct
import logging
import os
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

from ragforge.core.types import Document, SourceType
from ragforge.embeddings.base import BaseEmbedding
from ragforge.llm.base import BaseLLM, LLMResponse
from ragforge.vectorstore.qdrant_store import QdrantStore
from ragforge.ingest.chunker import TextChunker
from ragforge.retrieval.retriever import Retriever
from ragforge.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragforge.app")


# ── Mock Providers (no API keys needed) ─────────────────────────────

class MockEmbedding(BaseEmbedding):
    """Deterministic hash-based embeddings for demo/testing."""

    def __init__(self, dims: int = 128):
        self._dims = dims

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).digest()
        raw = (h * ((self._dims * 4 // len(h)) + 1))[: self._dims * 4]
        values = list(struct.unpack(f"{self._dims}f", raw[: self._dims * 4]))
        norm = max(sum(v * v for v in values) ** 0.5, 1e-10)
        return [v / norm for v in values]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims


class SmartMockLLM(BaseLLM):
    """Mock LLM that synthesizes answers from provided context."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        question = ""
        if "Question:" in prompt:
            question = prompt.split("Question:")[-1].strip().split("\n")[0]

        # Extract context blocks from the prompt
        context_blocks = []
        in_source = False
        current_block = ""
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("[Source"):
                if current_block:
                    context_blocks.append(current_block.strip())
                current_block = ""
                in_source = True
                continue
            if stripped == "---":
                if current_block:
                    context_blocks.append(current_block.strip())
                current_block = ""
                in_source = False
                continue
            if in_source and stripped:
                current_block += " " + stripped
        if current_block.strip():
            context_blocks.append(current_block.strip())

        # Build a coherent answer from context
        if context_blocks:
            # Find the most relevant snippets (containing question keywords)
            q_words = set(question.lower().split())
            scored = []
            for block in context_blocks:
                block_lower = block.lower()
                score = sum(1 for w in q_words if w in block_lower and len(w) > 2)
                scored.append((score, block))
            scored.sort(key=lambda x: -x[0])

            answer_parts = []
            for score, block in scored[:3]:
                # Trim to reasonable length
                if len(block) > 300:
                    block = block[:297] + "..."
                answer_parts.append(block)

            if answer_parts:
                answer = f"Based on the available knowledge base:\n\n"
                for i, part in enumerate(answer_parts):
                    answer += f"• {part}\n\n"
            else:
                answer = (
                    "I found some relevant context but couldn't determine a specific "
                    "answer to your question. The context may not contain enough detail."
                )
        else:
            answer = "I couldn't find any relevant information in the knowledge base to answer your question."

        return LLMResponse(
            text=answer,
            model="ragforge-mock-llm",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
        )

    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        return self.generate(prompt, system)


# ── Application State ───────────────────────────────────────────────

EMBED_DIMS = 128

embedding = MockEmbedding(dims=EMBED_DIMS)
store = QdrantStore(url=":memory:", collection="ragforge_qa", dimensions=EMBED_DIMS)
chunker = TextChunker(chunk_size=300, chunk_overlap=40)
llm = SmartMockLLM()
retriever = Retriever(vector_store=store, embedding=embedding, top_k=5)
pipeline = RAGPipeline(retriever=retriever, llm=llm)

# Track ingested documents
ingested_docs: list[dict] = []


# ── Seed some demo data ─────────────────────────────────────────────

def seed_demo_data():
    """Pre-populate the vector store with sample documents."""
    demo_documents = [
        Document(
            content=(
                "Python is a high-level, interpreted programming language known for its "
                "clear syntax and readability. Created by Guido van Rossum and first released "
                "in 1991, Python supports multiple programming paradigms including procedural, "
                "object-oriented, and functional programming. It uses dynamic typing and "
                "automatic memory management through garbage collection. Python's extensive "
                "standard library and large ecosystem of third-party packages make it suitable "
                "for web development, data science, machine learning, automation, and more."
            ),
            source="python-overview.txt",
            source_type=SourceType.FILE,
            metadata={"topic": "python", "category": "programming-language"},
        ),
        Document(
            content=(
                "FastAPI is a modern, high-performance web framework for building APIs with "
                "Python 3.7+ based on standard Python type hints. Key features include automatic "
                "API documentation (Swagger UI / ReDoc), data validation using Pydantic, "
                "dependency injection, and OAuth2 security. FastAPI uses Starlette for web "
                "handling and Pydantic for data serialization. It supports async/await natively "
                "and benchmarks show it is one of the fastest Python web frameworks available, "
                "comparable to Node.js and Go frameworks."
            ),
            source="fastapi-docs.md",
            source_type=SourceType.FILE,
            metadata={"topic": "fastapi", "category": "web-framework"},
        ),
        Document(
            content=(
                "Qdrant is an open-source vector similarity search engine and vector database. "
                "It provides a production-ready service with a convenient API to store, search, "
                "and manage point vectors with additional payload data. Qdrant is written in Rust "
                "for maximum performance and offers both REST and gRPC interfaces. It supports "
                "advanced filtering, horizontal scaling, and on-disk storage. Qdrant is designed "
                "specifically for neural-network matching and semantic search applications."
            ),
            source="qdrant-intro.md",
            source_type=SourceType.FILE,
            metadata={"topic": "qdrant", "category": "vector-database"},
        ),
        Document(
            content=(
                "RAGForge is a multi-source Retrieval-Augmented Generation (RAG) SDK for Python. "
                "It provides a clean interface for building RAG pipelines with pluggable embedding "
                "providers (OpenAI, HuggingFace, Ollama), multiple LLM backends (OpenAI GPT, "
                "Anthropic Claude, Ollama), and Qdrant vector storage. RAGForge supports ingestion "
                "from PDF, DOCX, TXT, web pages, REST APIs, and SQL databases. It includes "
                "configurable text chunking strategies (recursive, fixed-size, semantic) and "
                "optional LLM-based result reranking for improved relevance."
            ),
            source="ragforge-readme.md",
            source_type=SourceType.FILE,
            metadata={"topic": "ragforge", "category": "rag-sdk"},
        ),
        Document(
            content=(
                "Machine Learning is a subset of artificial intelligence that focuses on building "
                "systems that learn from data. Common ML tasks include classification, regression, "
                "clustering, and dimensionality reduction. Popular frameworks include TensorFlow, "
                "PyTorch, scikit-learn, and XGBoost. Deep learning, a subset of ML, uses neural "
                "networks with many layers to model complex patterns. Transfer learning allows "
                "pre-trained models to be fine-tuned on specific tasks, dramatically reducing "
                "training time and data requirements."
            ),
            source="ml-primer.txt",
            source_type=SourceType.FILE,
            metadata={"topic": "machine-learning", "category": "ai"},
        ),
    ]

    for doc in demo_documents:
        chunks = chunker.split(doc)
        if chunks:
            texts = [c.text for c in chunks]
            embeddings_list = embedding.embed_batch(texts)
            for chunk, emb in zip(chunks, embeddings_list):
                chunk.embedding = emb
            store.upsert(chunks)
            ingested_docs.append({
                "name": doc.source,
                "type": doc.source_type.value,
                "chunks": len(chunks),
                "chars": len(doc.content),
                "topic": doc.metadata.get("topic", "general"),
            })

    logger.info(f"Seeded {len(demo_documents)} demo documents ({store.count()} chunks)")


# ── Flask App ───────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/readme")
def readme_page():
    return send_from_directory("static", "readme.html")


@app.route("/algorithm-image")
def algorithm_image():
    return send_from_directory(".", "rag_3d_algorithm.png", mimetype="image/png")


@app.route("/api/query", methods=["POST"])
def api_query():
    """Run a RAG query and return the answer with sources."""
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = pipeline.query(question)
        sources = []
        for s in result.sources:
            sources.append({
                "text": s.chunk.text[:200] + "..." if len(s.chunk.text) > 200 else s.chunk.text,
                "score": round(s.score, 4),
                "source": s.chunk.metadata.get("source", "unknown"),
                "topic": s.chunk.metadata.get("topic", ""),
            })

        return jsonify({
            "answer": result.answer,
            "confidence": round(result.confidence, 4),
            "sources": sources,
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        })
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Ingest text content into the knowledge base."""
    data = request.json
    text = data.get("text", "").strip()
    source_name = data.get("source", "user-input")

    if not text:
        return jsonify({"error": "Text content is required"}), 400

    try:
        doc = Document(
            content=text,
            source=source_name,
            source_type=SourceType.TEXT,
            metadata={"topic": "user-added"},
        )
        chunks = chunker.split(doc)

        if not chunks:
            return jsonify({"error": "No chunks produced from the text"}), 400

        texts = [c.text for c in chunks]
        embeddings_list = embedding.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings_list):
            chunk.embedding = emb

        store.upsert(chunks)

        doc_info = {
            "name": source_name,
            "type": "text",
            "chunks": len(chunks),
            "chars": len(text),
            "topic": "user-added",
        }
        ingested_docs.append(doc_info)

        logger.info(f"Ingested '{source_name}': {len(chunks)} chunks")

        return jsonify({
            "message": f"Successfully ingested '{source_name}'",
            "chunks": len(chunks),
            "total_chunks": store.count(),
        })
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents", methods=["GET"])
def api_documents():
    """List all ingested documents."""
    return jsonify({
        "documents": ingested_docs,
        "total_chunks": store.count(),
    })


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Get knowledge base statistics."""
    return jsonify({
        "total_documents": len(ingested_docs),
        "total_chunks": store.count(),
        "embedding_dims": EMBED_DIMS,
        "model": "ragforge-mock-llm",
        "provider": "in-memory",
    })


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seed_demo_data()
    print("\n" + "=" * 60)
    print("  RAGForge Q&A — http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
