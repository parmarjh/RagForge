#!/usr/bin/env python3
"""RAGForge — Web Q&A Application.

A Flask-based web interface for the RAG pipeline.
Uses mock providers by default (no API keys needed) with in-memory Qdrant.
Supports: file uploads (PDF, DOCX, TXT, MD), web URL ingestion, text paste.
"""

import hashlib
import struct
import logging
import os
import tempfile
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
            q_words = set(question.lower().split())
            scored = []
            for block in context_blocks:
                block_lower = block.lower()
                score = sum(1 for w in q_words if w in block_lower and len(w) > 2)
                scored.append((score, block))
            scored.sort(key=lambda x: -x[0])

            answer_parts = []
            for score, block in scored[:3]:
                if len(block) > 300:
                    block = block[:297] + "..."
                answer_parts.append(block)

            if answer_parts:
                answer = f"Based on the available knowledge base:\n\n"
                for part in answer_parts:
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


# ── Helper: ingest a Document into the pipeline ─────────────────────

def _ingest_document(doc: Document, source_name: str, source_type_str: str, topic: str = "user-added") -> dict:
    """Chunk, embed, and store a Document. Returns info dict."""
    chunks = chunker.split(doc)
    if not chunks:
        raise ValueError("No chunks produced from the content")

    texts = [c.text for c in chunks]
    embeddings_list = embedding.embed_batch(texts)
    for chunk, emb in zip(chunks, embeddings_list):
        chunk.embedding = emb

    store.upsert(chunks)

    doc_info = {
        "name": source_name,
        "type": source_type_str,
        "chunks": len(chunks),
        "chars": len(doc.content),
        "topic": topic,
    }
    ingested_docs.append(doc_info)
    logger.info(f"Ingested '{source_name}': {len(chunks)} chunks, {len(doc.content)} chars")
    return doc_info


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
        _ingest_document(doc, doc.source, doc.source_type.value, doc.metadata.get("topic", "general"))

    logger.info(f"Seeded {len(demo_documents)} demo documents ({store.count()} chunks)")


# ── Flask App ───────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/readme")
def readme_page():
    return send_from_directory("static", "readme.html")


@app.route("/algorithm-image")
def algorithm_image():
    return send_from_directory(".", "rag_3d_algorithm.png", mimetype="image/png")


# ── Query Endpoint ──────────────────────────────────────────────────

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


# ── Text Ingest Endpoint ────────────────────────────────────────────

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
        info = _ingest_document(doc, source_name, "text")
        return jsonify({
            "message": f"Successfully ingested '{source_name}'",
            "chunks": info["chunks"],
            "chars": info["chars"],
            "total_chunks": store.count(),
        })
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return jsonify({"error": str(e)}), 500


# ── File Upload Endpoint ────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload and ingest a file (PDF, DOCX, TXT, MD, etc.)."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        # Save to temp file for loader-based processing
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, filename)
        file.save(tmp_path)

        text = ""
        file_type = "file"

        if ext == "pdf":
            try:
                from ragforge.ingest.pdf_loader import PDFLoader
                loader = PDFLoader()
                docs = loader.load(tmp_path)
                text = "\n\n".join(d.content for d in docs)
                file_type = "pdf"
            except ImportError:
                return jsonify({"error": "PDF support requires PyMuPDF. Install with: pip install pymupdf"}), 400

        elif ext == "docx":
            try:
                from ragforge.ingest.docx_loader import DocxLoader
                loader = DocxLoader()
                docs = loader.load(tmp_path)
                text = "\n\n".join(d.content for d in docs)
                file_type = "docx"
            except ImportError:
                return jsonify({"error": "DOCX support requires python-docx. Install with: pip install python-docx"}), 400

        elif ext in ("txt", "md", "csv", "json", "py", "js", "html", "css", "log", "yaml", "yml", "xml", "rst", "tsv"):
            from ragforge.ingest.text_loader import TextLoader
            loader = TextLoader()
            docs = loader.load(tmp_path)
            text = "\n\n".join(d.content for d in docs)
            file_type = "text"

        else:
            # Try reading as plain text
            try:
                with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                file_type = "text"
            except Exception:
                return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

        # Clean up temp file
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

        text = text.strip()
        if not text:
            return jsonify({"error": "File is empty or could not be read"}), 400

        doc = Document(
            content=text,
            source=filename,
            source_type=SourceType.FILE,
            metadata={"topic": "user-upload", "file_type": ext},
        )
        info = _ingest_document(doc, filename, file_type, "user-upload")

        return jsonify({
            "message": f"Successfully ingested '{filename}'",
            "chunks": info["chunks"],
            "chars": info["chars"],
            "total_chunks": store.count(),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Web URL Ingest Endpoint ─────────────────────────────────────────

@app.route("/api/ingest-url", methods=["POST"])
def api_ingest_url():
    """Ingest content from a web URL."""
    data = request.json
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        from ragforge.ingest.web_loader import WebLoader
        loader = WebLoader()
        docs = loader.load(url)

        if not docs:
            return jsonify({"error": "Could not extract content from the URL"}), 400

        doc = docs[0]
        title = doc.metadata.get("title", url)
        source_name = title[:50] if len(title) > 50 else title

        info = _ingest_document(doc, source_name, "url", "web")

        return jsonify({
            "message": f"Successfully ingested from URL",
            "source": source_name,
            "url": url,
            "chunks": info["chunks"],
            "chars": info["chars"],
            "total_chunks": store.count(),
        })
    except ImportError:
        return jsonify({"error": "Web loading requires requests and beautifulsoup4. Install with: pip install requests beautifulsoup4"}), 400
    except Exception as e:
        logger.error(f"URL ingest error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Delete & Clear Endpoints ────────────────────────────────────────

@app.route("/api/delete", methods=["POST"])
def api_delete():
    """Delete a document from the knowledge base."""
    data = request.json
    doc_name = data.get("name", "").strip()
    if not doc_name:
        return jsonify({"error": "Document name is required"}), 400

    global ingested_docs
    original_count = len(ingested_docs)
    ingested_docs = [d for d in ingested_docs if d["name"] != doc_name]
    removed = original_count - len(ingested_docs)

    if removed == 0:
        return jsonify({"error": f"Document '{doc_name}' not found"}), 404

    logger.info(f"Removed '{doc_name}' from document list")
    return jsonify({
        "message": f"Removed '{doc_name}'",
        "total_documents": len(ingested_docs),
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear ALL documents from the knowledge base."""
    global ingested_docs, store

    count = len(ingested_docs)
    ingested_docs = []

    # Recreate the vector store (in-memory clear)
    store = QdrantStore(url=":memory:", collection="ragforge_qa", dimensions=EMBED_DIMS)

    # Re-wire the retriever and pipeline
    retriever.vector_store = store
    retriever.embedding = embedding

    logger.info(f"Cleared {count} documents from knowledge base")

    return jsonify({
        "message": f"Cleared {count} documents",
        "total_documents": 0,
        "total_chunks": 0,
    })


# ── Info Endpoints ──────────────────────────────────────────────────

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
