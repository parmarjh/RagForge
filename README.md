# RAGForge

<p align="center">
  <img src="ragforge/rag_3d_algorithm.png" alt="RAGForge 3D Algorithm" width="700">
</p>

<p align="center">
  <strong>Multi-Source RAG SDK for Python</strong> — Build retrieval-augmented generation pipelines<br>
  with pluggable embeddings, LLMs, and vector stores.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/vector_store-Qdrant-purple?style=flat-square" alt="Qdrant">
  <img src="https://img.shields.io/badge/status-Production-brightgreen?style=flat-square" alt="Status">
</p>

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Multi-Source Ingestion** | PDF, DOCX, TXT/MD, web pages, REST APIs, SQL databases |
| **Flexible Embeddings** | OpenAI, HuggingFace (local), Ollama (local) |
| **Multiple LLM Backends** | OpenAI GPT, Anthropic Claude, Ollama (local) |
| **Qdrant Vector Store** | Production-grade vector similarity search |
| **Configurable Chunking** | Recursive, fixed-size, and semantic splitting strategies |
| **Optional Reranking** | LLM-based result reranking for improved relevance |
| **Web Q&A Interface** | Interactive chat-based Q&A with source attribution |
| **Clean SDK Interface** | Single `RAGEngine` entry point or composable pipeline |

---

## 🔬 How It Works — RAG Algorithm

RAGForge implements a **Multi-Source Retrieval-Augmented Generation** pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 PDF  📝 DOCX  📋 TXT  🌐 Web  🔗 API  🗄️ Database        │
│     │       │       │       │      │       │                    │
│     └───────┴───────┴───────┴──────┴───────┘                    │
│                      │                                          │
│              ┌───────▼────────┐                                 │
│              │  Text Chunker  │  (recursive / fixed / semantic) │
│              └───────┬────────┘                                 │
│                      │                                          │
│              ┌───────▼────────┐                                 │
│              │   Embeddings   │  (OpenAI / HuggingFace / Ollama)│
│              └───────┬────────┘                                 │
│                      │                                          │
│              ┌───────▼────────┐                                 │
│              │  Qdrant Store  │  Vector Database                │
│              └────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  👤 User Query                                                  │
│       │                                                         │
│  ┌────▼─────┐    ┌──────────────┐    ┌──────────┐              │
│  │  Embed   │───▶│ Vector Search│───▶│ Rerank   │ (optional)   │
│  │  Query   │    │  (Qdrant)    │    │ (LLM)    │              │
│  └──────────┘    └──────────────┘    └────┬─────┘              │
│                                           │                     │
│                                    ┌──────▼─────┐              │
│                                    │ LLM Generate│              │
│                                    │  (GPT/Claude)│             │
│                                    └──────┬──────┘              │
│                                           │                     │
│                                    ┌──────▼──────┐             │
│                                    │   Answer +   │             │
│                                    │   Sources    │             │
│                                    └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Algorithm Steps

1. **Document Ingestion** — Load content from multiple sources (PDF, DOCX, TXT, web, API, database)
2. **Text Chunking** — Split documents into optimal-sized chunks using recursive, fixed, or semantic strategies with configurable overlap
3. **Embedding Generation** — Convert text chunks into high-dimensional vectors using embedding models (OpenAI, HuggingFace, Ollama)
4. **Vector Storage** — Store embeddings in Qdrant vector database with metadata payload
5. **Query Embedding** — When user asks a question, embed the query using the same model
6. **Similarity Search** — Find the most relevant chunks via cosine similarity in vector space
7. **Reranking** *(optional)* — Use LLM to re-score results for improved relevance
8. **Answer Generation** — Feed retrieved context + question to LLM for synthesized answer with source attribution

---

## 📦 Installation

```bash
# Core (Qdrant + requests)
pip install ragforge

# With specific providers
pip install ragforge[openai]           # OpenAI embeddings + LLM
pip install ragforge[anthropic]        # Anthropic Claude LLM
pip install ragforge[huggingface]      # Local sentence-transformers
pip install ragforge[pdf,docx,web]     # Document loaders
pip install ragforge[all]              # Everything
```

### From Source

```bash
git clone https://github.com/parmarjh/RagForge.git
cd RagForge/ragforge
pip install -e .
```

---

## ⚡ Quick Start

### SDK Usage

```python
from ragforge import RAGEngine

engine = RAGEngine(
    embedding_provider="openai",        # or "huggingface", "ollama"
    llm_provider="anthropic",           # or "openai", "ollama"
    qdrant_url="http://localhost:6333",
)

# Ingest from multiple sources
engine.ingest_file("docs/manual.pdf")
engine.ingest_file("notes.docx")
engine.ingest_url("https://docs.python.org/3/tutorial/")
engine.ingest_directory("./knowledge_base/", glob="**/*.md")
engine.ingest_text("Inline text content to index")

# Query
result = engine.query("How do I handle exceptions in Python?")
print(result.answer)
print(result.sources)       # source chunks with metadata
print(result.confidence)    # retrieval relevance score
```

### Web Q&A Interface

Run the interactive web-based Q&A (no API keys needed — uses mock providers):

```bash
cd ragforge
python app.py
# Open http://localhost:5000
```

---

## 🏗️ Architecture

### Package Structure

| Module | Purpose |
|--------|---------|
| `ragforge.core` | Engine orchestrator, config, shared types |
| `ragforge.ingest` | Document loaders (PDF, DOCX, text, web, API, DB) + chunker |
| `ragforge.embeddings` | Embedding providers (OpenAI, HuggingFace, Ollama) |
| `ragforge.vectorstore` | Vector store implementations (Qdrant) |
| `ragforge.llm` | LLM providers (OpenAI, Anthropic, Ollama) |
| `ragforge.retrieval` | Retriever and reranker |
| `ragforge.pipeline` | RAG pipeline orchestration and prompt templates |

### Directory Structure

```
RagForge/
├── README.md                    # This file
├── index.html                   # Documentation page
├── ragforge/
│   ├── README.md                # SDK documentation
│   ├── pyproject.toml           # Package config
│   ├── requirements.txt         # Dependencies
│   ├── example.py               # In-memory demo (no API keys)
│   ├── app.py                   # Web Q&A server (Flask)
│   ├── rag_3d_algorithm.png     # Algorithm visualization
│   ├── static/
│   │   └── index.html           # Web Q&A interface
│   └── ragforge/                # SDK source code
│       ├── __init__.py
│       ├── core/                # Engine, config, types
│       ├── ingest/              # Loaders + chunker
│       ├── embeddings/          # Embedding providers
│       ├── vectorstore/         # Qdrant store
│       ├── llm/                 # LLM providers
│       ├── retrieval/           # Retriever + reranker
│       └── pipeline/            # RAG pipeline + prompts
```

---

## 🔧 Advanced Usage

### Composable Pipeline

```python
from ragforge import (
    RAGPipeline, QdrantStore, OpenAIEmbedding, AnthropicLLM,
    Retriever, TextChunker
)

# Build components individually
embedding = OpenAIEmbedding(model="text-embedding-3-small")
store = QdrantStore(url="http://localhost:6333", collection="my_docs", dimensions=1536)
llm = AnthropicLLM(model="claude-sonnet-4-20250514")

# Retriever
retriever = Retriever(vector_store=store, embedding=embedding, top_k=5)

# Pipeline
pipeline = RAGPipeline(retriever=retriever, llm=llm, rerank=True)
result = pipeline.query("What is the deployment process?")
```

### Database Ingestion

```python
engine.ingest_database(
    "postgresql://user:pass@localhost/mydb",
    query="SELECT content, title FROM articles WHERE published = true",
    text_column="content",
)
```

### API Ingestion

```python
engine.ingest_api(
    "https://api.example.com/v1/articles",
    headers={"Authorization": "Bearer xxx"},
    json_path="data.results",
    text_field="body",
)
```

### Custom Configuration

```python
from ragforge import RAGEngine, RAGConfig, EmbeddingConfig, LLMConfig

config = RAGConfig(
    embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-large", dimensions=3072),
    llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.0),
)
engine = RAGEngine(config=config)
```

---

## 🧩 Extending

All components use abstract base classes — implement your own providers:

```python
from ragforge import BaseEmbedding, BaseLLM, BaseVectorStore, BaseLoader

class MyCustomEmbedding(BaseEmbedding):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...
```

---

## 🗺️ Roadmap

- **Streaming LLM Responses** — Native token-by-token generation across all providers
- **Agentic Workflows** — Multi-step reasoning loops with tool use and long-term memory
- **Multimodal RAG** — Ingest, embed, and reason over images natively
- **Enhanced God Mode** — Dynamic vector visualization with PCA/t-SNE clusters

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/parmarjh">parmarjh</a>
</p>
