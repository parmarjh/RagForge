# RAGForge

**Multi-Source RAG SDK for Python** — build retrieval-augmented generation pipelines with pluggable embeddings, LLMs, and vector stores.

## Features

- **Multi-source ingestion**: PDF, DOCX, TXT/MD, web pages, REST APIs, SQL databases
- **Flexible embeddings**: OpenAI, HuggingFace (local), Ollama (local)
- **Multiple LLM backends**: OpenAI GPT, Anthropic Claude, Ollama (local)
- **Qdrant vector store**: Production-grade vector similarity search
- **Configurable chunking**: Recursive, fixed-size, and semantic splitting strategies
- **Optional reranking**: LLM-based result reranking for improved relevance
- **Clean SDK interface**: Single `RAGEngine` entry point or composable pipeline components

## Installation

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

## Quick Start

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

## Advanced Usage

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

## How It Works

RAGForge uses a Multi-Source Retrieval-Augmented Generation algorithm. Diverse documents (PDF, DOCX, web, databases) flow into chunking algorithms, followed by embeddings generation using robust models. This encoded data is stored in the Qdrant vector database. When a user sends a query, it is embedded, mathematically matched for similarity in the vector store, optionally reranked, and fed into an advanced Large Language Model to synthesize an accurate response.

![Working Algorithm 3D](rag_3d_algorithm.png)

## Architecture

```
User Input → Loader → Chunker → Embeddings → Qdrant (store)
                                                  ↓
User Query → Embed Query → Vector Search → [Rerank] → LLM Generate → Answer
```

### God Mode Bar

The **God Mode Bar** is a newly requested, advanced developer toolbar designed to monitor the active RAG pipelines in real-time.
To run and use the God Mode Bar:
```bash
# Enable the experimental feature
export RAGFORGE_GOD_MODE=true

# Or run with the bar directly
python -m ragforge.run_god_mode_bar
```
When enabled, it grants admin-level visualization of embeddings generation latency, DB indexing status, and LLM throughput over live metrics.

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

## Extending

All components use abstract base classes — implement your own providers:

```python
from ragforge import BaseEmbedding, BaseLLM, BaseVectorStore, BaseLoader

class MyCustomEmbedding(BaseEmbedding):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
```

## Future & Additional Work

There are several areas of active development planned for future releases:
- **Streaming LLM Responses**: Native support for token-by-token generation across all providers.
- **Agentic Workflows**: Multi-step reasoning loops integrating tool use and long-term memory.
- **Multimodal RAG**: Expanding the architecture to ingest, embed, and reason over images directly natively.
- **Enhanced God Mode**: Expanded telemetry features in the God Mode Bar, such as dynamic visualization of vectors in PCA/t-SNE clusters.

## License

MIT
