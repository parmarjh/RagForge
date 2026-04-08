# 🚀 RAGForge: Zero to End Comprehensive Guide

**Multi-Source RAG SDK for Python** — The ultimate framework for building retrieval-augmented generation pipelines. Whether you are a beginner looking to understand RAG from zero, or an advanced developer needing deeply customizable pipelines, RAGForge provides the tools needed.

---

## 📖 Table of Contents

1. [Understanding RAG (From Zero)](#understanding-rag-from-zero)
2. [What is RAGForge?](#what-is-ragforge)
3. [Deep Installation Guide](#deep-installation-guide)
4. [Working with the Web Application](#working-with-the-web-application)
5. [Quick Start: High-Level SDK](#quick-start-high-level-sdk)
6. [Deep Dive: Composable Pipelines](#deep-dive-composable-pipelines)
7. [Advanced Ingestion Sources](#advanced-ingestion-sources)
8. [Extending RAGForge](#extending-ragforge)
9. [Architecture & Internals](#architecture--internals)
10. [Roadmap](#roadmap)

---

## 🧠 Understanding RAG (From Zero)

If you are starting from zero, **Retrieval-Augmented Generation (RAG)** is a technique to give AI models (like ChatGPT) the ability to "read" your private documents. 
1. **Ingestion:** You take your documents (PDFs, text) and split them into smaller "chunks". 
2. **Embedding:** You run these chunks through a model that turns the text into mathematical vectors.
3. **Retrieval:** When a user asks a question, it is also turned into a vector to find the most mathematically similar document chunks.
4. **Generation:** An LLM reads those retrieved chunks and answers the question accurately based on *your* data, eliminating hallucinations.

## ⚙️ What is RAGForge?

RAGForge takes this paradigm and wraps it in a production-ready, highly composable Python SDK.

**Key Capabilities:**
- **Multi-source ingestion**: PDF, DOCX, TXT/MD, web pages, REST APIs, SQL databases
- **Flexible embeddings**: OpenAI, HuggingFace (local), Ollama (local)
- **Multiple LLM backends**: OpenAI GPT, Anthropic Claude, Ollama (local)
- **Qdrant vector store**: Production-grade vector similarity search
- **Configurable chunking**: Recursive, fixed-size, and semantic splitting strategies

---

## 🛠️ Deep Installation Guide

To get started, we recommend creating a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install RAGForge based on your exact needs. By default, only the core logic and Qdrant client are installed.

```bash
# Core installation for basic usage
pip install ragforge

# Installation with OpenAI support
pip install ragforge[openai]

# Installation for completely local AI (Ollama & HuggingFace)
pip install ragforge[huggingface]

# Complete installation with all document loaders (PDFs, DOCX) and providers
pip install ragforge[all]
```

---

## 🌐 Working with the Web Application

RAGForge includes a robust, visually appealing Flask-based web interface. The local server comes pre-loaded with mock data and a demonstration environment so you can use it immediately without needing any API keys. This is the fastest way to physically see RAG working.

### Starting the Server

```bash
# Ensure you have Flask installed, then run the app.py file:
pip install flask
python app.py
```

Once started, the server will be available at:
`http://localhost:5000`

### Interface Features

From the web dashboard, you can interact with the RAG engine seamlessly:
1. **Ingest Documents:** Upload `PDF`, `DOCX`, `TXT`, or `MD` files via drag-and-drop.
2. **URL Ingestion:** Paste web URLs to dynamically scrape and add remote content to the knowledge base.
3. **RAG Q&A:** Send contextual queries to test document vector retrieval against the synthetic mock LLM. The interface will highlight exactly which source chunks were used to generate the answer.

---

## ⚡ Quick Start: High-Level SDK

For most users, the `RAGEngine` class provides a simple, all-in-one controller to build your pipeline.

```python
from ragforge import RAGEngine

# 1. Initialize the Engine
engine = RAGEngine(
    embedding_provider="openai",        # Options: "huggingface", "ollama"
    llm_provider="anthropic",           # Options: "openai", "ollama"
    qdrant_url="http://localhost:6333", # URL to your Qdrant instance
)

# 2. Ingest your Data deeply
engine.ingest_file("docs/manual.pdf")
engine.ingest_directory("./knowledge_base/", glob="**/*.md")
engine.ingest_url("https://docs.python.org/3/tutorial/")

# 3. Query the Engine
result = engine.query("How do I handle exceptions in Python?")

print(f"Answer: {result.answer}")
print(f"Confidence Score: {result.confidence}")
for source in result.sources:
    print(f"Source Document: {source.metadata['source']}")
```

---

## 🔬 Deep Dive: Composable Pipelines

To go completely from "zero to end", advanced users will want to construct the custom pipeline components directly, bypassing the `RAGEngine` abstraction.

### 1. Vector Store Setup
```python
from ragforge import QdrantStore
store = QdrantStore(url="http://localhost:6333", collection="enterprise_docs", dimensions=1536)
```

### 2. Provider Configuration
```python
from ragforge import OpenAIEmbedding, AnthropicLLM
embedding = OpenAIEmbedding(model="text-embedding-3-small")
llm = AnthropicLLM(model="claude-3-5-sonnet-20240620")
```

### 3. Assembling the Retriever and Pipeline
```python
from ragforge import Retriever, RAGPipeline

# Create a retrieval layer that pulls the Top 5 documents
retriever = Retriever(vector_store=store, embedding=embedding, top_k=5)

# Assemble the full sequence
pipeline = RAGPipeline(retriever=retriever, llm=llm, rerank=True)

# Execute
response = pipeline.query("Explain advanced composable pipelines.")
```

---

## 🗄️ Advanced Ingestion Sources

Beyond basic files, RAGForge can deeply integrate with structural enterprise data.

### Relational Database SQL Ingestion
Ingest specific rows and textual columns from an active database.
```python
engine.ingest_database(
    dsn="postgresql://user:pass@localhost/mydb",
    query="SELECT content, title, updated_at FROM technical_articles WHERE published = true",
    text_column="content" # Which column acts as the actual text chunk
)
```

### REST API Integration
Pull documentation live from external systems via custom JSON traversal logic.
```python
engine.ingest_api(
    endpoint="https://api.example.com/v1/articles",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json_path="data.results", # Path inside the JSON payload to the list of objects
    text_field="body",        # The key containing the text to index
)
```

---

## 🧩 Extending RAGForge

If you are using proprietary internal models, RAGForge relies on Abstract Base Classes (ABCs), making writing plugins extremely simple.

### Custom LLM Implementation
```python
from ragforge import BaseLLM, LLMResponse

class MyCompanyLLM(BaseLLM):
    def __init__(self, api_endpoint: str):
        self.endpoint = api_endpoint

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        # Implement your custom API requests here
        answer_text = "Generated text from custom model"
        return LLMResponse(
            text=answer_text,
            model="custom-corporate-llm",
            prompt_tokens=100,
            completion_tokens=50
        )
```
Pass `llm=MyCompanyLLM(...)` into your `RAGPipeline` directly.

---

## 🏛️ Architecture & Internals

RAGForge uses an optimized mathematical architecture for semantic similarity:

```
User Input → Loader → Chunker → Embeddings → Qdrant (store)
                                                  ↓
User Query → Embed Query → Vector Search → [Rerank] → LLM Generate → Answer
```

![Working Algorithm 3D](rag_3d_algorithm.png)

### The God Mode Bar
The **God Mode Bar** is an advanced developer toolbar designed to dynamically monitor the active RAG pipelines in real-time. Enable it for deep profiling:
```bash
export RAGFORGE_GOD_MODE=true
python -m ragforge.run_god_mode_bar
```
It grants administrative visualization mapping latency profiles, embeddings generation statistics, DB indexing throughput, and token burn rates.

---

## 🚀 Roadmap

We are constantly pushing RAGForge further toward the "end":
- **Streaming LLM Responses**: Native generator support for token-by-token UI updates.
- **Agentic Loops (ReACT)**: Multi-step reasoning where the system can query, reflect, and query again before answering.
- **Multimodal Visual RAG**: Using CLIP/Multi-modal models to store and search across image content natively.

## 📄 License
MIT License.
