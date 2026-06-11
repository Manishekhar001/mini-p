# 🤖 RAG Chatbot

A Streamlit-powered chatbot with **Retrieval-Augmented Generation (RAG)**, featuring an **LLM judge** that decides whether to answer from your documents or general knowledge.

Built with **LangGraph**, **Groq**, **FAISS**, and **Nomic Embeddings**.

---

## Architecture

```
User ──→ Streamlit UI (app.py)
              │
              ▼
       LangGraph Pipeline (langgraph_backend.py)
              │
              ├── retrieve_node ──→ FAISS (Nomic embeddings)
              │
              ├── judge_node ─────→ Llama 3.1 (Groq)
              │                           │
              │                    relevance check
              │                           │
              ├── rag_chat_node ←─────────┘ (relevant)
              └── simple_chat_node ←───────┘ (not relevant)
```

### Data Flow

1. **Upload** → PDF/TXT files are chunked and embedded into a FAISS vector index
2. **Chat** → User sends a message → pipeline retrieves relevant chunks from FAISS → judge decides if they're useful → routes to RAG or simple LLM response

---

## Features

- **📄 Multi-format upload** — PDF and TXT files
- **🧠 LLM Judge** — Automatically decides when to use document context vs. plain LLM knowledge. The judge considers a summary of all available documents before making its decision.
- **💬 Chat history** — Persistent threads with SQLite checkpointing; each thread is auto-titled from the first message
- **📚 Expandable citations** — When RAG is used, sources are shown in collapsible sections
- **📂 Resource overview** — Sidebar shows all indexed files with descriptions and chunk counts, plus a **Clear all** button with confirmation to wipe the entire knowledge base and start fresh
- **⚡ Streaming responses** — Real-time token-by-token output

---

## Quick Start

### Prerequisites

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for [Groq](https://console.groq.com) and [Nomic](https://atlas.nomic.ai)

### Setup

```bash
# Clone and enter the project
cd project

# Create virtual environment
uv venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your Groq and Nomic API keys
```

### Run

```bash
streamlit run app.py
```

---

## Configuration

All settings are in `src/config.py` or overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `NOMIC_API_KEY` | — | Nomic API key (required) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM model for chat & judge |
| `EMBEDDING_MODEL` | `nomic-embed-text-v1.5` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size for indexing |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TEMPERATURE` | `0.7` | LLM temperature |
| `MAX_TOKENS` | `4096` | Max tokens per response |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |

---

## Project Structure

```
src/
├── config.py             # Centralized configuration
├── document_loader.py    # PDF/TXT file loading
├── embeddings.py         # Nomic embedding model
├── judge.py              # LLM relevance judge
├── rag_chain.py          # LLM creation + RAG prompt template
├── resource_manager.py   # Resource metadata (resources.json)
├── text_splitter.py      # Document chunking
└── vector_store.py       # FAISS create/save/load/query

langgraph_backend.py      # LangGraph pipeline + thread management
app.py                    # Streamlit UI
```

---

## How the LLM Judge Works

1. **Retrieve** — The user's question fetches the top-4 most similar chunks from FAISS
2. **Judge prompt** — The judge (same Groq LLM) receives:
   - The user's question
   - The retrieved context
   - A summary of all available documents
3. **Decision** — The judge outputs `RELEVANT` or `NOT_RELEVANT` based on whether the context actually helps answer the question
4. **Route** — `RELEVANT` → RAG answer with citations; `NOT_RELEVANT` → plain LLM answer

The resource summary helps the judge understand what documents exist even when retrieved chunks aren't directly relevant, reducing false negatives.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Stateful agent orchestration |
| `langchain-groq` | Groq LLM integration |
| `langchain-nomic` | Nomic embeddings |
| `langchain-community` | Document loaders (PyPDF, Text) |
| `faiss-cpu` | Vector similarity search |
| `streamlit` | Web UI framework |
| `pypdf` | PDF parsing |
| `python-dotenv` | Environment variable loading |
