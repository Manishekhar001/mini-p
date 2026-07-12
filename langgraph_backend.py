"""LangGraph backend for the RAG pipeline.

Orchestrates the RAG workflow with:
- Groq LLM (via langchain-groq)
- FAISS vector store with Nomic Embeddings
- LLM Judge that decides whether to use RAG or plain chat
- SQLite checkpoint-based thread memory
- Thread title management (first message as title)
- Resource metadata tracking for the LLM judge
"""

import re
import shutil
import sqlite3
import traceback
import uuid
from typing import Annotated, Generator, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

from src.config import DB_PATH, UPLOAD_DIR, FAISS_INDEX_DIR
from src.embeddings import get_embeddings
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    format_retrieved_context,
    add_to_vector_store,
)
from src.rag_chain import create_llm, RAG_SYSTEM_TEMPLATE
from src.judge import judge_relevance
from src.text_splitter import split_documents
from src.document_loader import load_document
from src.resource_manager import (
    add_resource,
    clear_resources,
    get_resource_summary,
    generate_description,
)


# ---------- State Definition ----------

class ChatState(TypedDict):
    """State for the LangGraph RAG chatbot with LLM judge."""
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: str
    use_rag: bool


# ---------- Database Setup ----------

def get_db_connection():
    """Get a connection to the SQLite database for thread metadata."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


metadata_conn = get_db_connection()


# ---------- Vector Store Manager ----------

class VectorStoreManager:
    """Manages the FAISS vector store lifecycle and resource metadata."""

    def __init__(self):
        self._embeddings = None
        self._vector_store = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = load_vector_store(self.embeddings)
        return self._vector_store

    def has_index(self) -> bool:
        return self.vector_store is not None

    def index_documents(self, documents: list[Document]) -> int:
        """Index documents into the FAISS vector store.

        Args:
            documents: Documents to index.

        Returns:
            Number of chunks indexed.
        """
        chunks = split_documents(documents)
        if not chunks:
            return 0

        if self._vector_store is not None:
            add_to_vector_store(self._vector_store, chunks)
        else:
            self._vector_store = create_vector_store(chunks, self.embeddings)

        save_vector_store(self._vector_store)
        return len(chunks)

    def index_file(self, filepath) -> int:
        """Load, split, index a file and store its resource metadata.

        Args:
            filepath: Path to the file to index.

        Returns:
            Number of chunks indexed, 0 on failure.
        """
        docs = load_document(filepath)
        if not docs:
            return 0

        chunks = self.index_documents(docs)
        if chunks == 0:
            return 0

        pages = len(docs)
        raw_text = "\n".join(d.page_content for d in docs)
        desc = generate_description(raw_text)
        size_bytes = filepath.stat().st_size

        add_resource(
            filename=filepath.name,
            description=desc,
            chunks=chunks,
            pages=pages,
            size_bytes=size_bytes,
        )
        return chunks

    def clear_all(self) -> None:
        """Clear all indexed resources: FAISS index, uploads, and metadata."""
        self._vector_store = None

        # Wipe the FAISS index directory
        if FAISS_INDEX_DIR.exists():
            shutil.rmtree(FAISS_INDEX_DIR)
        FAISS_INDEX_DIR.mkdir(parents=True)

        # Wipe uploaded files
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file():
                    f.unlink()

        # Clear resource metadata
        clear_resources()

    def retrieve(self, query: str, k: int = 4) -> str:
        """Retrieve and format context for a query."""
        if not self.has_index():
            return ""
        docs = self._vector_store.similarity_search(query, k=k)
        return format_retrieved_context(docs)


vector_store_manager = VectorStoreManager()


# ---------- Graph Nodes ----------

def retrieve_node(state: ChatState) -> dict:
    """Retrieve relevant context from FAISS (if index exists)."""
    messages = state["messages"]
    if not messages:
        return {"retrieved_context": ""}

    user_message = messages[-1].content
    context = vector_store_manager.retrieve(user_message)
    return {"retrieved_context": context}


def judge_node(state: ChatState) -> dict:
    """Judge whether the retrieved context is relevant to the user's query."""
    messages = state["messages"]
    if not messages:
        return {"use_rag": False}

    user_message = messages[-1].content
    context = state.get("retrieved_context", "")
    resource_summary = get_resource_summary()
    use_rag = judge_relevance(user_message, context, resource_summary=resource_summary)
    return {"use_rag": use_rag}


def rag_chat_node(state: ChatState) -> Generator[dict, None, None]:
    """Generate RAG response with token-level streaming from the LLM.

    Yields each token chunk as it arrives so the frontend can display
    incremental output instead of waiting for the full response.

    Uses AIMessage (not AIMessageChunk) with a fixed id so the SQLite
    checkpointer can cleanly serialize/deserialize the state. Without
    this, chatbot.get_state() would fail and no previous threads would
    load.
    """
    messages = state["messages"]
    if not messages:
        return

    context = state.get("retrieved_context", "")
    llm = create_llm()

    system_message = SystemMessage(content=RAG_SYSTEM_TEMPLATE.format(context=context))
    full_messages = [system_message] + list(messages)

    full_content = ""
    response_id = None
    for chunk in llm.stream(full_messages):
        if chunk.content:
            if response_id is None:
                response_id = chunk.id
            full_content += chunk.content
            yield {"messages": [AIMessage(content=full_content, id=response_id)]}


def simple_chat_node(state: ChatState) -> Generator[dict, None, None]:
    """Generate a plain LLM response with token-level streaming.

    Yields each token chunk as it arrives so the frontend can display
    incremental output instead of waiting for the full response.

    Uses AIMessage (not AIMessageChunk) with a fixed id so the SQLite
    checkpointer can cleanly serialize/deserialize the state.
    """
    messages = state["messages"]
    if not messages:
        return

    llm = create_llm()

    full_content = ""
    response_id = None
    for chunk in llm.stream(messages):
        if chunk.content:
            if response_id is None:
                response_id = chunk.id
            full_content += chunk.content
            yield {"messages": [AIMessage(content=full_content, id=response_id)]}


# ---------- Conditional Routing ----------

def route_after_judge(state: ChatState) -> Literal["rag_chat_node", "simple_chat_node"]:
    """Route to RAG or simple chat based on the judge's decision."""
    if state.get("use_rag", False):
        return "rag_chat_node"
    return "simple_chat_node"


# Simple greeting patterns — common casual openers that never need RAG
_GREETING_RE = re.compile(
    r"^"
    r"(hi+|hello+|hey|howdy|greetings|sup|yo"
    r"|what'?s\s+up|how'?s\s+(it\s+going|everything|life|you|your\s+day)"
    r"|good\s+(morning|afternoon|evening|day)"
    r"|nice\s+to\s+(meet|see)\s+you"
    r"|(i'?m|i\s+am)\s+(good|fine|great|doing\s+well)"
    r"|how\s+(are|'re)\s+you"
    r"|long\s+time\s+no\s+see"
    r"|cheers|hiya|heya|'sup|was\s+sup"
    r")"
    r"\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    """Check whether a message looks like casual chat, not a document query."""
    return bool(_GREETING_RE.match(text.strip()))


def route_from_start(state: ChatState) -> Literal["retrieve_node", "simple_chat_node"]:
    """Route to RAG pipeline or plain chat.

    Skips retrieval entirely when:
    - No documents are indexed, OR
    - The message is a simple greeting / casual chat

    This avoids paying for an embedding-API call and a judge LLM call on
    every single message when there is nothing to retrieve from.
    """
    if not vector_store_manager.has_index():
        return "simple_chat_node"

    messages = state["messages"]
    if messages and is_greeting(messages[-1].content):
        return "simple_chat_node"

    return "retrieve_node"


# ---------- Graph Compilation ----------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph with judge routing."""
    graph = StateGraph(ChatState)

    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("judge_node", judge_node)
    graph.add_node("rag_chat_node", rag_chat_node)
    graph.add_node("simple_chat_node", simple_chat_node)

    graph.add_conditional_edges(
        START,
        route_from_start,
        {"retrieve_node": "retrieve_node", "simple_chat_node": "simple_chat_node"},
    )
    graph.add_edge("retrieve_node", "judge_node")
    graph.add_conditional_edges(
        "judge_node",
        route_after_judge,
        {"rag_chat_node": "rag_chat_node", "simple_chat_node": "simple_chat_node"},
    )
    graph.add_edge("rag_chat_node", END)
    graph.add_edge("simple_chat_node", END)

    return graph


checkpointer = SqliteSaver(
    conn=sqlite3.connect(str(DB_PATH), check_same_thread=False)
)

chatbot = build_graph().compile(checkpointer=checkpointer)


# ---------- Thread Management ----------

def generate_thread_id() -> str:
    """Generate a unique thread ID."""
    return str(uuid.uuid4())


def add_thread(thread_id: str, title: str = "New Chat") -> None:
    """Add a thread to the metadata table."""
    cursor = metadata_conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO threads (thread_id, title) VALUES (?, ?)",
        (thread_id, title[:100]),
    )
    metadata_conn.commit()


def update_thread_title(thread_id: str, title: str) -> None:
    """Update the title of a thread."""
    cursor = metadata_conn.cursor()
    cursor.execute(
        "UPDATE threads SET title = ? WHERE thread_id = ?",
        (title[:100], thread_id),
    )
    metadata_conn.commit()


def delete_thread(thread_id: str) -> None:
    """Delete a thread and its checkpoints."""
    cursor = metadata_conn.cursor()
    cursor.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
    cursor.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
    )
    cursor.execute(
        "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
    )
    metadata_conn.commit()


def get_all_threads() -> list[dict]:
    """Get all threads ordered by most recent first."""
    cursor = metadata_conn.cursor()
    cursor.execute(
        "SELECT thread_id, title FROM threads ORDER BY created_at DESC"
    )
    return [
        {"thread_id": row[0], "title": row[1]}
        for row in cursor.fetchall()
    ]


def load_conversation(thread_id: str) -> list:
    """Load conversation messages from a thread checkpoint."""
    try:
        state = chatbot.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if state and state.values and "messages" in state.values:
            return state.values["messages"]
    except Exception as e:
        print(f"[load_conversation] get_state failed for {thread_id}: {e}")
        traceback.print_exc()
    return []


def clean_title(text: str, max_length: int = 30) -> str:
    """Create a clean title from the first message."""
    return text.strip().replace("\n", " ")[:max_length]
