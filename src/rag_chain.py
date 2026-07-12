"""RAG retrieval + generation chain.

Combines document retrieval and LLM generation using Ollama (local).
"""

from langchain_ollama import ChatOllama

from src.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, TEMPERATURE, MAX_TOKENS

# RAG system prompt template. Kept as a plain string (not a ChatPromptTemplate)
# because rag_chat_node needs to splice this in front of the FULL conversation
# history, not just the latest question -- see langgraph_backend.rag_chat_node.
RAG_SYSTEM_TEMPLATE = (
    "You are a helpful AI assistant. Answer the user's question based on the "
    "provided context. If the context doesn't contain enough information, "
    "say so and use your general knowledge to help.\n\n"
    "Context:\n{context}"
)


_llm_instance: ChatOllama | None = None


def create_llm() -> ChatOllama:
    """Get a cached Ollama LLM instance (created once, reused after that)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
        )
    return _llm_instance
