"""Embeddings module.

Provides local embeddings via Ollama for the RAG pipeline.
"""

from langchain_ollama import OllamaEmbeddings

from src.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def get_embeddings() -> OllamaEmbeddings:
    """Get an OllamaEmbeddings instance.

    Returns:
        Configured OllamaEmbeddings instance.
    """
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
