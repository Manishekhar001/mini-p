"""Embeddings module.

Provides Nomic Embeddings for the RAG pipeline.
"""

from langchain_nomic.embeddings import NomicEmbeddings

from src.config import EMBEDDING_MODEL, NOMIC_API_KEY


def get_embeddings() -> NomicEmbeddings:
    """Get a NomicEmbeddings instance.

    Returns:
        Configured NomicEmbeddings instance.
    """
    return NomicEmbeddings(
        model=EMBEDDING_MODEL,
        nomic_api_key=NOMIC_API_KEY,
    )
