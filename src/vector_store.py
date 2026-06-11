"""FAISS vector store module.

Handles creating, saving, loading, and querying the FAISS vector index.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.config import FAISS_INDEX_PATH


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """Create a FAISS vector store from a list of documents.

    Args:
        documents: List of document chunks to index.
        embeddings: Embeddings model to use for creating vectors.

    Returns:
        FAISS vector store instance.
    """
    return FAISS.from_documents(documents, embeddings)


def save_vector_store(vector_store: FAISS, path: Path = FAISS_INDEX_PATH) -> None:
    """Save a FAISS vector store to disk.

    Args:
        vector_store: The vector store to save.
        path: Path where to save the index file.
    """
    vector_store.save_local(
        str(path.parent),
        index_name=path.stem,
    )


def load_vector_store(
    embeddings: Embeddings,
    path: Path = FAISS_INDEX_PATH,
) -> Optional[FAISS]:
    """Load a FAISS vector store from disk.

    Args:
        embeddings: Embeddings model to use.
        path: Path to the saved index file.

    Returns:
        Loaded FAISS vector store, or None if not found.
    """
    if not path.exists():
        return None

    try:
        return FAISS.load_local(
            str(path.parent),
            embeddings,
            index_name=path.stem,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return None


def add_to_vector_store(
    vector_store: FAISS,
    documents: List[Document],
) -> None:
    """Add documents to an existing FAISS vector store.

    Args:
        vector_store: Existing FAISS vector store.
        documents: New document chunks to add.
    """
    vector_store.add_documents(documents)


def format_retrieved_context(documents: List[Document]) -> str:
    """Format retrieved documents into a context string for the LLM.

    Args:
        documents: Retrieved document chunks.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_str = f" (page {page})" if page != "" else ""
        parts.append(f"[Document {i} from {source}{page_str}]:\n{doc.page_content}")

    return "\n\n".join(parts)
