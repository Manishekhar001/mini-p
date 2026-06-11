"""Text splitting module.

Chunks documents into smaller segments for embedding and retrieval.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into smaller chunks.

    Args:
        documents: List of documents to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of split document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)
