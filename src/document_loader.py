"""Document loading module.

Supports loading documents from PDF and TXT files.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF file and return a list of Documents (one per page)."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise ImportError(
            "PyPDFLoader is required for PDF loading. "
            "Install it with: pip install pypdf langchain-community"
        )

    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_text(file_path: str | Path) -> List[Document]:
    """Load a plain text file and return a list with a single Document."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(file_path), encoding="utf-8")
    return loader.load()


def load_document(file_path: str | Path) -> List[Document]:
    """Load a document based on its file extension.

    Supported formats: .pdf, .txt
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .txt")
