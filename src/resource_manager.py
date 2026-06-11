"""Resource metadata manager.

Maintains a JSON file (`resources.json`) that stores metadata about each
indexed document — filename, short description, page/chunk count, etc.
This metadata is used by the LLM judge to understand what documents are
available and make better routing decisions.
"""

import json
from src.config import RESOURCES_PATH


def load_resources() -> dict:
    """Load the resource metadata from disk.

    Returns:
        Dict mapping filename -> resource info dict.
    """
    if not RESOURCES_PATH.exists():
        return {}
    try:
        with open(RESOURCES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_resources(resources: dict) -> None:
    """Save resource metadata to disk."""
    RESOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOURCES_PATH, "w", encoding="utf-8") as f:
        json.dump(resources, f, indent=2, ensure_ascii=False)


def add_resource(
    filename: str,
    description: str = "",
    chunks: int = 0,
    pages: int = 0,
    size_bytes: int = 0,
) -> None:
    """Add or update a resource entry.

    Args:
        filename: Name of the file.
        description: Short description of the document content.
        chunks: Number of chunks after splitting.
        pages: Number of pages (for PDFs) or sections (for TXT).
        size_bytes: File size in bytes.
    """
    resources = load_resources()
    resources[filename] = {
        "filename": filename,
        "description": description[:200],
        "chunks": chunks,
        "pages": pages,
        "size_bytes": size_bytes,
    }
    save_resources(resources)


def get_resource_summary() -> str:
    """Get a human-readable summary of all available resources.

    Used by the LLM judge to understand what documents are indexed.

    Returns:
        Summary string like:
        "Available documents:\n- paper.pdf: Research about CLoRA\n- notes.txt: Meeting notes"
        Or empty string if no resources.
    """
    resources = load_resources()
    if not resources:
        return ""

    lines = ["Available documents in the knowledge base:"]
    for info in resources.values():
        desc = info.get("description", "")
        pages = info.get("pages", 0)
        chunks = info.get("chunks", 0)
        if desc:
            lines.append(f"- {info['filename']}: {desc} ({chunks} chunks, {pages} pages)")
        else:
            lines.append(f"- {info['filename']}: ({chunks} chunks, {pages} pages)")

    return "\n".join(lines)


def clear_resources() -> None:
    """Clear all resource metadata by resetting to an empty dict."""
    save_resources({})


def generate_description(docs_text: str, max_len: int = 100) -> str:
    """Generate a short description from the first chunk of document text.

    Args:
        docs_text: Raw text from the document.
        max_len: Maximum description length.

    Returns:
        Truncated description string.
    """
    text = docs_text.strip()
    if not text:
        return ""

    first_line = text.split("\n")[0].strip()
    if len(first_line) > max_len:
        return first_line[:max_len] + "..."

    if len(first_line) < 20 and len(text) > 20:
        return text[:max_len].strip() + ("..." if len(text) > max_len else "")

    return first_line or text[:max_len]
