"""Configuration and settings for the RAG pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
DB_PATH = PROJECT_ROOT / "chatbot.db"
RESOURCES_PATH = DATA_DIR / "resources.json"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Text splitting settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# LLM settings
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# Retrieval settings
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
