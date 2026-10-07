"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    embedder: str            # "sentence-transformers" or "hashing" (offline, for tests/demos)
    embedding_model: str
    index_dir: str
    chunk_size: int          # characters
    chunk_overlap: int       # characters
    top_k: int
    min_score: float         # cosine similarity below this => "no relevant context"
    groq_api_key: str | None
    groq_model: str


def get_settings() -> Settings:
    return Settings(
        embedder=os.getenv("EMBEDDER", "sentence-transformers"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        index_dir=os.getenv("INDEX_DIR", "data/index"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "5")),
        min_score=float(os.getenv("MIN_SCORE", "0.30")),
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
