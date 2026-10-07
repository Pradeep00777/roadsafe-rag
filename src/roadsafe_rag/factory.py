"""Assemble a RAGPipeline from settings and an on-disk index."""
from __future__ import annotations

from .config import Settings, get_settings
from .embeddings import get_embedder
from .llm import ExtractiveLLM, GroqLLM
from .rag import RAGPipeline
from .vectorstore import load_store


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    settings = settings or get_settings()
    store, info = load_store(settings.index_dir)
    embedder = get_embedder(settings.embedder, settings.embedding_model)
    if embedder.name != info["embedder"]:
        raise RuntimeError(
            f"Index was built with '{info['embedder']}' but the query embedder is '{embedder.name}'. "
            "Re-run ingestion or fix EMBEDDER / EMBEDDING_MODEL."
        )
    llm = GroqLLM(settings.groq_api_key, settings.groq_model) if settings.groq_api_key else ExtractiveLLM()
    return RAGPipeline(embedder, store, llm, settings.top_k, settings.min_score)
