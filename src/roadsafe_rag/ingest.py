"""Build the vector index from a PDF.

    python -m roadsafe_rag.ingest --pdf data/raw/road-accidents-in-india-2024.pdf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .chunking import chunk_text
from .config import get_settings
from .embeddings import Embedder, get_embedder
from .vectorstore import NumpyStore, create_store, save_store


def read_pdf_pages(path: str | Path) -> list[str]:
    from pypdf import PdfReader

    return [(page.extract_text() or "") for page in PdfReader(str(path)).pages]


def build_chunks(pages: list[str], source: str, size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for page_no, raw in enumerate(pages, 1):
        text = re.sub(r"[ \t]+", " ", raw).strip()
        if not text:
            continue
        for n, piece in enumerate(chunk_text(text, size, overlap)):
            chunks.append({"id": f"{source}:p{page_no}:c{n}", "source": source, "page": page_no, "text": piece})
    return chunks


def ingest_pages(
    pages: list[str],
    source: str,
    embedder: Embedder,
    size: int = 900,
    overlap: int = 150,
    batch_size: int = 64,
) -> NumpyStore:
    chunks = build_chunks(pages, source, size, overlap)
    store = create_store(embedder.dim)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        store.add(embedder.encode([c["text"] for c in batch]), batch)
    return store


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the RoadSafe RAG index from a PDF.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--index-dir", default=settings.index_dir)
    args = parser.parse_args()

    embedder = get_embedder(settings.embedder, settings.embedding_model)
    pages = read_pdf_pages(args.pdf)
    store = ingest_pages(pages, Path(args.pdf).name, embedder, settings.chunk_size, settings.chunk_overlap)
    save_store(store, args.index_dir, embedder.name)
    print(f"Indexed {len(store)} chunks from {len(pages)} pages -> {args.index_dir} "
          f"(embedder={embedder.name}, backend={store.backend})")


if __name__ == "__main__":
    main()
