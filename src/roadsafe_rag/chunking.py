"""Sentence-aware chunking with overlap."""
from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SPLIT.split(text) if p and p.strip()]


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Pack sentences into chunks of at most `size` characters.

    The tail of each chunk (up to `overlap` characters, whole sentences only) is
    repeated at the start of the next one so answers that span a boundary are
    still retrievable. Sentences longer than `size` are hard-split.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    fresh = False  # does `current` hold sentences not yet emitted?

    def flush() -> None:
        nonlocal current, length, fresh
        if fresh:
            chunks.append(" ".join(current))
        kept: list[str] = []
        kept_len = 0
        for s in reversed(current):
            if kept_len + len(s) + 1 > overlap:
                break
            kept.insert(0, s)
            kept_len += len(s) + 1
        current, length, fresh = kept, kept_len, False

    for sentence in split_sentences(text):
        pieces = (
            [sentence[i:i + size] for i in range(0, len(sentence), size)]
            if len(sentence) > size
            else [sentence]
        )
        for piece in pieces:
            if length + len(piece) + 1 > size and current:
                flush()
                if length + len(piece) + 1 > size:  # overlap leaves no room
                    current, length = [], 0
            current.append(piece)
            length += len(piece) + 1
            fresh = True

    if fresh:
        chunks.append(" ".join(current))
    return chunks
