"""Embedding backends. Both return L2-normalised float32 matrices, so a dot
product equals cosine similarity."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

import numpy as np

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were what when where which who why with".split()
)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag-of-words + bigram hashing. No model download needed.

    Used for unit tests, CI and offline demos. Use SentenceTransformerEmbedder
    for real retrieval quality.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feat in features:
                h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, so lazy

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vecs, dtype="float32")


class FastEmbedEmbedder:
    """ONNX-based embeddings (fastembed): no PyTorch, so it fits small free hosts.

    Set FASTEMBED_CACHE_PATH so the model is cached on disk (the default is a temp dir).
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # lazy import

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name, cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None)
        self.dim = int(len(next(iter(self._model.embed(["dimension probe"])))))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(list(texts))), dtype="float32")
        return _normalize(vecs)


def get_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder(model_name)
    return SentenceTransformerEmbedder(model_name)
