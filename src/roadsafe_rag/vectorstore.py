"""Vector stores: FAISS when installed, otherwise an exact NumPy fallback.

Both keep vectors + metadata in memory and persist the same on-disk format
(vectors.npy + meta.json), so an index built with one backend loads in the other.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyStore:
    backend = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
        self.metadata: list[dict] = []

    def __len__(self) -> int:
        return len(self.metadata)

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        self.vectors = np.vstack([self.vectors, vectors.astype("float32")])
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, dict]]:
        if not self.metadata or k <= 0:
            return []
        scores = self.vectors @ query.astype("float32")
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.metadata[i]) for i in top]


class FaissStore(NumpyStore):
    backend = "faiss"

    def __init__(self, dim: int) -> None:
        import faiss  # raises ImportError if not installed

        super().__init__(dim)
        self._index = faiss.IndexFlatIP(dim)  # inner product == cosine on unit vectors

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        super().add(vectors, metadata)
        self._index.add(np.ascontiguousarray(vectors, dtype="float32"))

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, dict]]:
        if not self.metadata or k <= 0:
            return []
        k = min(k, len(self.metadata))
        q = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        scores, ids = self._index.search(q, k)
        return [(float(s), self.metadata[i]) for s, i in zip(scores[0], ids[0]) if i != -1]


def create_store(dim: int, prefer_faiss: bool = True) -> NumpyStore:
    if prefer_faiss:
        try:
            return FaissStore(dim)
        except ImportError:
            pass
    return NumpyStore(dim)


def save_store(store: NumpyStore, directory: str | Path, embedder_name: str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", store.vectors)
    info = {"embedder": embedder_name, "dim": store.dim, "chunks": store.metadata}
    (path / "meta.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


def load_store(directory: str | Path, prefer_faiss: bool = True) -> tuple[NumpyStore, dict]:
    path = Path(directory)
    vectors = np.load(path / "vectors.npy")
    info = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    store = create_store(int(info["dim"]), prefer_faiss)
    if len(vectors):
        store.add(vectors, info["chunks"])
    return store, {"embedder": info["embedder"], "dim": info["dim"], "chunks": len(info["chunks"])}
