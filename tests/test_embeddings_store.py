import tempfile
import unittest

import numpy as np

from roadsafe_rag.embeddings import HashingEmbedder
from roadsafe_rag.vectorstore import NumpyStore, load_store, save_store


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder()

    def test_deterministic_and_normalised(self):
        a = self.emb.encode(["road accidents in Rajasthan"])
        b = self.emb.encode(["road accidents in Rajasthan"])
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a[0])), 1.0, places=5)

    def test_similar_text_scores_higher(self):
        vecs = self.emb.encode(
            ["overspeeding causes road accidents", "overspeeding is a leading cause of accidents", "banana bread recipe"]
        )
        self.assertGreater(float(vecs[0] @ vecs[1]), float(vecs[0] @ vecs[2]))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder()
        self.texts = ["helmet use and two wheelers", "overspeeding and highways", "drunken driving at night"]
        self.store = NumpyStore(self.emb.dim)
        self.store.add(self.emb.encode(self.texts), [{"text": t, "source": "s", "page": i + 1} for i, t in enumerate(self.texts)])

    def test_search_ranks_best_match_first(self):
        hits = self.store.search(self.emb.encode(["overspeeding on highways"])[0], k=3)
        self.assertEqual(hits[0][1]["text"], "overspeeding and highways")
        scores = [s for s, _ in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_k_larger_than_corpus(self):
        self.assertEqual(len(self.store.search(self.emb.encode(["night"])[0], k=50)), 3)

    def test_empty_store(self):
        self.assertEqual(NumpyStore(8).search(np.ones(8, dtype="float32"), k=3), [])

    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add(np.ones((1, 4), dtype="float32"), [{"text": "x"}])

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_store(self.store, tmp, self.emb.name)
            loaded, info = load_store(tmp, prefer_faiss=False)
        self.assertEqual(info["embedder"], self.emb.name)
        q = self.emb.encode(["drunken driving"])[0]
        self.assertEqual(self.store.search(q, 1)[0][1], loaded.search(q, 1)[0][1])


if __name__ == "__main__":
    unittest.main()
