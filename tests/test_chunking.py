import unittest

from roadsafe_rag.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_size(self):
        text = " ".join(f"Sentence number {i} talks about road safety." for i in range(60))
        chunks = chunk_text(text, size=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(c) <= 200 for c in chunks))

    def test_overlap_repeats_tail_sentence(self):
        text = "Alpha one. Bravo two. Charlie three. Delta four. Echo five. Foxtrot six."
        chunks = chunk_text(text, size=40, overlap=15)
        self.assertGreater(len(chunks), 1)
        shared = [c for a, b in zip(chunks, chunks[1:]) for c in a.split(". ") if c and c in b]
        self.assertTrue(shared, "expected neighbouring chunks to overlap")

    def test_no_text_is_lost(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        joined = " ".join(chunk_text(text, size=30, overlap=0))
        for word in ("First", "Second", "Third"):
            self.assertIn(word, joined)

    def test_very_long_sentence_is_hard_split(self):
        chunks = chunk_text("x" * 1000, size=300, overlap=50)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertEqual(sum(len(c) for c in chunks), 1000)

    def test_empty_text(self):
        self.assertEqual(chunk_text("   ", 100, 10), [])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
