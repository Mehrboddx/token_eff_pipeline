import re
import unittest

import torch

from replicate.cpc_compressor import CPCCompressor, _EmbeddingCache


class WordTokenizer:
    """One token per whitespace-delimited word, with real char offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_):
        spans = [match.span() for match in re.finditer(r"\S+", text)]
        encoded = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = spans
        return encoded


def bare_compressor(max_seq_length):
    compressor = CPCCompressor.__new__(CPCCompressor)
    compressor.tokenizer = WordTokenizer()
    compressor.max_seq_length = max_seq_length
    compressor.chunk_safety_margin = 0
    compressor.embedding_cache = _EmbeddingCache(1 << 20)
    return compressor


class SessionChunkingTests(unittest.TestCase):
    def setUp(self):
        self.compressor = bare_compressor(max_seq_length=12)
        self.chunks = []

        def fake_chunk_embeddings(pieces):
            self.chunks.append(pieces)
            return torch.ones(len(pieces), 4)

        self.compressor._chunk_embeddings = fake_chunk_embeddings

    def test_chunks_never_cross_sessions_and_fit_the_budget(self):
        texts = ["one two three", "four five", "six seven eight", "nine ten", "eleven", "twelve thirteen fourteen fifteen"]
        prefixes = ["\nUser: ", " ", "\nAssistant: ", "\nUser: ", " ", "\nAssistant: "]
        groups = [0, 0, 0, 1, 1, 1]

        embeddings = self.compressor.embed_units(texts, prefixes, groups)

        self.assertEqual(tuple(embeddings.shape), (6, 4))
        for chunk in self.chunks:
            self.assertEqual(len({groups[unit] for unit, _, _ in chunk}), 1)
            layout, _ = CPCCompressor._layout_chunk(chunk)
            self.assertLessEqual(len(layout.split()), 12)
        self.assertEqual(sorted(unit for chunk in self.chunks for unit, _, _ in chunk), list(range(6)))

    def test_oversized_unit_is_split_and_pooled_back_into_one_row(self):
        long_text = " ".join(f"w{i}" for i in range(30))
        embeddings = self.compressor.embed_units([long_text], ["\nUser: "], [0])
        pieces = [piece for chunk in self.chunks for piece in chunk]
        self.assertGreater(len(pieces), 1)
        self.assertEqual(" ".join(text for _, _, text in pieces), long_text)
        self.assertAlmostEqual(embeddings[0].norm().item(), 1.0, places=5)


class LayoutTests(unittest.TestCase):
    def test_spans_cover_only_the_sentence_text(self):
        pieces = [(0, "\nUser: ", "Hello there."), (1, " ", "How are you?"), (2, "\nAssistant: ", "Fine.")]
        text, spans = CPCCompressor._layout_chunk(pieces)
        self.assertEqual(text, "User: Hello there. How are you?\nAssistant: Fine.")
        self.assertEqual([text[start:end] for start, end in spans], ["Hello there.", "How are you?", "Fine."])


class EmbeddingCacheTests(unittest.TestCase):
    def test_lru_eviction_by_bytes(self):
        cache = _EmbeddingCache(max_bytes=2 * 4 * 2)  # two float16 tensors of 4 values
        cache.put("a", torch.zeros(4, dtype=torch.float16))
        cache.put("b", torch.zeros(4, dtype=torch.float16))
        cache.get("a")
        cache.put("c", torch.zeros(4, dtype=torch.float16))
        self.assertIsNotNone(cache.get("a"))
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))


if __name__ == "__main__":
    unittest.main()
