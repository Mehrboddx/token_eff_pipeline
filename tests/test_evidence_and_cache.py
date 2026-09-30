import tempfile
import unittest

from tests import _fake_genai

_fake_genai.install()

from core.memory import Memory  # noqa: E402
from core.tokenWise import CompressionResult  # noqa: E402
from core.units import Unit  # noqa: E402
from eval.evidence import exact_evidence_recall, recent_cutoff, haystack_entries, text_matched_evidence_recall  # noqa: E402
from eval.score_cache import CachedScorer  # noqa: E402


def item_with_sessions():
    return {
        "haystack_session_ids": ["s0", "answer_s1", "s2"],
        "answer_session_ids": ["answer_s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "I like hiking in the mountains every weekend."},
             {"role": "assistant", "content": "Hiking is a great hobby for staying fit."}],
            [{"role": "user", "content": "My new puppy is called Biscuit and she is a beagle.", "has_answer": True},
             {"role": "assistant", "content": "Biscuit is a lovely name for a beagle."}],
            [{"role": "user", "content": "Can you suggest some pasta recipes for dinner tonight?"},
             {"role": "assistant", "content": "Sure, try a simple aglio e olio with garlic."}],
        ],
    }


def seed(item):
    memory = Memory()
    for session_index, session in enumerate(item["haystack_sessions"]):
        for turn in session:
            meta = {"session": session_index, "has_answer": bool(turn.get("has_answer"))}
            if turn["role"] == "user":
                memory.add_user_message(turn["content"], meta=meta)
            else:
                memory.add_model_content(_fake_genai.Content("model", [_fake_genai.Part(turn["content"])]), meta=meta)
    return memory


class ExactRecallTests(unittest.TestCase):
    def compression(self, selected_entries, cutoff):
        units = [Unit(i, "x", "user", entry) for i, entry in enumerate(selected_entries)]
        return {"cutoff_index": cutoff, "result": CompressionResult("x", units, 1), "candidates": 1}

    def test_kept_when_selected(self):
        item = item_with_sessions()
        recall = exact_evidence_recall(seed(item), item, self.compression([2], cutoff=4), "compressed_history")
        self.assertEqual((recall["evidence_turns_kept"], recall["answer_sessions_kept"]), (1, 1))

    def test_missed_when_neither_selected_nor_recent(self):
        item = item_with_sessions()
        recall = exact_evidence_recall(seed(item), item, self.compression([0], cutoff=4), "compressed_history")
        self.assertEqual((recall["evidence_turns"], recall["evidence_turns_kept"], recall["answer_sessions_kept"]), (1, 0, 0))

    def test_recent_window_counts_as_kept(self):
        item = item_with_sessions()
        recall = exact_evidence_recall(seed(item), item, self.compression([0], cutoff=2), "compressed_history")
        self.assertEqual(recall["evidence_turns_kept"], 1)

    def test_full_history_keeps_everything_and_abstractive_is_unmeasured(self):
        item = item_with_sessions()
        memory = seed(item)
        self.assertEqual(exact_evidence_recall(memory, item, None, "full_history")["evidence_turns_kept"], 1)
        abstractive = {"cutoff_index": 4, "result": CompressionResult("x", None, 1), "candidates": 1}
        self.assertIsNone(exact_evidence_recall(memory, item, abstractive, "compressed_history"))


class TextMatchedRecallTests(unittest.TestCase):
    def test_cutoff_counts_the_question_as_a_user_turn(self):
        entries = haystack_entries(item_with_sessions())
        self.assertEqual(recent_cutoff(entries, recent_turns=2), 4)
        self.assertIsNone(recent_cutoff(entries, recent_turns=4))

    def test_shingle_match_survives_labels_tags_and_partial_sentences(self):
        item = item_with_sessions()
        context = "(from 3 days before this question) User: My new puppy is called Biscuit and she"
        recall = text_matched_evidence_recall(item, context, "compressed_history", recent_turns=2)
        self.assertEqual((recall["evidence_turns_kept"], recall["answer_sessions_kept"]), (1, 1))

    def test_unrelated_context_misses(self):
        item = item_with_sessions()
        recall = text_matched_evidence_recall(item, "User: I like hiking in the mountains", "compressed_history", recent_turns=2)
        self.assertEqual(recall["evidence_turns_kept"], 0)


class CachedScorerTests(unittest.TestCase):
    def test_scores_once_then_serves_from_disk_and_misses_on_different_units(self):
        built = []

        class Scorer:
            def __init__(self):
                self.calls = []

            def score_units(self, units, queries):
                self.calls.append(list(queries))
                return [[float(len(query))] * len(units) for query in queries]

        def factory():
            built.append(Scorer())
            return built[-1]

        units = [Unit(0, "alpha", "user", 0, session=0), Unit(1, "beta", "model", 1, session=0)]
        with tempfile.TemporaryDirectory() as directory:
            first = CachedScorer(factory, directory, "k")
            self.assertEqual(first.score_units(units, ["ab"]), [[2.0, 2.0]])
            first.score_units(units, ["ab", "abc"])
            self.assertEqual(built[0].calls, [["ab"], ["abc"]])

            second = CachedScorer(factory, directory, "k")
            self.assertEqual(second.score_units(units, ["abc", "ab"]), [[3.0, 3.0], [2.0, 2.0]])
            self.assertEqual(len(built), 1)

            moved = [Unit(0, "alpha", "user", 0, session=1), units[1]]
            second.score_units(moved, ["ab"])
            self.assertEqual(len(built), 2)

    def test_embeddings_are_stored_for_top_candidates_and_served_from_disk(self):
        import numpy as np

        built = []

        class Scorer:
            def score_units(self, units, queries):
                return [[float(unit.id) for unit in units] for _ in queries]

            def unit_embeddings(self, units, ids):
                self.asked = sorted(ids)
                return {index: np.full(3, float(index)) for index in ids}

        def factory():
            built.append(Scorer())
            return built[-1]

        units = [Unit(i, f"text {i}", "user", i, session=0) for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            first = CachedScorer(factory, directory, "k", embedding_top_k=2)
            first.score_units(units, ["q"])
            vectors = first.unit_embeddings(units, [0])
            self.assertEqual(vectors[0].tolist(), [0.0, 0.0, 0.0])
            self.assertEqual(built[0].asked, [0, 3, 4])  # requested + top-2 by score

            second = CachedScorer(factory, directory, "k", embedding_top_k=2)
            second.score_units(units, ["q"])
            self.assertEqual(second.unit_embeddings(units, [4, 3])[4].tolist(), [4.0, 4.0, 4.0])
            self.assertEqual(len(built), 1)


if __name__ == "__main__":
    unittest.main()
