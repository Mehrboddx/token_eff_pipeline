import unittest

from tests import _fake_genai

_fake_genai.install()

from core.memory import Memory  # noqa: E402
from core.scorers import BM25Scorer, LexicalScorer  # noqa: E402
from core.selection import SelectionConfig, fuse_scores, select_units  # noqa: E402
from core.tokenWise import TokenWise  # noqa: E402
from core.units import Unit, prepare_units, render_units  # noqa: E402


def count_words(text):
    return len(text.split())


def entry(text, entry_index, role="user", session=None, time=None):
    return {"role": role, "text": text, "entry_index": entry_index, "meta": {"session": session, "time": time}}


def make_units(texts, sessions=None, roles=None, entries=None):
    units = []
    for index, text in enumerate(texts):
        units.append(Unit(
            id=index,
            text=text,
            role=(roles or ["user"] * len(texts))[index],
            entry_index=(entries or list(range(len(texts))))[index],
            session=(sessions or [None] * len(texts))[index],
            tokens=count_words(text),
        ))
    return units


class MemorySplittingTests(unittest.TestCase):
    def test_prefix_is_shown_to_the_model_but_never_scored(self):
        memory = Memory()
        memory.add_user_message("I went to MoMA. It was great!", meta={"session": 0}, prefix="(from 3 days before this question) ")

        self.assertEqual(memory.get_history()[0].parts[0].text, "(from 3 days before this question) I went to MoMA. It was great!")
        self.assertEqual(memory.get_sentences(), ["I went to MoMA.", "It was great!"])
        self.assertEqual(memory.get_sentence_entries()[0]["meta"], {"session": 0})

    def test_copy_is_independent(self):
        memory = Memory()
        memory.add_user_message("first message here.")
        clone = memory.copy()
        clone.add_user_message("second message here.")
        self.assertEqual(len(memory.get_history()), 1)
        self.assertEqual(len(clone.get_history()), 2)


class PrepareUnitsTests(unittest.TestCase):
    def test_tiny_fragments_merge_into_a_neighbour_of_the_same_turn(self):
        entries = [
            entry("**", 0),
            entry("Here is the full plan for the trip.", 0),
            entry("**Tuesday**", 0),
            entry("Thanks!", 1),
            entry("That really helps me a lot.", 1),
        ]
        units = prepare_units(entries, count_words, min_words=4)
        # "**" alone has no word characters at all, so it's dropped outright.
        self.assertEqual(
            [unit.text for unit in units],
            ["Here is the full plan for the trip. **Tuesday**", "Thanks! That really helps me a lot."],
        )
        self.assertEqual([unit.entry_index for unit in units], [0, 1])

    def test_exact_repeats_keep_first_occurrence_and_punctuation_only_is_dropped(self):
        entries = [entry("I adopted a cat named Luna.", 0), entry("---", 0), entry("I adopted a cat named Luna.", 2)]
        units = prepare_units(entries, count_words)
        self.assertEqual([(unit.text, unit.entry_index) for unit in units], [("I adopted a cat named Luna.", 0)])

    def test_metadata_is_carried(self):
        units = prepare_units([entry("A sentence with enough words.", 5, session=3, time="2 days before this question")], count_words)
        self.assertEqual((units[0].session, units[0].time, units[0].tokens), (3, "2 days before this question", 5))


class RenderTests(unittest.TestCase):
    def test_one_header_per_session_one_label_per_turn_and_gap_markers(self):
        units = [
            Unit(0, "First sentence of turn one.", "user", 0, session=0, time="9 days before this question"),
            Unit(2, "Third sentence of turn one.", "user", 0, session=0, time="9 days before this question"),
            Unit(3, "Reply sentence.", "model", 1, session=0, time="9 days before this question"),
            Unit(7, "Later session sentence.", "user", 4, session=2, time="1 days before this question"),
        ]
        self.assertEqual(
            render_units(units),
            "[Earlier session, 9 days before this question]\n"
            "User: First sentence of turn one. … Third sentence of turn one.\n"
            "Assistant: Reply sentence.\n"
            "\n"
            "[Earlier session, 1 days before this question]\n"
            "User: Later session sentence.",
        )

    def test_session_ids_are_never_rendered(self):
        rendered = render_units([Unit(0, "text here", "user", 0, session="answer_4be1b6b4_2")])
        self.assertNotIn("answer_", rendered)


class FusionTests(unittest.TestCase):
    def test_multi_query_max_pooling_covers_each_sub_query(self):
        # Unit 0 answers sub-query A, unit 1 answers sub-query B; the full
        # question alone prefers the distractor (unit 2) over unit 1.
        full = [0.9, 0.1, 0.5, 0.0]
        sub_a = [0.9, 0.0, 0.1, 0.0]
        sub_b = [0.0, 0.9, 0.1, 0.0]
        single = fuse_scores([full], SelectionConfig())
        multi = fuse_scores([full, sub_a, sub_b], SelectionConfig())
        self.assertLess(single[1], single[2])
        self.assertGreater(multi[1], multi[2])

    def test_hybrid_weight_blends_bm25(self):
        dense = [[1.0, 0.0, 0.0]]
        lexical = [[0.0, 0.0, 1.0]]
        fused = fuse_scores(dense, SelectionConfig(lexical_weight=0.75), lexical)
        self.assertGreater(fused[2], fused[0])

    def test_recency_bonus_favours_later_units(self):
        fused = fuse_scores([[1.0, 1.0, 0.0]], SelectionConfig(recency_weight=0.5))
        self.assertGreater(fused[1], fused[0])


class SelectTests(unittest.TestCase):
    def test_budget_includes_headers_and_labels(self):
        units = make_units(["one two three"] * 6, sessions=[0, 0, 1, 1, 2, 2], entries=[0, 1, 2, 3, 4, 5])
        for unit in units:
            unit.time = "5 days before this question"
        scores = [6, 5, 4, 3, 2, 1]
        budget = 20
        selected = select_units(units, scores, budget, count_words)
        rendered = render_units([units[index] for index in selected])
        self.assertLessEqual(count_words(rendered), budget)
        self.assertTrue(selected)

    def test_ties_prefer_the_more_recent_unit(self):
        units = make_units(["alpha beta", "gamma delta", "epsilon zeta"])
        self.assertEqual(select_units(units, [1.0, 1.0, 0.0], 3, count_words), [1])

    def test_mmr_skips_near_duplicates(self):
        units = make_units([
            "March 20 2023 2 pm slot",
            "March 20 2023 2 pm slot again",
            "The dentist is on Elm Street",
        ])
        scores = [3.0, 2.9, 2.0]
        plain = select_units(units, scores, 15, count_words)
        diverse = select_units(units, scores, 15, count_words, SelectionConfig(mmr_lambda=2.0))
        self.assertEqual(plain, [0, 1])
        self.assertEqual(diverse, [0, 2])

    def test_semantic_mmr_catches_paraphrases_that_share_no_words(self):
        import numpy as np

        units = make_units(["I moved to Denver", "we relocated to Colorado", "my cat likes tuna"])
        scores = [3.0, 2.9, 2.0]
        # 0 and 1 mean the same thing; 2 is unrelated (a third axis).
        embeddings = {0: np.array([1.0, 0.1, 0.0]), 1: np.array([0.98, 0.12, 0.0]), 2: np.array([0.0, 0.0, 1.0])}
        budget = 10
        lexical = select_units(units, scores, budget, count_words, SelectionConfig(mmr_lambda=2.0))
        semantic = select_units(units, scores, budget, count_words,
                                SelectionConfig(mmr_lambda=2.0, mmr_similarity="semantic"), embeddings)
        self.assertEqual(lexical, [0, 1])
        self.assertEqual(semantic, [0, 2])

    def test_semantic_mmr_without_embeddings_is_an_error(self):
        units = make_units(["a b c"])
        with self.assertRaises(ValueError):
            select_units(units, [1.0], 10, count_words, SelectionConfig(mmr_lambda=1.0, mmr_similarity="semantic"))

    def test_two_level_restricts_to_best_sessions(self):
        units = make_units(["a b", "c d", "e f", "g h"], sessions=[0, 0, 1, 1], entries=[0, 1, 2, 3])
        scores = [5.0, 4.0, 4.5, 0.0]
        selected = select_units(units, scores, 100, count_words, SelectionConfig(top_sessions=1))
        self.assertEqual(selected, [0, 1])

    def test_window_and_pair_turns_expand_the_pick(self):
        units = make_units(
            ["q one here", "q two here", "q three here", "answer alpha", "answer beta"],
            roles=["user", "user", "user", "model", "model"],
            entries=[0, 0, 0, 1, 1],
            sessions=[0, 0, 0, 0, 0],
        )
        scores = [0.0, 5.0, 0.0, 0.1, 3.0]
        config = SelectionConfig(window=1, pair_turns=True)
        # Best unit 1 brings its neighbours 0 and 2 plus the reply's best
        # sentence 4: 2 (header) + 1 + 9 (user turn) + 1 + 2 (reply) = 15.
        self.assertEqual(select_units(units, scores, 15, count_words, config), [0, 1, 2, 4])

    def test_bundle_that_does_not_fit_falls_back_to_the_unit_alone(self):
        units = make_units(["q one here", "q two here", "q three here"], entries=[0, 0, 0])
        selected = select_units(units, [0.0, 5.0, 0.0], 4, count_words, SelectionConfig(window=1))
        self.assertEqual(selected, [1])


class ScorerTests(unittest.TestCase):
    def test_bm25_prefers_rare_matching_terms(self):
        units = make_units(["the cat sat", "the dog sat", "zanzibar trip planned"])
        scores = BM25Scorer().score_units(units, ["my zanzibar trip"])[0]
        self.assertEqual(max(range(3), key=scores.__getitem__), 2)

    def test_lexical_scorer_overlap(self):
        units = make_units(["red car", "blue boat"])
        self.assertEqual(LexicalScorer().score_units(units, ["red"])[0], [0.5, 0.0])


class TokenWiseTests(unittest.TestCase):
    def test_abstractive_model_receives_rendered_units(self):
        class Abstractive:
            def compress(self, query, token_budget, context):
                self.context = context
                return "short"

        model = Abstractive()
        tokenwise = TokenWise(model=model)
        result = tokenwise.compress_units("q", 10, make_units(["some text here"], sessions=[0]))
        self.assertEqual(result.text, "short")
        self.assertIsNone(result.selected)
        self.assertIn("User: some text here", model.context)

    def test_extractive_scorer_gets_all_queries_once(self):
        class Scorer:
            calls = []

            def score_units(self, units, queries):
                self.calls.append(list(queries))
                return [[1.0] * len(units) for _ in queries]

        scorer = Scorer()
        result = TokenWise(model=scorer).compress_units("q", 50, make_units(["a b c", "d e f"]), extra_queries=["s1", "q", "s1"])
        self.assertEqual(scorer.calls, [["q", "s1"]])
        self.assertEqual(len(result.selected), 2)


if __name__ == "__main__":
    unittest.main()
