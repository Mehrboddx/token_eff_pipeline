import unittest

from tests import _fake_genai

_fake_genai.install()

from core.provence import (  # noqa: E402
    CLS_ID, SEP_ID, ProvenceCompressor, ProvenceSelector, normalize_query, pack_windows, unit_scores,
)
from core.tokenWise import TokenWise  # noqa: E402
from core.units import Unit  # noqa: E402


def unit(index, text, role, entry_index, session=0):
    return Unit(id=index, text=text, role=role, entry_index=entry_index, session=session, tokens=len(text.split()))


class FakeModel:
    """One token per word (ids >= 10 so they never clash with [CLS]/[SEP]);
    words mentioning Rome or Roscioli get keep probability 0.9, others 0.05."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        return [self.vocab.setdefault(word, 10 + len(self.vocab)) for word in text.split()]

    def keep_probs(self, sequences):
        words = {token: word for word, token in self.vocab.items()}
        return [[0.9 if ("rome" in words.get(t, "").lower() or "roscioli" in words.get(t, "").lower()) else 0.05
                 for t in sequence] for sequence in sequences]


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()

    def test_window_layout_and_role_labels(self):
        units = [unit(0, "Where to eat?", "user", 0), unit(1, "Try Roscioli.", "model", 1), unit(2, "It is cozy.", "model", 1)]
        query = self.model.encode("rome")
        [window] = pack_windows(units, query, self.model.encode)
        self.assertEqual(window.ids[0], CLS_ID)
        self.assertEqual(window.ids[len(query) + 1], SEP_ID)
        self.assertEqual(window.ids[-1], SEP_ID)
        # labels "User:" and "Assistant:" sit between, never inside, unit spans
        covered = {position for _, start, end in window.spans for position in range(start, end)}
        self.assertEqual(len(covered), 3 + 2 + 3)
        self.assertEqual(len(window.ids), 1 + len(query) + 1 + 2 + len(covered) + 1)

    def test_sessions_never_share_a_window(self):
        units = [unit(0, "Hello there friend.", "user", 0, session="a"), unit(1, "Hello again friend.", "user", 1, session="b")]
        self.assertEqual(len(pack_windows(units, [], self.model.encode)), 2)

    def test_windows_respect_the_length_limit_and_long_units_are_split(self):
        units = [unit(0, " ".join(f"w{i}" for i in range(50)), "user", 0), unit(1, "short one here", "model", 1)]
        windows = pack_windows(units, [99], self.model.encode, max_len=20)
        self.assertTrue(all(len(window.ids) <= 20 for window in windows))
        pieces = [span for window in windows for span in window.spans if span[0] == 0]
        self.assertEqual(sum(end - start for _, start, end in pieces), 50)

    def test_unit_scores_mean_and_threshold_share(self):
        units = [unit(0, "a b c d", "user", 0)]
        windows = pack_windows(units, [], self.model.encode)
        start = windows[0].spans[0][1]
        probs = [[0.0] * len(windows[0].ids)]
        probs[0][start:start + 4] = [0.9, 0.9, 0.05, 0.05]
        self.assertAlmostEqual(unit_scores(windows, probs, 1)[0], 0.475)
        self.assertAlmostEqual(unit_scores(windows, probs, 1, threshold=0.1)[0], 0.5)

    def test_query_is_normalised_like_provences_training_data(self):
        self.assertEqual(normalize_query("Where's   the Rome dinner?"), "wheres the rome dinner")


class CompressorTests(unittest.TestCase):
    def setUp(self):
        self.units = []
        entry = 0
        for topic in ["gardening", "taxes", "Rome", "Roscioli", "running"]:
            self.units.append(unit(len(self.units), f"Question about {topic} here.", "user", entry))
            self.units.append(unit(len(self.units), f"An answer about {topic} here.", "model", entry + 1))
            entry += 2

    def test_budgeted_mode_ranks_relevant_units_first(self):
        compressor = ProvenceCompressor(model_factory=FakeModel)
        [scores] = compressor.score_units(self.units, ["Which restaurant in Rome?"])
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:4]
        self.assertEqual(sorted(top), [4, 5, 6, 7])

    def test_tokenwise_fills_the_budget_with_provence_scores(self):
        tokenwise = TokenWise(model=ProvenceCompressor(model_factory=FakeModel))
        result = tokenwise.compress_units("Which restaurant in Rome?", token_budget=30, units=self.units)
        self.assertIn("Roscioli", result.text)
        self.assertNotIn("gardening", result.text)

    def test_native_mode_keeps_what_clears_the_threshold_and_ignores_the_budget(self):
        tokenwise = TokenWise(model=ProvenceSelector(threshold=0.1, model_factory=FakeModel))
        result = tokenwise.compress_units("Which restaurant in Rome?", token_budget=1, units=self.units)
        self.assertEqual([u.id for u in result.selected], [4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
