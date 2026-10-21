import unittest

from tests import _fake_genai

_fake_genai.install()

from core.dycp import DyCPCompressor, group_turns, kadane_dial  # noqa: E402
from core.tokenWise import TokenWise  # noqa: E402
from core.units import Unit  # noqa: E402


def unit(index, text, role, entry_index, session=0):
    return Unit(id=index, text=text, role=role, entry_index=entry_index, session=session, tokens=len(text.split()))


class KadaneDialTests(unittest.TestCase):
    def test_picks_the_contiguous_high_block_first(self):
        scores = [0, 0, 5, 6, 5, 0, 0, 0]
        self.assertEqual(kadane_dial(scores)[0], (2, 4))

    def test_finds_a_second_block_and_skips_noise(self):
        scores = [9, 9, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0]
        self.assertEqual(sorted(kadane_dial(scores)), [(0, 1), (6, 7)])

    def test_always_returns_the_best_span_even_when_nothing_clears_theta(self):
        self.assertEqual(kadane_dial([1, 1, 1, 1]), [(0, 0)])

    def test_empty_input(self):
        self.assertEqual(kadane_dial([]), [])

    def test_higher_theta_keeps_fewer_spans(self):
        scores = [9, 0, 0, 7, 0, 0, 6, 0, 0, 0]
        self.assertLessEqual(len(kadane_dial(scores, theta=3.0)), len(kadane_dial(scores, theta=0.5)))


class TurnGroupingTests(unittest.TestCase):
    def test_user_message_and_following_reply_form_one_turn(self):
        units = [
            unit(0, "Where should I eat in Rome?", "user", 0),
            unit(1, "Try Roscioli.", "model", 1),
            unit(2, "It is cozy.", "model", 1),
            unit(3, "Thanks, and in Milan?", "user", 2),
            unit(4, "Try Ratanà.", "model", 3),
        ]
        self.assertEqual(group_turns(units), [[0, 1, 2], [3, 4]])

    def test_leading_reply_without_its_question_is_its_own_turn(self):
        units = [unit(0, "As I said, Roscioli.", "model", 5), unit(1, "Great, thanks!", "user", 6)]
        self.assertEqual(group_turns(units), [[0], [1]])


class FakeEncoder:
    """Query 'rome' lights up dimension 0; texts mentioning Rome or Roscioli do too."""

    def encode(self, texts):
        return [[1.0 if ("rome" in t.lower() or "roscioli" in t.lower()) else 0.0, 1.0] for t in texts]


class DyCPCompressorTests(unittest.TestCase):
    def setUp(self):
        self.units = []
        entry = 0
        for topic in ["gardening", "taxes", "Rome dinner", "Roscioli booking", "running", "cooking", "music", "travel"]:
            self.units.append(unit(len(self.units), f"Question about {topic} here.", "user", entry))
            self.units.append(unit(len(self.units), f"An answer about {topic} here.", "model", entry + 1))
            entry += 2

    def test_selects_the_relevant_turns_whole_and_in_order(self):
        compressor = DyCPCompressor(encoder_factory=FakeEncoder)
        selected = compressor.select_units(self.units, "Which restaurant in Rome?")
        self.assertEqual(selected, [4, 5, 6, 7])

    def test_tokenwise_uses_the_selector_and_ignores_the_budget(self):
        tokenwise = TokenWise(model=DyCPCompressor(encoder_factory=FakeEncoder))
        result = tokenwise.compress_units("Which restaurant in Rome?", token_budget=1, units=self.units)
        self.assertEqual([u.id for u in result.selected], [4, 5, 6, 7])
        self.assertIn("Roscioli", result.text)
        self.assertNotIn("gardening", result.text)


if __name__ == "__main__":
    unittest.main()
