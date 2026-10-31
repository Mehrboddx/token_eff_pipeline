import unittest

from tests import _fake_genai

_fake_genai.install()

from core.cpcCompressor import CPCCompressor, cpc_cache_key  # noqa: E402
from core.selection import SelectionConfig, select_units  # noqa: E402
from core.tokenWise import TokenWise  # noqa: E402
from core.units import Unit, render_flat  # noqa: E402
from eval.longmemeval import (  # noqa: E402
    CompressionSettings,
    compress_question,
    paper_transcript,
    retrieval_queries,
    run_tag,
    seed_memory_from_haystack,
)
from tests.test_paper_cpc import item  # noqa: E402


def units():
    return [
        Unit(0, "I bought a smoker today.", "user", 0, session=0, time="10 days before this question"),
        Unit(1, "Nice, enjoy the barbecue!", "model", 1, session=0, time="10 days before this question"),
        Unit(2, "Try brisket in your new smoker.", "model", 3, session=1, time="5 days before this question"),
    ]


def encoder(**flags):
    """A CPCCompressor without its model: prefixes and groups need none."""
    compressor = object.__new__(CPCCompressor)
    compressor.speaker_labels = flags.get("speaker_labels", True)
    compressor.session_chunks = flags.get("session_chunks", True)
    return compressor


class LengthScorer:
    def score_units(self, units, queries):
        return [[float(len(unit.text)) for unit in units] for _ in queries]


class AblationTests(unittest.TestCase):
    def test_flat_output_is_plain_text_and_pays_no_layout_tokens(self):
        self.assertEqual(render_flat(list(reversed(units()))),
                         "I bought a smoker today. Nice, enjoy the barbecue! Try brisket in your new smoker.")
        count = lambda text: len(text.split())  # noqa: E731
        budget = sum(count(unit.text) for unit in units())
        self.assertEqual(len(select_units(units(), [3, 2, 1], budget, count, SelectionConfig(flat_output=True))), 3)
        self.assertLess(len(select_units(units(), [3, 2, 1], budget, count, SelectionConfig())), 3)

        result = TokenWise(model=LengthScorer(), selection=SelectionConfig(flat_output=True)).compress_units(
            "q", 4000, units())
        self.assertNotIn("[Earlier session", result.text)
        self.assertNotIn("User:", result.text)

    def test_encoder_labels_and_session_groups_can_be_switched_off(self):
        self.assertEqual(encoder().unit_prefixes(units()), ["\nUser: ", "\nAssistant: ", "\nAssistant: "])
        self.assertEqual(encoder(speaker_labels=False).unit_prefixes(units()), ["\n", "\n", "\n"])
        self.assertEqual(encoder().unit_groups(units()), [0, 0, 1])
        self.assertEqual(encoder(session_chunks=False).unit_groups(units()), [None, None, None])
        self.assertNotEqual(cpc_cache_key("mistral", 6144), cpc_cache_key("mistral", 6144, speaker_labels=False))
        self.assertNotEqual(cpc_cache_key("mistral", 6144), cpc_cache_key("mistral", 6144, session_chunks=False))

    def test_date_in_text_puts_the_date_into_the_scored_sentences(self):
        plain = seed_memory_from_haystack(item()).get_sentences()
        dated = seed_memory_from_haystack(item(), date_in_text=True).get_sentences()
        self.assertFalse(any("days before" in sentence for sentence in plain))
        self.assertTrue(any("days before" in sentence for sentence in dated))
        raw = seed_memory_from_haystack(item(), date_in_text=True).get_history()[0].parts[0].text
        self.assertEqual(raw.count("days before this question"), 1)  # not shown twice

    def test_dated_query_scores_the_date_tagged_question(self):
        self.assertEqual(retrieval_queries(item()), ["When did I buy the smoker?"])
        self.assertTrue(retrieval_queries(item(), dated=True)[0].startswith("(Today's date is"))

    def test_ablations_reach_compression_and_the_run_name(self):
        seen = {}

        class Recording(LengthScorer):
            def score_units(self, units, queries):
                seen["queries"] = list(queries)
                seen["texts"] = [unit.text for unit in units]
                return super().score_units(units, queries)

        settings = CompressionSettings(sentence_threshold=0, token_budget=4000, recent_turns=1,
                                       ablate=("date-in-text", "dated-query"))
        compress_question(item(), TokenWise(model=Recording()), settings)
        self.assertTrue(seen["queries"][0].startswith("(Today's date is"))
        self.assertTrue(any("days before" in text for text in seen["texts"]))

        self.assertEqual(run_tag("compress", {"compressor": "cpc", "cpc_preset": "mistral", "token_budget": 4000,
                                              "ablate": ["date-in-text", "flat-output"]}),
                         "compress_cpc-mistral_b4000_abl-date-in-text_abl-flat-output")

    def test_paper_turn_dates_put_the_date_on_every_turn(self):
        lines = paper_transcript(seed_memory_from_haystack(item()), turn_dates=True).splitlines()
        self.assertEqual(lines[0], "[10 days before this question] User: I bought a smoker today. It was on sale.")
        self.assertEqual(len(lines), 4)
        self.assertEqual(run_tag("compress", {"compressor": "cpc", "cpc_preset": "mistral", "token_budget": 4000,
                                              "paper_cpc": True, "paper_dates": "turn"}),
                         "compress_cpc-mistral_b4000_paper-turndates")


if __name__ == "__main__":
    unittest.main()
