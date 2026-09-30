import unittest

from tests import _fake_genai

_fake_genai.install()

from core.memory import Memory  # noqa: E402
from core.runner import Runner  # noqa: E402
from core.tokenWise import CompressionResult  # noqa: E402


def count_words(text):
    return len(text.split())


class RecordingTokenWise:
    def __init__(self):
        self.calls = []

    def count_tokens(self, text):
        return count_words(text)

    def compress_units(self, query, token_budget, units, extra_queries=()):
        self.calls.append({"query": query, "extra": list(extra_queries), "units": list(units)})
        return CompressionResult(text="SELECTED", selected=list(units), tokens=1)


def seeded_memory():
    memory = Memory()
    for index in range(4):
        meta = {"session": index, "time": f"{10 - index} days before this question"}
        prefix = f"(from {10 - index} days before this question) "
        memory.add_user_message(f"User fact number {index} is quite specific.", meta=meta, prefix=prefix)
        memory.add_model_content(
            _fake_genai.Content(role="model", parts=[_fake_genai.Part.from_text(prefix + f"Reply {index} acknowledges it.")]),
            meta=meta,
            scored_text=f"Reply {index} acknowledges it.",
        )
    return memory


class RunnerCompressionTests(unittest.TestCase):
    def build(self, memory, tokenwise):
        return Runner(agent=None, memory=memory, tokenwise=tokenwise, recent_turns=2, compression_sentence_threshold=0)

    def test_recent_turns_pass_through_raw_and_older_ones_are_compressed(self):
        memory = seeded_memory()
        tokenwise = RecordingTokenWise()
        runner = self.build(memory, tokenwise)
        memory.add_user_message("What was fact number one?")

        history, mode = runner._build_history("What was fact number one?", previous_sentences=["x"])

        self.assertEqual(mode, "compressed_history")
        self.assertTrue(history[0].parts[0].text.startswith("Relevant earlier context:\nSELECTED"))
        cutoff = runner.last_compression["cutoff_index"]
        self.assertEqual(history[1:], memory.get_history()[cutoff:])
        self.assertTrue(all(unit.entry_index < cutoff for unit in tokenwise.calls[0]["units"]))

    def test_scored_units_carry_metadata_beside_text_not_inside_it(self):
        memory = seeded_memory()
        tokenwise = RecordingTokenWise()
        runner = self.build(memory, tokenwise)
        memory.add_user_message("q")

        runner._build_history("q", previous_sentences=["x"])

        units = tokenwise.calls[0]["units"]
        self.assertTrue(units)
        for unit in units:
            self.assertNotIn("days before", unit.text)
            self.assertNotIn("User:", unit.text)
            self.assertIsNotNone(unit.time)
        self.assertIn("(from 10 days before this question)", memory.get_history()[0].parts[0].text)

    def test_retrieval_queries_are_what_gets_scored(self):
        memory = seeded_memory()
        tokenwise = RecordingTokenWise()
        runner = self.build(memory, tokenwise)
        asked = "(Today's date is 2023/05/01.) How many days between A and B?"
        memory.add_user_message(asked)

        runner._build_history(asked, ["x"], retrieval_queries=["How many days between A and B?", "when was A", "when was B"])

        call = tokenwise.calls[0]
        self.assertEqual(call["query"], "How many days between A and B?")
        self.assertEqual(call["extra"], ["when was A", "when was B"])

    def test_short_history_is_not_compressed(self):
        memory = Memory()
        memory.add_user_message("only one turn")
        runner = self.build(memory, RecordingTokenWise())

        history, mode = runner._build_history("q", previous_sentences=["x"])

        self.assertEqual(mode, "full_history")
        self.assertIsNone(runner.last_compression)


if __name__ == "__main__":
    unittest.main()
