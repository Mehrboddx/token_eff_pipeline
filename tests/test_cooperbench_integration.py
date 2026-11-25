"""integrations/cooperbench: unit building, rendering, the service's budget
logic (with a fake scorer), and the report's statistics. No model, no
CooperBench install needed."""

import json
import tempfile
import unittest
from pathlib import Path

from core.selection import SelectionConfig
from core.units import Unit
from integrations.cooperbench import report
from integrations.cooperbench.compress_server import Compressor, entries_to_units
from integrations.cooperbench.render import message_text, render_excerpt


def count(text):
    return len(text.split())


class FakeTokenWise:
    """Keeps the units whose text mentions the query word, in budget order."""

    count_tokens = staticmethod(count)

    def __init__(self):
        self.calls = 0
        self.selection = SelectionConfig()  # per-request overrides are applied to it

    def compress_units(self, query, budget, units, extra_queries=()):
        self.calls += 1
        kept, used = [], 0
        for unit in units:
            if query in unit.text and used + unit.tokens <= budget:
                kept.append(unit)
                used += unit.tokens

        class Result:
            selected = kept

        return Result()


def fake_compressor():
    compressor = Compressor.__new__(Compressor)
    compressor.tokenwise = FakeTokenWise()
    import threading

    compressor.lock = threading.Lock()
    return compressor


class TestUnits(unittest.TestCase):
    def test_lines_become_units_and_keep_their_entry(self):
        entries = [
            {"role": "assistant", "text": "I will open the parser module now.\n$ sed -n 1,40p src/parser.py"},
            {"role": "tool", "text": "def parse(text):\n    return tokenize(text) if text else []"},
        ]
        units = entries_to_units(entries, count)
        self.assertEqual([unit.entry_index for unit in units], [0, 0, 1, 1])
        self.assertIn("sed -n 1,40p src/parser.py", units[1].text)
        self.assertEqual(units[3].text, "    return tokenize(text) if text else []")  # indentation kept

    def test_short_lines_join_the_previous_unit_by_newline(self):
        code = (
            "def load_config(path, strict=False):\n    import os\n    import re\n}\nreturn the parsed config to caller"
        )
        units = entries_to_units([{"role": "tool", "text": code}], count)
        self.assertEqual(units[0].text, "def load_config(path, strict=False):\n    import os\n    import re\n}")
        self.assertEqual(len(units), 2)

    def test_exact_repeats_are_dropped(self):
        text = "\n".join(["the build failed on line twelve again"] * 3 + ["a different line of output here"])
        self.assertEqual(len(entries_to_units([{"role": "tool", "text": text}], count)), 2)
        self.assertEqual(len(entries_to_units([{"role": "tool", "text": text}], count, dedupe=False)), 4)

    def test_blank_lines_are_skipped(self):
        units = entries_to_units(
            [{"role": "tool", "text": "first line of output here\n\n\nsecond line of output here"}], count
        )
        self.assertEqual(len(units), 2)


class TestRender(unittest.TestCase):
    def unit(self, uid, entry, text):
        return Unit(id=uid, text=text, role="tool", entry_index=entry)

    def test_gap_marker_and_step_labels(self):
        entries = [{"role": "assistant", "text": ""}, {"role": "tool", "text": ""}]
        selected = [self.unit(0, 0, "a"), self.unit(2, 0, "c"), self.unit(3, 1, "d")]
        self.assertEqual(render_excerpt(selected, entries), "[step 1, you]\na\n…\nc\n\n[step 2, tool output]\nd")
        numbered = [{"role": "tool", "text": "", "step": 0, "label": "x"}, {"role": "tool", "text": "", "step": 7}]
        self.assertEqual(render_excerpt(selected, numbered), "[step 0, x]\na\n…\nc\n\n[step 7, tool output]\nd")

    def test_single_entry_has_no_label(self):
        selected = [self.unit(1, 0, "b"), self.unit(0, 0, "a")]
        self.assertEqual(render_excerpt(selected, [{"role": "colleague", "text": ""}]), "a\nb")

    def test_message_text_includes_commands(self):
        message = {
            "role": "assistant",
            "content": "Checking the tests.",
            "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps({"command": "pytest -x"})}}],
        }
        self.assertEqual(message_text(message), "Checking the tests.\n$ pytest -x")


class TestService(unittest.TestCase):
    def test_short_message_is_returned_verbatim_without_scoring(self):
        compressor = fake_compressor()
        text = "I take parser.py, you take lexer.py please."
        result = compressor.compress(
            {"entries": [{"role": "colleague", "text": text}], "ratio": 0.5, "min_tokens": 150}
        )
        self.assertEqual(result["text"], text)
        self.assertEqual(compressor.tokenwise.calls, 0)

    def test_budget_comes_from_the_raw_text_not_the_deduplicated_units(self):
        compressor = fake_compressor()
        text = "\n".join(["I take parser.py and you take lexer.py please"] * 40)
        result = compressor.compress(
            {"entries": [{"role": "colleague", "text": text}], "query": "parser", "ratio": 0.5, "min_tokens": 150}
        )
        self.assertEqual(result["tokens_in"], count(text))
        self.assertEqual(compressor.tokenwise.calls, 1)  # 360 words > 150: compressed, though one unique line

    def test_long_message_is_cut_to_ratio(self):
        compressor = fake_compressor()
        lines = [f"line {i} mentions parser and more words here" for i in range(10)]
        lines += [f"line {i} is about something unrelated entirely" for i in range(10, 20)]
        result = compressor.compress(
            {
                "entries": [{"role": "colleague", "text": "\n".join(lines)}],
                "query": "parser",
                "ratio": 0.5,
                "min_tokens": 5,
            }
        )
        self.assertEqual(compressor.tokenwise.calls, 1)
        self.assertLessEqual(result["tokens_out"], result["budget"] + 2)  # "…" markers
        self.assertNotIn("unrelated", result["text"])


class TestReport(unittest.TestCase):
    def test_mcnemar(self):
        better, worse, p = report.mcnemar([False] * 6 + [True] * 4, [True] * 6 + [True] * 4)
        self.assertEqual((better, worse), (6, 0))
        self.assertAlmostEqual(p, 2 / 64)

    def test_load_pair_sums_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            pair = Path(tmp)
            (pair / "eval.json").write_text(json.dumps({"both_passed": True, "feature1": {"passed": True}}))
            rows = [
                {
                    "event": "llm",
                    "kind": "step",
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "cached_tokens": 40,
                    "reasoning_tokens": 5,
                },
                {
                    "event": "llm",
                    "kind": "summary",
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                },
                {"event": "compress", "kind": "message", "tokens_in": 300, "tokens_out": 150, "ms": 12.0},
            ]
            (pair / "agent1_ledger.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            row = report.load_pair(pair)
        assert row is not None
        self.assertEqual((row["input"], row["output"], row["total"]), (150, 30, 180))
        self.assertEqual((row["steps"], row["summaries"], row["messages"]), (1, 1, 1))
        self.assertEqual(row["feature_passed"], 1)


if __name__ == "__main__":
    unittest.main()
