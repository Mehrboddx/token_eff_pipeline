import sys
import types as pytypes
import unittest


class Content:
    def __init__(self, role=None, parts=None):
        self.role = role
        self.parts = parts or []


class Part:
    def __init__(self, text):
        self.text = text

    @staticmethod
    def from_text(text):
        return Part(text)


fake_google = pytypes.ModuleType("google")
fake_genai = pytypes.ModuleType("google.genai")
fake_genai.types = pytypes.SimpleNamespace(Content=Content, Part=Part)
sys.modules.setdefault("google", fake_google)
sys.modules.setdefault("google.genai", fake_genai)

from core.runner import Runner


class DummyMemory:
    def __init__(self, history):
        self._history = history

    def get_history(self):
        return self._history

    def get_sentences(self, role=None):
        sentences = []
        for content in self._history:
            text = " ".join(part.text for part in content.parts if getattr(part, "text", None))
            if role is None or content.role == role:
                sentences.append(text)
        return sentences


class DummyTokenWise:
    def __init__(self):
        self.calls = []

    def compress(self, query, token_budget, sentences=None, context=None):
        self.calls.append(list(sentences or []))
        return "compressed:" + " | ".join(sentences or [])


class RunnerCompressionTests(unittest.TestCase):
    def test_compression_excludes_recent_history(self):
        history = [
            Content(role="user", parts=[Part.from_text(text="Older user turn")]),
            Content(role="model", parts=[Part.from_text(text="Older model turn")]),
            Content(role="user", parts=[Part.from_text(text="Recent user turn")]),
            Content(role="model", parts=[Part.from_text(text="Recent model turn")]),
        ]
        memory = DummyMemory(history)
        tokenwise = DummyTokenWise()
        runner = Runner(agent=None, memory=memory, tokenwise=tokenwise, recent_turns=2, compression_sentence_threshold=0)

        built_history, mode = runner._build_history("summarize the conversation", previous_sentences=["x"])

        self.assertEqual(mode, "compressed_history")
        self.assertEqual(len(built_history), 3)
        self.assertTrue(built_history[0].parts[0].text.startswith("Relevant user context:"))
        self.assertEqual([content.role for content in built_history[1:]], ["user", "model"])
        self.assertEqual([part.text for part in built_history[1].parts], ["Recent user turn"])
        self.assertEqual([part.text for part in built_history[2].parts], ["Recent model turn"])
        self.assertEqual(tokenwise.calls[0], ["Older user turn"])
        self.assertEqual(tokenwise.calls[1], ["Older model turn"])


if __name__ == "__main__":
    unittest.main()
