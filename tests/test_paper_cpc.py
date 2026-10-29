import unittest

from tests import _fake_genai

_fake_genai.install()

from core.tokenWise import TokenWise  # noqa: E402
from eval.longmemeval import CompressionSettings, compress_question, paper_transcript, seed_memory_from_haystack  # noqa: E402


def item():
    return {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question": "When did I buy the smoker?",
        "question_date": "2023/05/30 (Tue) 10:00",
        "answer": "May 20",
        "haystack_dates": ["2023/05/20 (Sat) 19:50", "2023/05/25 (Thu) 08:00"],
        "haystack_session_ids": ["answer_1", "s2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I bought a smoker today. It was on sale.", "has_answer": True},
                {"role": "assistant", "content": "Nice, enjoy the barbecue!"},
            ],
            [
                {"role": "user", "content": "What should I cook this weekend?"},
                {"role": "assistant", "content": "Try brisket in your new smoker."},
            ],
        ],
    }


class FakePaperCPC:
    """Records what the paper path hands to CPC and keeps the first line."""

    def __init__(self):
        self.calls = []

    def compress_document(self, context, question, token_budget, count_tokens=None):
        self.calls.append((context, question, token_budget))
        return "I bought a smoker today. It was on sale."


class PaperCPCTests(unittest.TestCase):
    def test_transcript_has_dates_and_labels_in_the_text(self):
        transcript = paper_transcript(seed_memory_from_haystack(item()))
        self.assertEqual(transcript.splitlines(), [
            "[Session: 10 days before this question]",
            "User: I bought a smoker today. It was on sale.",
            "Assistant: Nice, enjoy the barbecue!",
            "[Session: 5 days before this question]",
            "User: What should I cook this weekend?",
            "Assistant: Try brisket in your new smoker.",
        ])

    def test_whole_history_is_compressed_against_the_plain_question(self):
        model = FakePaperCPC()
        settings = CompressionSettings(token_budget=4000, paper=True)
        described, mode, question, info = compress_question(item(), TokenWise(model=model), settings)

        context, scored_question, budget = model.calls[0]
        self.assertIn("Try brisket in your new smoker.", context)  # no recent turns held back
        self.assertEqual(scored_question, "When did I buy the smoker?")
        self.assertEqual(budget, 4000)

        self.assertEqual(mode, "compressed_history")
        self.assertEqual([entry["kind"] for entry in described], ["compressed_context", "user"])
        self.assertEqual(described[1]["text"], question)
        self.assertTrue(question.startswith("(Today's date is"))
        self.assertEqual(info["evidence"]["evidence_turns_kept"], 1)


if __name__ == "__main__":
    unittest.main()
