import unittest

from tests import _fake_genai

_fake_genai.install()

from core.tokenWise import TokenWise  # noqa: E402
from core.units import prepare_units  # noqa: E402
from eval.locomo import ADVERSARIAL_ANSWER, conversation_items, evidence_ids  # noqa: E402
from eval.longmemeval import seed_memory_from_haystack  # noqa: E402


def sample():
    return {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to the LGBTQ support group yesterday."},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "That sounds wonderful, I painted a sunrise.",
                 "blip_caption": "a painting of a sunrise over a lake"},
            ],
            "session_2_date_time": "9:00 am on 20 May, 2023",
            "session_2": [
                {"speaker": "Melanie", "dia_id": "D2:1", "text": "I ran a charity race for mental health."},
                {"speaker": "Caroline", "dia_id": "D2:2", "text": "Amazing, how did it go for you?"},
            ],
            "session_3_date_time": "noon on 1 June, 2023",
            "session_3": [],
        },
        "qa": [
            {"question": "When did Caroline go to the LGBTQ support group?", "answer": "7 May 2023",
             "evidence": ["D1:1"], "category": 2},
            {"question": "What did Melanie paint and what did she run?", "answer": 2022,
             "evidence": ["D1:2; D2:1"], "category": 1},
            {"question": "What did Caroline paint?", "adversarial_answer": "a sunrise",
             "evidence": ["D1:2"], "category": 5},
        ],
    }


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.items = conversation_items(sample())

    def test_evidence_ids_handle_lists_and_skip_malformed(self):
        self.assertEqual(evidence_ids(["D8:6; D9:17", "D:11:26", "D9:1 D4:4"]), {(8, 6), (9, 17), (9, 1), (4, 4)})

    def test_items_have_longmemeval_shape(self):
        temporal, multi, adversarial = self.items
        self.assertEqual(temporal["question_id"], "conv-1_q000")
        self.assertEqual(temporal["question_type"], "temporal")
        self.assertEqual(multi["answer"], "2022")
        self.assertEqual(adversarial["answer"], ADVERSARIAL_ANSWER)
        # empty session_3 is dropped, so sessions, ids and dates stay aligned
        self.assertEqual(temporal["haystack_session_ids"], ["conv-1_S1", "conv-1_S2"])
        self.assertEqual(temporal["session_time_labels"], ["1:56 pm on 8 May, 2023", "9:00 am on 20 May, 2023"])
        self.assertEqual(multi["answer_session_ids"], ["conv-1_S1", "conv-1_S2"])

    def test_evidence_flags_are_per_question_and_shared_turns_untouched(self):
        temporal, multi, _ = self.items
        self.assertEqual([t["has_answer"] for s in temporal["haystack_sessions"] for t in s], [True, False, False, False])
        self.assertEqual([t["has_answer"] for s in multi["haystack_sessions"] for t in s], [False, True, True, False])

    def test_speakers_map_to_roles_and_captions_are_kept(self):
        first, second = self.items[0]["haystack_sessions"][0]
        self.assertEqual((first["role"], first["speaker"]), ("user", "Caroline"))
        self.assertEqual((second["role"], second["speaker"]), ("assistant", "Melanie"))
        self.assertIn("[shares a photo: a painting of a sunrise over a lake]", second["content"])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.item = conversation_items(sample())[0]
        self.memory = seed_memory_from_haystack(self.item)

    def test_raw_history_shows_absolute_date_and_speaker(self):
        self.assertEqual(self.memory.history[0].parts[0].text,
                         "(1:56 pm on 8 May, 2023) Caroline: I went to the LGBTQ support group yesterday.")

    def test_scored_text_has_no_metadata(self):
        texts = [entry["text"] for entry in self.memory.sentences]
        self.assertFalse(any("Caroline:" in text or "2023" in text for text in texts))

    def test_compressed_context_labels_turns_by_speaker_under_dated_headers(self):
        tokenwise = TokenWise()
        units = prepare_units(self.memory.sentences, tokenwise.count_tokens)
        result = tokenwise.compress_units("LGBTQ support group", token_budget=200, units=units)
        self.assertIn("[Earlier session, 1:56 pm on 8 May, 2023]", result.text)
        self.assertIn("Caroline: I went to the LGBTQ support group yesterday.", result.text)
        self.assertNotIn("User:", result.text)


if __name__ == "__main__":
    unittest.main()
