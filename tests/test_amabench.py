"""AMA-Bench: trajectory formatting, budgeted contexts, answer parsing, judging."""

import unittest
from unittest import mock

from integrations.amabench import answer, contexts, data, judge

TRAJECTORY = [
    {"turn_idx": 0, "action": "open the drawer", "observation": "The drawer is open. A key lies inside."},
    {"turn_idx": 1, "action": "take key", "observation": "You take the brass key."},
    {"turn_idx": 2, "action": "go north", "observation": None},
]


def words(text):
    return len((text or "").split())


class TestData(unittest.TestCase):
    def test_reference_formatting(self):
        self.assertEqual(data.trajectory_text(TRAJECTORY[:1]),
                         "Step 0:\nAction: open the drawer\nObservation: The drawer is open. A key lies inside.\n")
        self.assertIn("Observation: \n", data.trajectory_text(TRAJECTORY))  # missing observation

    def test_sample_takes_n_per_domain(self):
        episodes = [{"episode_id": f"{d}{i}", "domain": d} for d in "ab" for i in range(5)] + [{"episode_id": "c0", "domain": "c"}]
        chosen = data.sample(episodes, 2)
        self.assertEqual(sorted(e["domain"] for e in chosen), ["a", "a", "b", "b", "c"])
        self.assertEqual(chosen, data.sample(list(reversed(episodes)), 2))  # independent of file order


class TestContexts(unittest.TestCase):
    def test_pack_takes_in_rank_order_and_shows_in_step_order(self):
        blocks = ["one two", "three four five six", "seven"]
        with mock.patch.object(contexts, "count_tokens", words):
            self.assertEqual(contexts.pack(blocks, [2, 0, 1], budget=5), "one two\nseven")

    def test_actions_latest_first(self):
        with mock.patch.object(contexts, "count_tokens", words):
            text = contexts.actions_context(TRAJECTORY, budget=12)
        self.assertEqual(text, "Step 1: Action: take key\nStep 2: Action: go north")

    def test_truncate_keeps_head_and_tail_within_budget(self):
        text = " ".join(f"w{i}" for i in range(2000))
        cut = contexts.truncate(text, 200)
        self.assertTrue(cut.startswith("w0 w1"))
        self.assertTrue(cut.endswith("w1999"))
        self.assertIn("middle of the trajectory cut", cut)
        self.assertLessEqual(data.count_tokens(cut), 202)
        self.assertEqual(contexts.truncate("short", 200), "short")

    def test_bm25_prefers_the_matching_step(self):
        try:
            import rank_bm25  # noqa: F401
        except ImportError:
            self.skipTest("rank-bm25 not installed")
        text = contexts.bm25_context(TRAJECTORY, "what key did the agent take", budget=25)
        self.assertIn("Step 1:", text)

    def test_service_entries_carry_the_true_step(self):
        entries = contexts.service_entries(TRAJECTORY)
        self.assertEqual([entry["step"] for entry in entries], [0, 1, 2])
        self.assertEqual(entries[1]["text"], "Action: take key\nObservation: You take the brass key.")

    def test_service_call_refits_to_the_budget(self):
        import json

        sent = []

        class Reply:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self): return self.body

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data)
            sent.append(payload)
            text = " ".join(["x"] * (payload["budget"] + 3))  # labels push it 3 over
            return Reply(json.dumps({"text": text, "ms": 1, "units_in": 9, "units_kept": 2}).encode())

        with mock.patch.object(contexts, "count_tokens", words), mock.patch.object(contexts.urllib.request, "urlopen", fake_urlopen):
            row = contexts.service_context(TRAJECTORY, "q", 20, "http://x", {"selector": "random", "seed": 0})
        self.assertEqual([p["budget"] for p in sent], [20, 17])
        self.assertEqual(words(row["text"]), 20)
        self.assertFalse(sent[0]["dedupe"])
        self.assertEqual(sent[0]["selector"], "random")
        self.assertEqual(sent[0]["query"], "q")


class TestActionsWithExcerpts(unittest.TestCase):
    def test_action_log_and_observation_lines_under_their_step(self):
        import json

        sent = []

        class Reply:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self): return self.body

        def fake_urlopen(request, timeout):
            sent.append(json.loads(request.data))
            return Reply(json.dumps({"text": "[step 0, observation]\nA key lies inside.", "ms": 1}).encode())

        with mock.patch.object(contexts, "count_tokens", words), mock.patch.object(contexts.urllib.request, "urlopen", fake_urlopen):
            row = contexts.actions_with_excerpts(TRAJECTORY, "which key?", 24, "http://x", {})
        # half the budget (12 words) holds the last two actions; step 0 appears for its excerpt only
        self.assertEqual(row["text"], "Step 0:\n--- observation (excerpt) ---\nA key lies inside.\n\n"
                                      "Step 1: Action: take key\n\nStep 2: Action: go north")
        self.assertEqual([entry["step"] for entry in sent[0]["entries"]], [0, 1, 2])
        self.assertEqual(sent[0]["entries"][2]["text"], "")  # no observation
        self.assertEqual(sent[0]["budget"], 24 - words("Step 1: Action: take key\n\nStep 2: Action: go north"))


class TestAnswer(unittest.TestCase):
    def test_parse_batched_and_single(self):
        self.assertEqual(answer.parse_answers("Answer[1]: north\nAnswer[2]: the key\n", 2), ["north", "the key"])
        self.assertEqual(answer.parse_answers("Answer[2]: only two", 2), ["", "only two"])
        self.assertEqual(answer.parse_answers("<think>hm</think>It was brass.", 1), ["It was brass."])

    def test_prompts_follow_the_reference(self):
        single = answer.single_prompt("find the key", "[step 1, x]\ntake key", "What was taken?")
        self.assertIn("(only some steps and lines are shown)", single)
        self.assertTrue(single.endswith("Question 1: What was taken?\n\n## Instructions\nProvide a direct and concise answer.\n\nAnswer[1]: [your answer here]"))
        batch = answer.batch_prompt("find the key", "Step 0: ...", ["a?", "b?"])
        self.assertNotIn("only some steps", batch)
        self.assertIn("Question 2: b?\n", batch)
        self.assertTrue(batch.endswith("Answer[1]: [your answer here]\nAnswer[2]: [your answer here]"))


class TestJudge(unittest.TestCase):
    def test_last_yes_or_no_decides(self):
        self.assertEqual(judge.verdict("yes"), 1.0)
        self.assertEqual(judge.verdict("No."), 0.0)
        self.assertEqual(judge.verdict("<think>yes maybe</think> no"), 0.0)
        self.assertEqual(judge.verdict("no... actually yes"), 1.0)
        self.assertIsNone(judge.verdict("unsure"))

    def test_prompt_has_reference_and_prediction(self):
        prompt = judge.judge_prompt({"episode_id": "e", "task": "t", "task_type": "tt"},
                                    {"question": "q?", "answer": "ref"}, "pred")
        self.assertIn("Task Type: tt\nEpisode ID: e\nTask Context: t", prompt)
        self.assertIn("Reference Answer: ref\n\nPredicted Answer: pred", prompt)


if __name__ == "__main__":
    unittest.main()
