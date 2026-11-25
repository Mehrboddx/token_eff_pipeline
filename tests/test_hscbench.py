"""HSC-Bench rebuild: trajectory conversion, receiver parsing, scoring."""

import unittest

from integrations.hscbench.receive import parse
from integrations.hscbench.score import bootstrap, f1, leaked, line_set, normalize, score_task, visibility
from integrations.hscbench.trajectories import cap_observation, from_miniswe, from_openhands, from_sweagent, history_text


class TestTrajectories(unittest.TestCase):
    def test_miniswe_pairs_follow_the_task(self):
        trajectory = {"messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "THOUGHT: look\n```bash\nls\n```"}, {"role": "user", "content": "a.py"},
            {"role": "assistant", "content": "THOUGHT: done"},
        ]}
        self.assertEqual(from_miniswe(trajectory), [{"action": "THOUGHT: look\n```bash\nls\n```", "observation": "a.py", "command": "$ ls"}])

    def test_sweagent_steps(self):
        trajectory = {"trajectory": [{"response": "I look\nopen a.py", "action": "open a.py\n", "observation": "1:x"},
                                     {"response": "no output yet"}]}
        self.assertEqual(from_sweagent(trajectory), [{"action": "I look\nopen a.py", "observation": "1:x", "command": "$ open a.py"}])

    def test_openhands_joins_calls_and_outputs(self):
        log = {"messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "assistant", "content": [], "tool_calls": [
                {"function": {"name": "execute_bash", "arguments": '{"command": "grep -n foo a.py", "security_risk": "LOW"}'}},
                {"function": {"name": "str_replace_editor", "arguments": '{"command": "view", "path": "/w/a.py"}'}}]},
            {"role": "tool", "content": [{"type": "text", "text": "3:foo"}]},
            {"role": "tool", "content": "1\tx"},
            {"role": "assistant", "content": "thinking only"},
        ]}
        pairs = from_openhands(log)
        self.assertEqual(len(pairs), 1)
        self.assertIn("$ grep -n foo a.py", pairs[0]["action"])
        self.assertIn('str_replace_editor({"command": "view", "path": "/w/a.py"})', pairs[0]["action"])
        self.assertEqual(pairs[0]["observation"], "3:foo\n1\tx")
        self.assertEqual(pairs[0]["command"], '$ grep -n foo a.py\nstr_replace_editor({"command": "view", "path": "/w/a.py"})')

    def test_long_observation_keeps_head_and_tail(self):
        self.assertEqual(cap_observation("short", limit=10), "short")
        capped = cap_observation("a" * 10 + "b" * 10, limit=10)
        self.assertTrue(capped.startswith("aaaaa\n[... 10 characters cut ...]\n"))
        self.assertTrue(capped.endswith("bbbbb"))

    def test_history_numbers_steps(self):
        text = history_text([{"action": "a", "observation": "b"}, {"action": "c", "observation": "d"}])
        self.assertTrue(text.startswith("[step 1]\na\n--- output ---\nb"))
        self.assertIn("[step 2]", text)


class TestContextCompression(unittest.TestCase):
    def test_tool_calls_verbatim_and_excerpts_under_their_step(self):
        import json
        from unittest import mock

        from integrations.hscbench import handoff

        class Reply:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self): return self.body

        sent = []

        def fake_urlopen(request, timeout):
            sent.append(json.loads(request.data))
            return Reply(json.dumps({"text": "[step 2, output]\n10: def f():\n…\n12: return x", "units_in": 9, "units_kept": 2}).encode())

        task = {"issue": "bug in f", "history": [
            {"action": "think", "observation": "a.py\nb.py", "command": "$ ls"},
            {"action": "think more", "observation": "10: def f():\n11: y\n12: return x", "command": "$ cat -n a.py"}]}
        words = lambda text: len(text.split())
        with mock.patch.object(handoff, "count_tokens", words), mock.patch.object(handoff.urllib.request, "urlopen", fake_urlopen):
            result = handoff.cpc_context(task, budget=40, url="http://x")
        self.assertEqual(result["text"], "[step 1] $ ls\n\n[step 2] $ cat -n a.py\n--- output (excerpt) ---\n10: def f():\n…\n12: return x")
        self.assertEqual([entry["text"] for entry in sent[0]["entries"]], ["a.py\nb.py", "10: def f():\n11: y\n12: return x"])
        self.assertEqual(sent[0]["budget"], 40 - words("[step 1] $ ls\n\n[step 2] $ cat -n a.py"))

        sent.clear()
        with mock.patch.object(handoff, "count_tokens", words), mock.patch.object(handoff.urllib.request, "urlopen", fake_urlopen):
            only = handoff.cpc_context(task, budget=40, url="http://x", outputs=False)
            handoff.cpc_context(task, budget=40, url="http://x", extra=handoff.CONTEXT_VARIANTS["ctx_bm25"][1])
        self.assertEqual(only["text"], "[step 1] $ ls\n\n[step 2] $ cat -n a.py")
        self.assertEqual(len(sent), 1)  # commands only: no service call
        self.assertEqual(sent[0]["selection"], {"provence_weight": 0.0, "lexical_weight": 1.0})


class TestReceiverParsing(unittest.TestCase):
    def test_parse_limits_and_validates(self):
        text = 'Here: {"locations": [{"file": "a.py", "spans": [[3, 5], [0, 2], [9, 7], ["x", 1]]}, {"spans": [[1, 2]]}]}'
        self.assertEqual(parse(text), [{"file": "a.py", "spans": [[3, 5]]}])

    def test_unparsable_is_none(self):
        self.assertIsNone(parse("no json here"))
        self.assertIsNone(parse('{"files": []}'))


class TestScoring(unittest.TestCase):
    def test_paths_match_gold_by_suffix(self):
        gold = {"src/a.py", ".github/x.yml"}
        self.assertEqual(normalize("/testbed/src/a.py", gold), "src/a.py")
        self.assertEqual(normalize("./.github/x.yml", gold), ".github/x.yml")
        self.assertEqual(normalize("/testbed/other/b.py", gold), "testbed/other/b.py")

    def test_overlapping_spans_merge(self):
        lines = line_set([{"file": "a.py", "spans": [[1, 3], [2, 4]]}], {"a.py"})
        self.assertEqual(lines, {("a.py", line) for line in range(1, 5)})

    def test_line_and_file_f1(self):
        task = {"gold": [{"file": "a.py", "start": 1, "end": 4}, {"file": "b.py", "start": 10, "end": 10}]}
        result = score_task(task, [{"file": "/w/a.py", "spans": [[3, 6]]}])
        self.assertAlmostEqual(result["line_f1"], f1({("a.py", 3), ("a.py", 4), ("a.py", 5), ("a.py", 6)},
                                                     {("a.py", 1), ("a.py", 2), ("a.py", 3), ("a.py", 4), ("b.py", 10)}))
        self.assertAlmostEqual(result["file_f1"], 2 / 3)
        self.assertEqual(score_task(task, None), {"line_f1": 0.0, "file_f1": 0.0, "parsed": False})

    def test_visibility_groups(self):
        task = {"gold": [{"file": "src/a.py", "start": 1, "end": 2}]}
        self.assertEqual(visibility(task, "cat src/a.py", "cat src/a.py"), "still recent")
        self.assertEqual(visibility(task, "ls", "early: src/a.py"), "earlier only")
        self.assertEqual(visibility(task, "ls", "ls"), "not observed")

    def test_leaked_reference_fix(self):
        self.assertTrue(leaked({"history": [{"action": "x", "command": "$ cat /home/fix.patch"}]}))
        self.assertFalse(leaked({"history": [{"action": "x", "command": "$ git diff"}]}))

    def test_bootstrap_by_repository(self):
        mean, low, high = bootstrap({"r1": [0.2, 0.4], "r2": [0.6]}, resamples=2000)
        self.assertAlmostEqual(mean, 0.4)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)


if __name__ == "__main__":
    unittest.main()
