"""CPC recall history processor: pool, query, rendering, and the processor with a stub masker."""

import json
import sys
import types
import unittest
from unittest import mock

from integrations.swebench_ctx.recall import RECALL_HEADER, CPCRecall, current_query, masked_pool, render_recall
from integrations.swebench_ctx.report import sign_p


def history(n_steps: int) -> list[dict]:
    items = [{"role": "system", "content": "sys", "message_type": "system_prompt"},
             {"role": "user", "content": "issue", "message_type": "observation"}]
    for step in range(1, n_steps + 1):
        items.append({"role": "assistant", "content": f"t{step}", "thought": f"thought {step}", "action": f"cmd {step}",
                      "message_type": "action"})
        items.append({"role": "tool", "content": [{"type": "text", "text": f"out {step} line a\nout {step} line b"}],
                      "message_type": "observation"})
    return items


class StubLastN:
    """SWE-agent's LastNObservations: replaces observations before the last n (never the first)."""

    def __init__(self, n):
        self.n = n

    def __call__(self, items):
        obs = [i for i, e in enumerate(items) if e["message_type"] == "observation"]
        drop = set(obs[1:max(0, len(obs) - self.n)])
        out = []
        for i, e in enumerate(items):
            if i in drop:
                e = dict(e, content=[{"type": "text", "text": "Old environment output: (2 lines omitted)"}])
            out.append(e)
        return out


class TestRecall(unittest.TestCase):
    def test_pool_holds_masked_outputs_with_their_commands(self):
        items = history(4)
        pool = masked_pool(items, StubLastN(2)(items))
        self.assertEqual([p["action"] for p in pool], ["cmd 1", "cmd 2"])
        self.assertEqual(pool[0]["output"], "out 1 line a\nout 1 line b")

    def test_query_is_latest_thought_and_action(self):
        self.assertEqual(current_query(history(3)), "thought 3\ncmd 3")

    def test_render_puts_excerpts_under_commands(self):
        pool = [{"action": "cmd 1", "output": "x"}, {"action": "cmd 2", "output": "y"}]
        text = render_recall(pool, "[step 2, output]\nout 2 line b\n…\nmore")
        self.assertEqual(text, RECALL_HEADER + "\n\n$ cmd 2\nout 2 line b\n…\nmore")
        self.assertEqual(render_recall(pool, ""), "")

    def test_processor_appends_recall_to_newest_output_only(self):
        stub = types.ModuleType("sweagent.agent.history_processors")
        stub.LastNObservations = StubLastN
        sent = []

        class Reply:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self): return json.dumps({"text": "[step 1, output]\nout 1 line b", "units_in": 4, "units_kept": 1}).encode()

        def fake_urlopen(request, timeout):
            sent.append(json.loads(request.data))
            return Reply()

        items = history(4)
        with mock.patch.dict(sys.modules, {"sweagent": types.ModuleType("sweagent"), "sweagent.agent": types.ModuleType("sweagent.agent"),
                                           "sweagent.agent.history_processors": stub}), \
             mock.patch("integrations.swebench_ctx.recall.urllib.request.urlopen", fake_urlopen):
            out = CPCRecall(n=2, budget=100)(items)
        self.assertEqual(sent[0]["query"], "thought 4\ncmd 4")
        self.assertEqual([e["text"] for e in sent[0]["entries"]], ["out 1 line a\nout 1 line b", "out 2 line a\nout 2 line b"])
        self.assertTrue(out[-1]["content"][0]["text"].endswith("$ cmd 1\nout 1 line b"))
        self.assertEqual(out[:-1], StubLastN(2)(items)[:-1])  # everything before the newest output = masking
        self.assertNotIn(RECALL_HEADER, items[-1]["content"][0]["text"])  # the stored history is untouched

    def test_processor_falls_back_to_masking_on_error(self):
        stub = types.ModuleType("sweagent.agent.history_processors")
        stub.LastNObservations = StubLastN
        items = history(4)
        with mock.patch.dict(sys.modules, {"sweagent": types.ModuleType("sweagent"), "sweagent.agent": types.ModuleType("sweagent.agent"),
                                           "sweagent.agent.history_processors": stub}), \
             mock.patch("integrations.swebench_ctx.recall.urllib.request.urlopen", side_effect=OSError("down")):
            self.assertEqual(CPCRecall(n=2)(items), StubLastN(2)(items))

    def test_sign_test(self):
        self.assertEqual(sign_p(0, 0), 1.0)
        self.assertAlmostEqual(sign_p(5, 0), 0.0625)


if __name__ == "__main__":
    unittest.main()
