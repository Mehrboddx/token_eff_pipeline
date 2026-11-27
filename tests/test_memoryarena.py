"""MemoryArena: memory wrappers, the LiteLLM shim, the in-process loop, scoring."""

import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from integrations.memoryarena import llm_shim, memory, run_formal, score


def words(text):
    return len((text or "").split())


class Reply:
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def read(self): return self.body


class TestMemory(unittest.TestCase):
    def test_empty_and_none_use_the_benchmark_format(self):
        self.assertEqual(memory.NoMemory().wrap_user_prompt("solve x"),
                         "<memory_context>\nNone\n</memory_context>\nUser: solve x")
        compressed = memory.CompressedMemory("cpc", "http://x")
        self.assertEqual(compressed.wrap_user_prompt("q"), memory.wrap("", "q"))  # nothing stored, no call
        self.assertEqual(memory.memory_context(memory.wrap("a b", "q")), "a b")

    def test_compressed_memory_sends_sessions_and_refits(self):
        sent = []

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data)
            sent.append(payload)
            text = " ".join(["x"] * (payload["budget"] + 5))  # labels push it over
            return Reply(json.dumps({"text": text, "ms": 2, "units_in": 9, "units_kept": 3}).encode())

        mem = memory.CompressedMemory("random", "http://svc/", budget=40)
        mem.add_chunk("Initial result: Empty\n")
        mem.add_chunk("## Task: t1\n## solution: 42\n")
        with mock.patch.object(memory, "count_tokens", words), mock.patch.object(memory.urllib.request, "urlopen", fake_urlopen):
            wrapped = mem.wrap_user_prompt("next problem")
        self.assertEqual([p["budget"] for p in sent], [40, 27])
        self.assertEqual([e["step"] for e in sent[0]["entries"]], [1, 2])
        self.assertEqual(sent[0]["query"], "next problem")
        self.assertEqual(sent[0]["selector"], "random")
        self.assertFalse(sent[0]["dedupe"])
        self.assertLessEqual(words(memory.memory_context(wrapped)), 40)
        self.assertTrue(wrapped.endswith("User: next problem"))
        self.assertEqual(mem.last["units_kept"], 3)

    def test_head_conditions_send_headings_and_mark_excerpts(self):
        replies = iter([{"text": "kept line", "units_in": 9, "units_kept": 3}, {"text": "all", "units_in": 4, "units_kept": 4}])
        sent = []

        def fake_urlopen(request, timeout):
            sent.append(json.loads(request.data))
            return Reply(json.dumps(next(replies)).encode())

        mem = memory.make_memory("cpc_head", compress_url="http://svc")
        mem.add_chunk("a")
        with mock.patch.object(memory.urllib.request, "urlopen", fake_urlopen):
            cut = memory.memory_context(mem.wrap_user_prompt("q1"))
            whole = memory.memory_context(mem.wrap_user_prompt("q2"))
        self.assertTrue(sent[0]["headings"])
        self.assertNotIn("selector", sent[0])
        self.assertEqual(cut, memory.EXCERPT_NOTE + "\nkept line")
        self.assertEqual(whole, "all")  # nothing dropped: no note
        self.assertEqual(memory.make_memory("random_head", compress_url="u").selector, "random")

    def test_cpc_sends_no_selector(self):
        mem = memory.CompressedMemory("cpc", "http://svc")
        mem.add_chunk("a")
        sent = []
        with mock.patch.object(memory.urllib.request, "urlopen",
                               lambda request, timeout: sent.append(json.loads(request.data)) or Reply(b'{"text": "a"}')):
            mem.wrap_user_prompt("q")
        self.assertNotIn("selector", sent[0])
        self.assertEqual(sent[0]["budget"], memory.BUDGET)


class TestReadable(unittest.TestCase):
    def test_json_entries_are_laid_out_by_field_and_others_untouched(self):
        chunk = json.dumps({"name": "Eric", "scratchpad": [{"step_idx": 0, "tool_results": [
            {"name": "AccommodationSearch", "result": "NAME  price\nCozy Loft  120\nBig House  300"}]}],
            "final_plan": "=== Eric's Plan ===\nDay 1:"})
        text = memory.readable(chunk)
        self.assertIn("name: Eric", text)
        self.assertIn("\n          | Cozy Loft  120\n", text)  # a table row is its own line, in the value's block
        self.assertIn("final_plan:\n  | === Eric's Plan ===\n  | Day 1:", text)
        for other in ("## Task: t\n## solution: 4\n", "Initial result: Empty\n", "42", ""):
            self.assertEqual(memory.readable(other), other)


class TestShim(unittest.TestCase):
    def test_forwarding_reasoning_and_thought_signatures(self):
        calls = []

        class Response:
            def __init__(self, body): self.body = body
            def model_dump(self): return self.body

        def completion(**kwargs):
            calls.append(json.loads(json.dumps(kwargs)))
            return Response({"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "reasoning", "arguments": "{}"},
                 "provider_specific_fields": {"thought_signature": "sig"}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3}})

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"litellm": types.SimpleNamespace(completion=completion)}):
            usage = Path(tmp) / "u.jsonl"
            shim = llm_shim.Shim({"agent": {"model": "vertex_ai/m", "reasoning_effort": "low"}}, str(usage))
            shim.complete({"model": "agent|cpc", "messages": [{"role": "user", "content": "hi"}],
                           "temperature": 0.0, "max_completion_tokens": 99, "tools": [{"type": "function"}]})
            # next turn: the OpenAI client sends the call back without provider fields
            shim.complete({"model": "agent|cpc", "messages": [
                {"role": "assistant", "content": "x", "tool_calls": [{"id": "c1", "type": "function", "function": {}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "r"}]})
            logged = [json.loads(line) for line in open(usage)]
        self.assertEqual(calls[0]["model"], "vertex_ai/m")
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(calls[0]["max_tokens"], 99)
        self.assertNotIn("temperature", calls[0])
        self.assertEqual(calls[1]["max_tokens"], 8192)
        self.assertEqual(calls[1]["messages"][0]["tool_calls"][0]["provider_specific_fields"], {"thought_signature": "sig"})
        self.assertEqual([(r["tag"], r["prompt_tokens"]) for r in logged], [("cpc", 10), ("cpc", 10)])

    def test_unknown_role_is_an_error(self):
        with self.assertRaises(ValueError):
            llm_shim.Shim({"agent": {"model": "m"}}).resolve("judge|x")


class FakeAgent:
    def __init__(self, **kwargs):
        self.prompts = []

    def build_prompt(self, task, background=None):
        return f"### PROBLEM: {task}"

    def act(self, prompt):
        self.prompts.append(prompt)
        return {"type": "final", "answer": "ans:" + prompt[-3:], "tool_trace": [{"tool": "reasoning"}], "input": prompt}

    def build_memory_entry(self, task, action, observation=None, reward=None):
        return f"## Task: {task}\n## solution: {action['answer']}\n"


class FakeEnv:
    def __init__(self, config):
        self.config = config

    def reset(self):
        return {"step": 0}

    def step(self, action, ground_truth=None, need_judge=False):
        return {"final": action["answer"], "judge_result": "yes"}, ground_truth == "good", {}


class FakeLongContext:
    def __init__(self):
        self.chunks = []

    def add_chunk(self, chunk):
        self.chunks.append(chunk)

    def wrap_user_prompt(self, prompt):
        return memory.wrap("\n".join(self.chunks), prompt)


class TestLoop(unittest.TestCase):
    def test_memory_carries_earlier_sessions_into_later_prompts(self):
        up = types.SimpleNamespace(MathAgent=FakeAgent, MathEnvironment=FakeEnv, LongContextMemorySystem=FakeLongContext)
        agents = []
        up.MathAgent = lambda **kw: agents.append(FakeAgent(**kw)) or agents[-1]
        task = {"domain": "math", "paper_name": "p", "questions": ["q1", "q2"], "answers": ["good", "bad"],
                "backgrounds": ["", ""]}
        args = types.SimpleNamespace(agent_max_tokens=10, compress_url="", budget=100)
        row = run_formal.run_task(task, "long_context", up, "http://shim/v1", args)
        first, second = agents[0].prompts
        self.assertIn("Initial result: Empty", first)
        self.assertIn("## Task: q1", second)  # the first session is in memory for the second
        self.assertEqual([s["correct"] for s in row["subtasks"]], [True, False])
        self.assertEqual(row["subtasks"][0]["memory_raw_tokens"], memory.count_tokens("Initial result: Empty\n"))

    def test_none_condition_shows_no_memory(self):
        agents = []
        up = types.SimpleNamespace(MathAgent=lambda **kw: agents.append(FakeAgent(**kw)) or agents[-1], MathEnvironment=FakeEnv)
        task = {"domain": "phys", "paper_name": "p", "questions": ["q1", "q2"], "answers": ["good", "good"], "backgrounds": [None, None]}
        row = run_formal.run_task(task, "none", up, "u", types.SimpleNamespace(agent_max_tokens=1, compress_url="", budget=1))
        self.assertNotIn("q1", agents[0].prompts[1])
        self.assertEqual([s["memory_tokens"] for s in row["subtasks"]], [0, 0])


class FakeTravelAgent:
    def __init__(self, model_name, max_steps):
        self.contexts, self.base, self.last_result = [], None, None

    def reset(self): pass
    def set_base_person(self, name, query, plan): self.base = plan

    def prepare_for_person(self, name, round_idx, include_previous_plans, memory_context, memory_system):
        self.contexts.append((name, include_previous_plans, memory_context))

    def act(self, query):
        name = self.contexts[-1][0]
        self.last_result = types.SimpleNamespace(total_steps=3, success=True, error_message=None)
        return f"=== {name}'s Plan ===\nDay 1:\nAccommodation: Loft, X"

    def build_memory_entry(self, task, action, observation=None, reward=None):
        return json.dumps({"query": task, "final_plan": action})


def fake_travel_upstream(agents):
    from integrations.memoryarena import run_travel  # noqa: F401

    def parse_all_plans(text, queries):
        return [{"person_idx": i, "name": q["name"], "result": text} for i, q in enumerate(queries, 1)]

    evaluation = types.SimpleNamespace(
        find_constraint_slots=lambda gt, base, gid, idx: {(1, "accommodation")},
        check_slot_pass=lambda gt_plan, plan, day, slot: plan == "parsed",
        check_person_full_pass=lambda gt_plan, plan: plan == "parsed")
    return types.SimpleNamespace(
        TravelPlannerAgent=lambda **kw: agents.append(FakeTravelAgent(**kw)) or agents[-1],
        format_person_plan=lambda name, days: f"=== {name}'s Plan ===",
        parse_all_plans=parse_all_plans, parse_plan_text=lambda text: "parsed" if "Loft" in text else None,
        evaluation=evaluation, LongContextMemorySystem=FakeLongContext)


class TestTravelLoop(unittest.TestCase):
    GROUP = {"id": 7, "base_person": {"name": "Ann", "query": "I am Ann.", "daily_plans": []},
             "questions": [{"round_idx": 1, "name": "Bob", "query": "I am Bob. Join Ann."},
                           {"round_idx": 2, "name": "Cy", "query": "I am Cy. Stay with Bob."}],
             "answers": [{"round_idx": 1, "daily_plans": []}, {"round_idx": 2, "daily_plans": []}]}
    ARGS = types.SimpleNamespace(max_steps=30, compress_url="", budget=100)

    def test_memory_holds_the_base_plan_and_earlier_travelers(self):
        from integrations.memoryarena import run_travel

        agents = []
        row = run_travel.run_group(self.GROUP, "long_context", fake_travel_upstream(agents), self.ARGS)
        (_, prev1, ctx1), (_, prev2, ctx2) = agents[0].contexts
        self.assertFalse(prev1)  # with a memory system, no in-prompt plan history
        self.assertIn("Ann's Plan", ctx1)
        self.assertIn("Bob. Join Ann", ctx2)  # Bob's entry is in memory for Cy
        self.assertEqual((row["ps"], row["sr"], row["sps"]), (1.0, 1.0, 1.0))
        self.assertEqual([r["steps"] for r in row["rounds"]], [3, 3])

    def test_none_keeps_their_in_prompt_history(self):
        from integrations.memoryarena import run_travel

        agents = []
        row = run_travel.run_group(self.GROUP, "none", fake_travel_upstream(agents), self.ARGS)
        self.assertEqual(agents[0].base, "=== Ann's Plan ===")
        self.assertEqual([(prev, ctx) for _, prev, ctx in agents[0].contexts], [(True, None), (True, None)])
        self.assertEqual([r["memory_tokens"] for r in row["rounds"]], [0, 0])


class TestScore(unittest.TestCase):
    def test_paired_tasks_and_metrics(self):
        def row(task, condition, correct):
            return {"domain": "math", "task": task, "condition": condition, "n": len(correct),
                    "subtasks": [{"correct": c, "memory_tokens": 5} for c in correct]}

        self.assertEqual(score.task_metrics(row("t", "cpc", [True, False, True])), {"ps": 2 / 3, "sr": 1.0, "correct": 2, "n": 3})
        rows = [row("a", "cpc", [True, True]), row("a", "random", [True, False]), row("b", "cpc", [False, True])]
        out = io.StringIO()
        with redirect_stdout(out):
            score.report(rows, ["cpc", "random"], 50, "all")
        text = out.getvalue()
        self.assertIn("1 tasks, 2 subtasks", text)  # task b lacks a random run
        self.assertIn("cpc - random PS: +0.500", text)


if __name__ == "__main__":
    unittest.main()
