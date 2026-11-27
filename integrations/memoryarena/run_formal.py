"""MemoryArena formal reasoning (math, physics): the benchmark's loop, run in-process.

For each task (a chain of subtasks over one paper), memory starts empty and, per
subtask, exactly as in MemoryArena's run_math.py:
    query  = agent.build_prompt(subtask, background)
    prompt = memory.wrap_user_prompt(query)
    action = agent.act(prompt)                                 # their MathAgent (tool loop)
    obs, reward = env.step(action, ground_truth, need_judge)   # their MathEnvironment judge
    memory.add_chunk(agent.build_memory_entry(subtask, action, obs, reward=None))
Their run goes through HTTP servers (environment, memory); here the same objects
are called directly, and their OpenAI client calls go to the LiteLLM shim
(integrations/memoryarena/llm_shim.py) for Vertex Gemini.

One JSON line per (domain, task, condition) with every subtask's result; a rerun
skips what is done. Usage (tokens per call, tagged by condition) goes to --usage.
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from integrations.memoryarena import llm_shim, upstream
from integrations.memoryarena.memory import BUDGET, CONDITIONS, count_tokens, make_memory, memory_context

DOMAINS = ("math", "phys")


def load_tasks(data_dir: Path, domains, limit: int = 0, names=()) -> list[dict]:
    tasks = []
    for domain in domains:
        rows = [json.loads(line) for line in open(data_dir / f"formal_reasoning_{domain}.jsonl", encoding="utf-8")]
        rows.sort(key=lambda row: str(row["paper_name"]))
        if names:
            rows = [row for row in rows if row["paper_name"] in names]
        if limit:
            rows = rows[:limit]
        tasks.extend({**row, "domain": domain} for row in rows)
    return tasks


def run_task(task: dict, condition: str, up, shim_url: str, args) -> dict:
    agent = up.MathAgent(model_name=f"agent|{condition}", temperature=0.0, max_tokens=args.agent_max_tokens,
                         backend="openai", base_url=shim_url)
    env = up.MathEnvironment(config={"model_name": f"judge|{condition}", "temperature": 1.0, "max_tokens": 4096,
                                     "backend": "openai", "base_url": shim_url, "max_steps": 10})
    memory = make_memory(condition, up, args.compress_url, args.budget)
    added: list[str] = []

    def add(chunk: str) -> None:
        memory.add_chunk(chunk)
        added.append(chunk)

    if env.reset() is not None:
        add("Initial result: Empty\n")
    subtasks = []
    for index, (question, answer, background) in enumerate(zip(task["questions"], task["answers"], task["backgrounds"])):
        start = time.time()
        query = agent.build_prompt(task=question, background=background)
        prompt = memory.wrap_user_prompt(query)
        compress = dict(getattr(memory, "last", {}) or {})
        action = agent.act(prompt)
        observation, reward, _ = env.step(action, ground_truth=answer, need_judge=True)
        add(agent.build_memory_entry(task=question, observation=observation, action=action, reward=None))
        shown = memory_context(prompt)
        subtasks.append({
            "index": index,
            "correct": bool(reward),
            "judge": observation.get("judge_result"),
            "response": observation.get("final"),
            "memory_tokens": count_tokens(shown) if shown != "None" else 0,
            "memory_raw_tokens": count_tokens("\n".join(added[:-1])),  # everything stored before this subtask
            "compress": compress,
            "tool_calls": len(action.get("tool_trace") or []) if isinstance(action, dict) else 0,
            "seconds": round(time.time() - start, 1),
        })
    return {"domain": task["domain"], "task": task["paper_name"], "condition": condition,
            "n": len(subtasks), "subtasks": subtasks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--domains", default="math,phys")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--limit", type=int, default=0, help="first N tasks per domain (by paper name)")
    parser.add_argument("--tasks", default="", help="comma-separated paper names")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--compress-url", default="http://127.0.0.1:8765")
    parser.add_argument("--agent-model", default="vertex_ai/gemini-3.5-flash-lite")
    parser.add_argument("--agent-reasoning", default="low")
    parser.add_argument("--agent-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-model", default="vertex_ai/gemini-3.6-flash")
    parser.add_argument("--judge-reasoning", default="minimal")
    args = parser.parse_args()

    up = upstream.load(args.repo)
    shim = llm_shim.Shim({"agent": {"model": args.agent_model, "reasoning_effort": args.agent_reasoning or None},
                          "judge": {"model": args.judge_model, "reasoning_effort": args.judge_reasoning or None}},
                         usage_path=str(args.usage))
    _, shim_url = llm_shim.serve(shim)
    os.environ["OPENAI_BASE_URL"] = shim_url  # MathAgent's tool loop reads these directly
    os.environ.setdefault("OPENAI_API_KEY", "unused")

    tasks = load_tasks(args.data_dir, [d for d in args.domains.split(",") if d],
                       args.limit, set(filter(None, args.tasks.split(","))))
    conditions = [c for c in args.conditions.split(",") if c]
    done = set()
    if args.out.exists():
        for line in open(args.out, encoding="utf-8"):
            row = json.loads(line)
            done.add((row["domain"], row["task"], row["condition"]))
    jobs = [(task, c) for task in tasks for c in conditions if (task["domain"], task["paper_name"], c) not in done]
    print(f"{len(tasks)} tasks x {len(conditions)} conditions: {len(jobs)} to run, {len(done)} done", flush=True)

    lock = threading.Lock()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, task, c, up, shim_url, args): (task, c) for task, c in jobs}
        for k, future in enumerate(as_completed(futures), 1):
            task, condition = futures[future]
            try:
                row = future.result()
            except Exception:
                failures += 1
                print(f"[{k}/{len(jobs)}] FAILED {condition} {task['domain']} {task['paper_name']}\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
                continue
            with lock, open(args.out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            correct = sum(s["correct"] for s in row["subtasks"])
            print(f"[{k}/{len(jobs)}] {condition} {row['domain']} {row['task'][:40]} {correct}/{row['n']} ok", flush=True)
    print(f"done; {failures} failed", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
