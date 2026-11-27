"""MemoryArena group travel planning: the benchmark's loop (run_travel.py), run in-process.

Per group: the base traveler's fixed plan goes into memory (or, with no memory
system, straight into the agent's prompt), then each joining traveler in turn:
    wrapped = memory.wrap_user_prompt(query)          # retrieval with the traveler's query
    agent.prepare_for_person(..., memory_context=<the memory block>)
    plan = agent.act(query)                           # their ReAct agent over the CSV tools
    memory.add_chunk(agent.build_memory_entry(query, plan, ...))   # query, tool calls and results, plan
Settings as in their configs: max 30 steps, judgement mode "none" (no feedback between
travelers), no step memory. "none" is their no-memory-system path: the agent keeps the
base plan, all queries and all earlier final plans in its own prompt.

Scoring uses their eval.py functions on the final parsed plans: PS (traveler passes every
slot), SPS (share of the slots that differ from the base plan met, mean per group), SR
(every traveler in the group passes). One JSON line per (group, condition).
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from integrations.memoryarena import llm_shim, upstream
from integrations.memoryarena.memory import BUDGET, CONDITIONS, HEAD_CONDITIONS, count_tokens, make_memory, memory_context


def shared_tools(up) -> None:
    """Every agent builds a ToolExecutor that reads all the CSVs (flights alone is millions of
    rows); build it once and hand the same read-only instance to every agent."""
    real = up.agent_module.ToolExecutor
    lock = threading.Lock()
    instance = {}

    def executor(db_path=None):
        with lock:
            if "tools" not in instance:
                instance["tools"] = real(db_path=db_path)
            return instance["tools"]

    up.agent_module.ToolExecutor = executor


def score_group(up, group: dict, final: list[dict]) -> dict:
    ev = up.evaluation
    gid = group["id"]
    gt = {(gid, a["round_idx"]): a["daily_plans"] for a in group["answers"]}
    base_plans = {gid: group["base_person"]["daily_plans"]}
    persons = []
    for person in final:
        idx = person["person_idx"]
        plan = up.parse_plan_text(person.get("result", ""))
        gt_plan = gt.get((gid, idx))
        slots = ev.find_constraint_slots(gt, base_plans, gid, idx)
        passed = sum(ev.check_slot_pass(gt_plan, plan, day, slot) for day, slot in slots)
        persons.append({"person_idx": idx, "name": person.get("name"), "pass": bool(ev.check_person_full_pass(gt_plan, plan)),
                        "constraint_rate": passed / len(slots) if slots else None})
    rates = [p["constraint_rate"] for p in persons if p["constraint_rate"] is not None]
    return {"persons": persons, "ps": sum(p["pass"] for p in persons) / len(persons),
            "sps": sum(rates) / len(rates) if rates else None, "sr": float(all(p["pass"] for p in persons))}


def run_group(group: dict, condition: str, up, args) -> dict:
    agent = up.TravelPlannerAgent(model_name=f"agent|{condition}", max_steps=args.max_steps)
    agent.reset()
    memory = None if condition == "none" else make_memory(condition, up, args.compress_url, args.budget)
    added: list[str] = []

    def add(chunk: str) -> None:
        memory.add_chunk(chunk)
        added.append(chunk)

    base = group["base_person"]
    base_plan = up.format_person_plan(base["name"], base["daily_plans"])
    accumulated = base_plan
    base_entry = None
    if memory is not None:
        base_entry = json.dumps({"name": base["name"], "query": base["query"], "is_base_person": True,
                                 "final_plan": base_plan}, ensure_ascii=False)
        add(base_entry)
    else:
        agent.set_base_person(base["name"], base["query"], base_plan)

    queries = group["questions"]
    all_results, rounds = [], []
    for item in queries:
        start = time.time()
        context = None
        compress = {}
        if memory is not None:
            wrapped = memory.wrap_user_prompt(item["query"])
            context = (wrapped.split("</memory_context>")[0] + "</memory_context>").strip()
            compress = dict(getattr(memory, "last", {}) or {})
        agent.prepare_for_person(name=item["name"], round_idx=item["round_idx"],
                                 include_previous_plans=memory is None, memory_context=context, memory_system=None)
        plan = agent.act(item["query"])
        accumulated += f"\n\n{plan}" if accumulated else plan
        result = agent.last_result
        all_results.append(up.parse_all_plans(accumulated, queries[:item["round_idx"]]))
        shown = memory_context(context) if context else ""
        rounds.append({"round": item["round_idx"], "name": item["name"], "steps": result.total_steps,
                       "finished": result.success, "error": result.error_message,
                       "memory_tokens": count_tokens(shown) if shown and shown != "None" else 0,
                       "memory_raw_tokens": count_tokens("\n".join(added)), "compress": compress,
                       "plan": plan, "seconds": round(time.time() - start, 1)})
        if memory is not None:
            entry = agent.build_memory_entry(task=item["query"], action=plan,
                                             observation={"final": plan, "judgement": None}, reward=None)
            add(entry)
            if getattr(args, "log_memory", False):
                rounds[-1].update(memory_shown=context, memory_entry=entry)
    row = {"group": group["id"], "condition": condition, "n": len(queries), "rounds": rounds,
           **score_group(up, group, all_results[-1])}
    if getattr(args, "log_memory", False):
        row["base_entry"] = base_entry
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="group_travel_planner data.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--conditions", default=",".join(CONDITIONS + HEAD_CONDITIONS))
    parser.add_argument("--groups", default="", help="comma-separated group ids")
    parser.add_argument("--sample", type=int, default=0, help="N groups drawn with --seed (if no --groups)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--compress-url", default="http://127.0.0.1:8765")
    parser.add_argument("--agent-model", default="vertex_ai/gemini-3.5-flash-lite")
    parser.add_argument("--agent-reasoning", default="low")
    parser.add_argument("--log-memory", action="store_true",
                        help="keep each traveler's memory entry and the memory shown (for diagnose_travel)")
    args = parser.parse_args()

    up = upstream.load_travel(args.repo)
    shared_tools(up)
    shim = llm_shim.Shim({"agent": {"model": args.agent_model, "reasoning_effort": args.agent_reasoning or None}},
                         usage_path=str(args.usage))
    _, shim_url = llm_shim.serve(shim)
    os.environ["OPENAI_API_BASE"] = shim_url  # their OpenAIClient reads these
    os.environ.setdefault("OPENAI_API_KEY", "unused")

    groups = [up.convert_row(json.loads(line)) for line in open(args.data, encoding="utf-8")]
    groups.sort(key=lambda g: g["id"])
    if args.groups:
        wanted = {int(x) for x in args.groups.split(",") if x}
        groups = [g for g in groups if g["id"] in wanted]
    elif args.sample:
        groups = sorted(random.Random(args.seed).sample(groups, args.sample), key=lambda g: g["id"])
    conditions = [c for c in args.conditions.split(",") if c]
    done = set()
    if args.out.exists():
        done = {(r["group"], r["condition"]) for r in map(json.loads, open(args.out, encoding="utf-8"))}
    jobs = [(g, c) for g in groups for c in conditions if (g["id"], c) not in done]
    print(f"groups {[g['id'] for g in groups]} x {len(conditions)} conditions: {len(jobs)} to run, {len(done)} done", flush=True)

    lock = threading.Lock()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_group, g, c, up, args): (g, c) for g, c in jobs}
        for k, future in enumerate(as_completed(futures), 1):
            group, condition = futures[future]
            try:
                row = future.result()
            except Exception:
                failures += 1
                print(f"[{k}/{len(jobs)}] FAILED {condition} group {group['id']}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                continue
            with lock, open(args.out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"[{k}/{len(jobs)}] {condition} group {row['group']} PS {row['ps']:.2f} SR {row['sr']:.0f} "
                  f"steps {sum(r['steps'] for r in row['rounds'])}", flush=True)
    print(f"done; {failures} failed", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
