"""Why does a memory condition help or not on group travel? Diagnostics on runs made with
run_travel.py --log-memory (no agent calls; the compressor service for the replays).

For every traveler, the slots their plan must get right that differ from the base plan
(their eval.py's constraint slots) and the value each needs (a restaurant, hotel or
attraction name, a flight number). For each such value:
  - is it anywhere in memory at all (else it must come from a fresh tool search),
    and is it in an earlier traveler's final plan;
  - is it in the memory the agent was shown in the run;
  - replayed on the same memory: is it in what cpc, random, bm25 and long_context keep.
Also where the 6k of cpc and random goes: earlier final plans, queries, or other lines
(tool results, thoughts), and whether a slot came out right when its value was shown.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from integrations.memoryarena import upstream
from integrations.memoryarena.memory import BUDGET, CompressedMemory, count_tokens, memory_context, readable


def needed_items(slot: str, value: str) -> list[str]:
    value = (value or "").strip()
    if not value or value == "-":
        return []
    if slot == "transportation":
        flights = re.findall(r"F\d{5,}", value)
        if flights:
            return flights
        return [mode for mode in ("self-driving", "taxi") if mode in value.lower()]
    parts = value.split(";") if slot == "attraction" else [value]
    return [part.rsplit(",", 1)[0].strip().lower() for part in parts if part.strip() and part.strip() != "-"]


def contains(text: str, item: str) -> bool:
    return item.lower() in (text or "").lower()


def classify(context: str, plan_lines: set, query_lines: set) -> Counter:
    tokens = Counter()
    for line in context.splitlines():
        text = line.strip()
        if not text or text.startswith("[step "):
            continue
        kind = "plans" if text in plan_lines else "queries" if text in query_lines else "other"
        tokens[kind] += count_tokens(text)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--compress-url", default="http://127.0.0.1:8765")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    up = upstream.load_travel(args.repo)
    ev = up.evaluation
    groups = {g["id"]: g for g in (up.convert_row(json.loads(line)) for line in open(args.data, encoding="utf-8"))}
    rows = [r for r in map(json.loads, open(args.results, encoding="utf-8")) if r.get("base_entry")]

    replay = ["cpc", "random", "bm25", "long_context"]
    found = defaultdict(lambda: [0, 0])  # (scope, who) -> [present, total]
    budget_use = {s: Counter() for s in ("cpc", "random")}
    slot_ok = defaultdict(lambda: [0, 0])  # (condition, shown?) -> [correct, total]
    for row in rows:
        group = groups[row["group"]]
        gid = group["id"]
        gt = {(gid, a["round_idx"]): a["daily_plans"] for a in group["answers"]}
        base_plans = {gid: group["base_person"]["daily_plans"]}
        chunks = [row["base_entry"]]
        plans = [json.loads(row["base_entry"])["final_plan"]]
        for i, item in enumerate(group["questions"]):
            run_round = row["rounds"][i]
            raw = "\n".join(readable(c) for c in chunks)
            plan_text = "\n".join(plans)
            contexts = {"shown in run": memory_context(run_round.get("memory_shown") or "")}
            for selector in replay:
                if selector in ("cpc", "random"):
                    memory = CompressedMemory(selector, args.compress_url, args.budget)
                elif selector == "bm25":
                    memory = up.RAGMemorySystem(retrieval_method="bm25")
                else:
                    memory = up.LongContextMemorySystem()
                for chunk in chunks:
                    memory.add_chunk(chunk)
                contexts[selector] = memory_context(memory.wrap_user_prompt(item["query"]))
            plan_lines = {line.strip() for text in plans for line in text.splitlines() if line.strip()}
            query_lines = {line.strip() for g_item in group["questions"][:i + 1] for line in g_item["query"].splitlines()}
            for selector in ("cpc", "random"):
                budget_use[selector].update(classify(contexts[selector], plan_lines, query_lines))

            own = up.parse_all_plans(run_round["plan"], [item])[0]["result"]
            parsed = up.parse_plan_text(own)
            gt_plan = gt[(gid, item["round_idx"])]
            for day, slot in sorted(ev.find_constraint_slots(gt, base_plans, gid, item["round_idx"])):
                day_plan = ev.get_day(gt_plan, day) or {}
                items = needed_items(slot, day_plan.get(slot, "-"))
                if not items:
                    continue
                in_memory = all(contains(raw, x) for x in items)
                in_plans = all(contains(plan_text, x) for x in items)
                scope = "in an earlier plan" if in_plans else "in memory (tool output only)" if in_memory else "not in memory"
                found[(scope, "all")][1] += 1
                for who, text in contexts.items():
                    hit = all(contains(text, x) for x in items)
                    found[(scope, who)][0] += hit
                    found[(scope, who)][1] += 1
                shown = all(contains(contexts["shown in run"], x) for x in items)
                ok = ev.check_slot_pass(gt_plan, parsed, day, slot)
                slot_ok[(row["condition"], scope, shown)][0] += ok
                slot_ok[(row["condition"], scope, shown)][1] += 1
            chunks.append(run_round["memory_entry"])
            plans.append(run_round["plan"])

    print(f"{len(rows)} runs ({Counter(r['condition'] for r in rows)}), budget {args.budget}")
    print("\n== needed values (constraint slots), by where they are in memory; share kept by each memory")
    for scope in ("in an earlier plan", "in memory (tool output only)", "not in memory"):
        total = found[(scope, "all")][1]
        if not total:
            continue
        cells = "  ".join(f"{who} {found[(scope, who)][0] / total:.2f}"
                          for who in ["shown in run"] + replay)
        print(f"   {scope:<30} n={total:<4} {cells}")
    print("\n== where the budget goes (tokens kept, replay)")
    for selector, use in budget_use.items():
        total = sum(use.values()) or 1
        print(f"   {selector:<7} " + "  ".join(f"{k} {use[k] / total:.2f}" for k in ("plans", "queries", "other")))
    print("\n== slot right in the run, by whether its value was in the memory shown")
    for (condition, scope, shown), (ok, n) in sorted(slot_ok.items()):
        print(f"   {condition:<13} {scope:<30} shown={str(shown):<5} {ok}/{n} = {ok / n:.2f}")


if __name__ == "__main__":
    main()
