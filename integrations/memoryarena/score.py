"""Score MemoryArena formal-reasoning runs, paired across memory conditions.

Per task, as in MemoryArena's eval (env/env_systems/formal_reasoning_env/eval.py):
  PS (progress score) = fraction of its subtasks judged correct;
  SR (success)        = whether its last subtask is correct.
Also the pooled subtask pass rate. Only tasks run under every listed condition count.
95% CIs and paired differences come from a bootstrap over tasks.
Cost uses the usage log and list prices per million tokens (input, output).
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from integrations.memoryarena.memory import CONDITIONS

PAIRS = [("cpc_skel", "cpc"), ("cpc_skel", "long_context"), ("cpc_skel", "random_skel"), ("cpc_skel", "none"),
         ("cpc_head", "random_head"), ("cpc_head", "cpc"), ("random_head", "random"), ("cpc_head", "long_context"),
         ("cpc_head", "none"), ("cpc", "random"), ("cpc", "bm25"), ("cpc", "long_context"), ("cpc", "none"),
         ("random", "none"), ("bm25", "none"), ("long_context", "none")]
PRICES = {"vertex_ai/gemini-3.5-flash-lite": (0.30, 2.50),
          "vertex_ai/gemini-3.6-flash": (0.75, 3.75)}  # introductory until 31 Dec 2026


def task_metrics(row: dict) -> dict:
    subtasks = row["subtasks"]
    correct = [bool(s["correct"]) for s in subtasks]
    return {"ps": sum(correct) / len(correct), "sr": float(correct[-1]), "correct": sum(correct), "n": len(correct)}


def bootstrap(keys, values_by_key, resamples: int, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values_by_key[rng.choice(keys)] for _ in keys]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def report(rows: list[dict], conditions: list[str], resamples: int, title: str) -> None:
    by = {(r["domain"], r["task"], r["condition"]): r for r in rows}
    keys = sorted({(d, t) for d, t, _ in by if all((d, t, c) in by for c in conditions)})
    if not keys:
        print(f"== {title}: no task run under every condition")
        return
    subtasks = sum(by[(*k, conditions[0])]["n"] for k in keys)
    print(f"== {title}: {len(keys)} tasks, {subtasks} subtasks, run under every condition")
    print(f"   {'condition':<13} {'PS [95% CI]':<24} {'SR [95% CI]':<24} {'subtask pass':<13} memory tokens (median, max)")
    metrics = {c: {k: task_metrics(by[(*k, c)]) for k in keys} for c in conditions}
    for c in conditions:
        ps = {k: m["ps"] for k, m in metrics[c].items()}
        sr = {k: m["sr"] for k, m in metrics[c].items()}
        pooled = sum(m["correct"] for m in metrics[c].values()) / subtasks
        shown = [s["memory_tokens"] for k in keys for s in by[(*k, c)]["subtasks"]]
        lo, hi = bootstrap(keys, ps, resamples)
        slo, shi = bootstrap(keys, sr, resamples)
        print(f"   {c:<13} {sum(ps.values()) / len(keys):.3f} [{lo:.3f}, {hi:.3f}]     "
              f"{sum(sr.values()) / len(keys):.3f} [{slo:.3f}, {shi:.3f}]     {pooled:.3f}         "
              f"{statistics.median(shown):.0f}, {max(shown)}")
    for a, b in PAIRS:
        if a in conditions and b in conditions:
            for name in ("ps", "sr"):
                diff = {k: metrics[a][k][name] - metrics[b][k][name] for k in keys}
                lo, hi = bootstrap(keys, diff, resamples)
                print(f"   {a} - {b} {name.upper()}: {sum(diff.values()) / len(keys):+.3f} [{lo:+.3f}, {hi:+.3f}]")


def cost(usage_path: Path) -> None:
    if not usage_path or not usage_path.exists():
        return
    totals = defaultdict(lambda: [0, 0, 0, 0])  # calls, prompt, completion, errors
    for line in open(usage_path, encoding="utf-8"):
        row = json.loads(line)
        total = totals[(row["tag"], row["role"], row["model"])]
        total[0] += 1
        total[1] += row["prompt_tokens"]
        total[2] += row["completion_tokens"]
        total[3] += bool(row["error"])
    print("== usage")
    grand = 0.0
    for (tag, role, model), (calls, prompt, completion, errors) in sorted(totals.items()):
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        dollars = prompt / 1e6 * price_in + completion / 1e6 * price_out
        grand += dollars
        print(f"   {tag:<13} {role:<6} calls {calls:>6}  in {prompt:>11,}  out {completion:>10,}  errors {errors:>3}  ${dollars:.2f}")
    print(f"   total ${grand:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--usage", type=Path, default=None)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.results, encoding="utf-8")]
    conditions = [c for c in args.conditions.split(",") if c]
    report(rows, conditions, args.resamples, "all")
    for domain in sorted({r["domain"] for r in rows}):
        report([r for r in rows if r["domain"] == domain], conditions, args.resamples, domain)
    cost(args.usage)


if __name__ == "__main__":
    main()
