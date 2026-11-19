"""Compare CooperBench arms: does compression change success, and how many tokens does it save?

Reads, for each run, CooperBench's logs/<run>/coop/<repo>/<task>/<pair>/
eval.json (both features' tests on the merged patches) and the agents'
ledgers written by integrations/cooperbench/agent.py. The first run is the
baseline; every other run is compared with it on the pairs both completed:
exact McNemar test on pass/fail, and a paired bootstrap CI on the change in
tokens per pair. Standard library only, so it runs wherever CooperBench does.

    python -m integrations.cooperbench.report --logs logs flash_none flash_messages flash_full
"""

import argparse
import json
import random
import statistics
from math import comb
from pathlib import Path

# USD per million tokens (Vertex AI, gemini-3-flash-preview, standard tier); override with flags.
PRICE_INPUT, PRICE_CACHED, PRICE_OUTPUT = 0.50, 0.05, 3.00


def load_pair(pair_dir: Path) -> dict | None:
    eval_file = pair_dir / "eval.json"
    if not eval_file.exists():
        return None
    result = json.loads(eval_file.read_text())
    row = {
        "passed": bool(result.get("both_passed")),
        "feature_passed": sum(bool((result.get(key) or {}).get("passed")) for key in ("feature1", "feature2")),
        "step_in": 0,
        "step_out": 0,
        "summary_in": 0,
        "summary_out": 0,
        "cached": 0,
        "reasoning": 0,
        "steps": 0,
        "summaries": 0,
        "messages": 0,
        "message_in": 0,
        "message_out": 0,
        "histories": 0,
        "history_in": 0,
        "history_out": 0,
        "compress_ms": 0.0,
        "errors": 0,
    }
    for ledger in pair_dir.glob("*_ledger.jsonl"):
        for line in ledger.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            kind = event.get("kind")
            if event["event"] == "llm":
                prefix = "step" if kind == "step" else "summary"
                row[f"{prefix}_in"] += event["prompt_tokens"]
                row[f"{prefix}_out"] += event["completion_tokens"]
                row["cached"] += event["cached_tokens"]
                row["reasoning"] += event["reasoning_tokens"]
                row["steps" if kind == "step" else "summaries"] += 1
            elif event["event"] == "compress":
                prefix = "message" if kind == "message" else "history"
                row["messages" if kind == "message" else "histories"] += 1
                row[f"{prefix}_in"] += event["tokens_in"] or 0
                row[f"{prefix}_out"] += event["tokens_out"] or 0
                row["compress_ms"] += event["ms"] or 0
            elif event["event"] == "compress_error":
                row["errors"] += 1
    row["input"] = row["step_in"] + row["summary_in"]
    row["output"] = row["step_out"] + row["summary_out"]
    row["total"] = row["input"] + row["output"]
    return row


def load_run(logs: Path, run: str) -> dict:
    pairs = {}
    for eval_file in sorted((logs / run).glob("coop/*/*/*/eval.json")):
        pair_dir = eval_file.parent
        row = load_pair(pair_dir)
        if row is not None:
            pairs[str(pair_dir.relative_to(logs / run))] = row
    return pairs


def cost(row: dict, prices: tuple[float, float, float]) -> float:
    price_in, price_cached, price_out = prices
    uncached = row["input"] - row["cached"]
    return (uncached * price_in + row["cached"] * price_cached + row["output"] * price_out) / 1e6


def mcnemar(base: list[bool], variant: list[bool]) -> tuple[int, int, float]:
    better = sum(v and not b for b, v in zip(base, variant))
    worse = sum(b and not v for b, v in zip(base, variant))
    n, k = better + worse, min(better, worse)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return better, worse, p


def bootstrap_ratio(base: list[float], variant: list[float], samples: int = 10000, seed: int = 0):
    """Ratio of summed tokens (variant / base) with a 95% paired bootstrap CI."""
    rng = random.Random(seed)
    n = len(base)
    ratios = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        ratios.append(sum(variant[i] for i in idx) / max(sum(base[i] for i in idx), 1))
    ratios.sort()
    return sum(variant) / max(sum(base), 1), ratios[int(0.025 * samples)], ratios[int(0.975 * samples)]


def describe(name: str, pairs: dict, prices) -> None:
    rows = list(pairs.values())
    n = len(rows)
    if not n:
        print(f"{name}: no evaluated pairs")
        return
    mean = lambda key: statistics.mean(row[key] for row in rows)  # noqa: E731
    print(f"\n== {name}: {n} pairs")
    print(
        f"  pass (both features, merged)  {sum(r['passed'] for r in rows)}/{n} = {100 * mean('passed'):.1f}%"
        f"   features passed {sum(r['feature_passed'] for r in rows)}/{2 * n}"
    )
    print(
        f"  tokens per pair  input {mean('input'):,.0f} (cached {mean('cached'):,.0f})  output {mean('output'):,.0f}"
        f" (reasoning {mean('reasoning'):,.0f})  total {mean('total'):,.0f}   ~${statistics.mean(cost(r, prices) for r in rows):.3f}"
    )
    print(
        f"  LLM calls per pair  steps {mean('steps'):.1f}  summaries {mean('summaries'):.2f}"
        f" (summary tokens in/out {mean('summary_in'):,.0f}/{mean('summary_out'):,.0f})"
    )
    if any(r["messages"] or r["histories"] for r in rows):
        msg_in, msg_out = sum(r["message_in"] for r in rows), sum(r["message_out"] for r in rows)
        hist_in, hist_out = sum(r["history_in"] for r in rows), sum(r["history_out"] for r in rows)
        print(
            f"  compressor  messages {sum(r['messages'] for r in rows)} ({msg_in:,} -> {msg_out:,} tokens)"
            f"  histories {sum(r['histories'] for r in rows)} ({hist_in:,} -> {hist_out:,} tokens)"
            f"  GPU time per pair {mean('compress_ms') / 1000:.1f}s  errors {sum(r['errors'] for r in rows)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run names; the first is the baseline")
    parser.add_argument("--logs", default="logs")
    parser.add_argument(
        "--price",
        type=float,
        nargs=3,
        default=(PRICE_INPUT, PRICE_CACHED, PRICE_OUTPUT),
        metavar=("INPUT", "CACHED", "OUTPUT"),
        help="USD per million tokens",
    )
    args = parser.parse_args()

    logs = Path(args.logs)
    runs = {run: load_run(logs, run) for run in args.runs}
    for run, pairs in runs.items():
        describe(run, pairs, args.price)

    base_name, base = args.runs[0], runs[args.runs[0]]
    for run in args.runs[1:]:
        common = sorted(set(base) & set(runs[run]))
        if not common:
            continue
        variant = runs[run]
        better, worse, p = mcnemar([base[k]["passed"] for k in common], [variant[k]["passed"] for k in common])
        print(f"\n== {run} vs {base_name} on {len(common)} common pairs")
        print(
            f"  pass {sum(base[k]['passed'] for k in common)} -> {sum(variant[k]['passed'] for k in common)}"
            f"   (+{better} / -{worse}, exact McNemar p = {p:.3f})"
        )
        for key in ("input", "output", "total"):
            ratio, low, high = bootstrap_ratio([base[k][key] for k in common], [variant[k][key] for k in common])
            print(
                f"  {key:6s} tokens  x{ratio:.3f}  ({100 * (ratio - 1):+.1f}%, 95% CI {100 * (low - 1):+.1f}% to "
                f"{100 * (high - 1):+.1f}%)"
            )


if __name__ == "__main__":
    main()
