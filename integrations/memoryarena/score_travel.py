"""Score MemoryArena group travel runs, paired across memory conditions.

PS  = share of travelers whose plan passes every slot (pooled over travelers, as in their eval.py);
SPS = share of the slots that differ from the base plan that the traveler got right,
      averaged per group, then over groups;
SR  = share of groups where every traveler passes.
Only groups run under every listed condition count; CIs from a bootstrap over groups.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from integrations.memoryarena.memory import CONDITIONS, HEAD_CONDITIONS
from integrations.memoryarena.score import PAIRS, cost


def ratio_ci(keys, num, den, resamples: int, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [rng.choice(keys) for _ in keys]
        values.append(sum(num[k] for k in sample) / max(1e-9, sum(den[k] for k in sample)))
    values.sort()
    return values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]


def report(rows: list[dict], conditions: list[str], resamples: int) -> None:
    by = {(r["group"], r["condition"]): r for r in rows}
    keys = sorted({g for g, _ in by if all((g, c) in by for c in conditions)})
    if not keys:
        print("no group run under every condition")
        return
    travelers = sum(by[(k, conditions[0])]["n"] for k in keys)
    print(f"== {len(keys)} groups, {travelers} travelers, run under every condition")
    print(f"   {'condition':<13} {'PS [95% CI]':<22} {'SPS [95% CI]':<22} SR      steps/traveler  memory tokens (median, max)")
    ones = {k: 1.0 for k in keys}
    for c in conditions:
        passes = {k: sum(p["pass"] for p in by[(k, c)]["persons"]) for k in keys}
        counts = {k: len(by[(k, c)]["persons"]) for k in keys}
        sps = {k: by[(k, c)]["sps"] or 0.0 for k in keys}
        sr = sum(by[(k, c)]["sr"] for k in keys) / len(keys)
        lo, hi = ratio_ci(keys, passes, counts, resamples)
        slo, shi = ratio_ci(keys, sps, ones, resamples)
        steps = statistics.mean(r["steps"] for k in keys for r in by[(k, c)]["rounds"])
        shown = [r["memory_tokens"] for k in keys for r in by[(k, c)]["rounds"]]
        print(f"   {c:<13} {sum(passes.values()) / sum(counts.values()):.3f} [{lo:.3f}, {hi:.3f}]   "
              f"{sum(sps.values()) / len(keys):.3f} [{slo:.3f}, {shi:.3f}]   {sr:.3f}   {steps:5.1f}           "
              f"{statistics.median(shown):.0f}, {max(shown)}")
    for a, b in PAIRS:
        if a in conditions and b in conditions:
            diff = {k: (by[(k, a)]["sps"] or 0.0) - (by[(k, b)]["sps"] or 0.0) for k in keys}
            lo, hi = ratio_ci(keys, diff, ones, resamples)
            print(f"   {a} - {b} SPS: {sum(diff.values()) / len(keys):+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--usage", type=Path, default=None)
    parser.add_argument("--conditions", default=",".join(CONDITIONS + HEAD_CONDITIONS))
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.results, encoding="utf-8")]
    report(rows, [c for c in args.conditions.split(",") if c], args.resamples)
    cost(args.usage)


if __name__ == "__main__":
    main()
