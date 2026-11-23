"""Solve rate, cost and tokens of SWE-agent runs, against each other and the paper.

For each run in --out/<name>: the SWE-bench harness report (resolved ids), the
trajectories' model stats (cost, tokens, calls), and for CPC recall the
compressor calls. Pairs of runs, and each run against the paper's Gemini 2.5
Flash results on the same instances (paper_instance_results.csv, from
"The Complexity Trap" auxiliary data), are compared per instance: instances
solved by only one side, with an exact two-sided sign test.

    python -m integrations.swebench_ctx.report --out logs_swebench mask recall
"""

import argparse
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
PAPER = {"Raw Agent": "paper raw", "Observation Masking": "paper mask", "LLM-Summary": "paper summary"}


def sign_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_run(run_dir: Path) -> dict:
    ids = [line.strip() for line in (HERE / "verified_mini.txt").open() if line.strip()]
    resolved = None
    for report in run_dir.glob("*.json"):
        try:
            data = json.loads(report.read_text())
        except ValueError:
            continue
        if isinstance(data, dict) and "resolved_ids" in data:
            resolved = set(data["resolved_ids"])
    stats = {}
    for traj in run_dir.glob("*/*.traj"):
        info = json.loads(traj.read_text()).get("info", {})
        model = info.get("model_stats", {})
        stats[traj.stem] = {"cost": model.get("instance_cost", 0.0), "sent": model.get("tokens_sent", 0),
                            "received": model.get("tokens_received", 0), "calls": model.get("api_calls", 0),
                            "exit": info.get("exit_status")}
    calls = []
    log = run_dir / "recall_calls.jsonl"
    if log.exists():
        calls = [json.loads(line) for line in log.open()]
    return {"ids": ids, "resolved": resolved, "stats": stats, "recall": calls}


def paper_runs() -> dict[str, dict[str, int]]:
    import csv

    runs: dict[str, dict[str, int]] = {}
    with (HERE / "paper_instance_results.csv").open() as source:
        for row in csv.DictReader(source):
            runs.setdefault(PAPER[row["strategy"]], {})[row["instance_id"]] = int(row["outcome"])
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="logs_swebench")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    outcomes: dict[str, dict[str, int]] = paper_runs()
    print(f"{'run':16s} {'solved':>10s} {'cost/inst':>10s} {'sent/inst':>10s} {'calls':>6s}  exits")
    for name, outcome in outcomes.items():
        print(f"{name:16s} {sum(outcome.values()):3d}/{len(outcome):<3d}({100 * sum(outcome.values()) / len(outcome):4.1f}%)")
    for name in args.runs:
        run = load_run(Path(args.out) / name)
        stats = run["stats"]
        n = max(1, len(stats))
        exits: dict[str, int] = {}
        for s in stats.values():
            exits[str(s["exit"])] = exits.get(str(s["exit"]), 0) + 1
        if run["resolved"] is not None:
            outcomes[name] = {i: int(i in run["resolved"]) for i in run["ids"] if i in stats}
            solved = f"{sum(outcomes[name].values()):3d}/{len(outcomes[name]):<3d}({100 * sum(outcomes[name].values()) / max(1, len(outcomes[name])):4.1f}%)"
        else:
            solved = "not evaluated"
        print(f"{name:16s} {solved:>10s} ${sum(s['cost'] for s in stats.values()) / n:9.3f} "
              f"{sum(s['sent'] for s in stats.values()) / n:10.0f} {sum(s['calls'] for s in stats.values()) / n:6.1f}  {exits}")
        if run["recall"]:
            calls = run["recall"]
            ok = [c for c in calls if "error" not in c]
            ms = sorted(c["ms"] for c in ok) or [0]
            print(f"{'':16s} recall: {len(calls)} calls, {len(calls) - len(ok)} failed, median {ms[len(ms) // 2] / 1000:.1f}s, "
                  f"p90 {ms[int(0.9 * (len(ms) - 1))] / 1000:.1f}s, mean tokens out {sum(c.get('tokens_out') or 0 for c in ok) / max(1, len(ok)):.0f}, "
                  f"mean masked outputs {sum(c['pool'] for c in calls) / len(calls):.1f}")
    names = list(outcomes)
    print("\npaired (instances solved only by A / only by B, sign test):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.startswith("paper") and b.startswith("paper"):
                continue
            common = set(outcomes[a]) & set(outcomes[b])
            only_a = sum(outcomes[a][t] and not outcomes[b][t] for t in common)
            only_b = sum(outcomes[b][t] and not outcomes[a][t] for t in common)
            print(f"  {a} vs {b} on {len(common)}: {only_a} / {only_b}, p={sign_p(only_a, only_b):.2f}")


if __name__ == "__main__":
    main()
