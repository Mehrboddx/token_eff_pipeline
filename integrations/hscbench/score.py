"""Score repository-track predictions (HSC-Bench Section 4.2, Appendix A).

line-F1 per task over (file, line) pairs: predicted spans are inclusive,
overlapping spans merged, at most 12 files and 12 spans per file; an
unparsable answer is an empty prediction (F1 0). file-F1 the same over
files. Predicted paths are matched to gold files when a gold path is the
predicted path or a suffix of it at a directory boundary (agents saw
absolute paths such as /testbed/<repo path>). Means are over tasks;
95% intervals resample whole repositories (20,000 times), as in the paper.

    python -m integrations.hscbench.score --tasks logs_hsc/tasks.jsonl \
        --contexts logs_hsc/contexts.jsonl --predictions logs_hsc/predictions.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ORDER = ["recent", "summary", "state", "cpc", "cpc_2x", "cpc_ctx", "cpc_ctx_2x", "full"]


def normalize(path: str, gold_files: set[str]) -> str:
    path = path.strip()
    while path.startswith(("./", "/")):
        path = path[2:] if path.startswith("./") else path[1:]
    if path in gold_files:
        return path
    for gold in gold_files:
        if path.endswith("/" + gold):
            return gold
    return path


def line_set(locations: list[dict], gold_files: set[str]) -> set[tuple[str, int]]:
    lines = set()
    for location in locations or []:
        file = normalize(location["file"], gold_files)
        for start, end in location.get("spans") or []:
            lines.update((file, line) for line in range(start, end + 1))
    return lines


def f1(predicted: set, gold: set) -> float:
    if not predicted or not gold:
        return 0.0
    hit = len(predicted & gold)
    return 2 * hit / (len(predicted) + len(gold))


def score_task(task: dict, locations: list[dict] | None) -> dict:
    gold_files = {span["file"] for span in task["gold"]}
    gold_lines = {(span["file"], line) for span in task["gold"] for line in range(span["start"], span["end"] + 1)}
    predicted_files = {normalize(location["file"], gold_files) for location in locations or []}
    return {"line_f1": f1(line_set(locations, gold_files), gold_lines), "file_f1": f1(predicted_files, gold_files),
            "parsed": locations is not None}


def visibility(task: dict, recent_text: str, full_text: str) -> str:
    """Where the annotated files appear before handoff (Table 6): any still in the
    recent transcript, else only earlier in the history, else not observed."""
    files = {span["file"] for span in task["gold"]}
    if any(file in recent_text for file in files):
        return "still recent"
    if any(file in full_text for file in files):
        return "earlier only"
    return "not observed"


def bootstrap(values_by_repo: dict[str, list[float]], resamples: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    repos = list(values_by_repo)
    flat = [value for values in values_by_repo.values() for value in values]
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [value for repo in (rng.choice(repos) for _ in repos) for value in values_by_repo[repo]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(flat) / len(flat), means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="logs_hsc/tasks.jsonl")
    parser.add_argument("--contexts", default="logs_hsc/contexts.jsonl")
    parser.add_argument("--predictions", default="logs_hsc/predictions.jsonl")
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()

    tasks = {task["task_id"]: task for task in map(json.loads, Path(args.tasks).open(encoding="utf-8"))}
    contexts = {row["task_id"]: row for row in map(json.loads, Path(args.contexts).open(encoding="utf-8"))}
    scores: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)  # (receiver, condition) -> task -> score
    for row in map(json.loads, Path(args.predictions).open(encoding="utf-8")):
        if row["task_id"] in tasks and not row["error"]:
            scores[(row["receiver"], row["condition"])][row["task_id"]] = score_task(tasks[row["task_id"]], row["locations"])

    for receiver in sorted({key[0] for key in scores}):
        conditions = [c for c in ORDER if (receiver, c) in scores] + sorted(c for r, c in scores if r == receiver and c not in ORDER)
        common = set.intersection(*(set(scores[(receiver, c)]) for c in conditions))
        print(f"\n== receiver {receiver}: {len(common)} tasks scored under every condition")
        budgets = sorted(contexts[t]["budget"] for t in common if t in contexts)
        if budgets:
            print(f"   matched budget (structured-state tokens): median {budgets[len(budgets) // 2]}, range {budgets[0]}-{budgets[-1]}")
        print(f"   {'condition':10s} {'line-F1 [95% CI]':28s} {'file-F1':8s} {'tokens':>7s}  parsed")
        for condition in conditions:
            by_repo = defaultdict(list)
            for task_id in common:
                by_repo[tasks[task_id]["repo"]].append(scores[(receiver, condition)][task_id]["line_f1"])
            mean, low, high = bootstrap(by_repo, args.resamples)
            file_f1 = sum(scores[(receiver, condition)][t]["file_f1"] for t in common) / len(common)
            tokens = sum(contexts[t]["tokens"][condition] for t in common if t in contexts) / max(1, len(common))
            parsed = sum(scores[(receiver, condition)][t]["parsed"] for t in common)
            print(f"   {condition:10s} {mean:.3f} [{low:.3f}, {high:.3f}]       {file_f1:.3f}    {tokens:7.0f}  {parsed}/{len(common)}")
        for base in ("recent", "state"):
            if base not in conditions:
                continue
            for condition in conditions:
                if condition == base:
                    continue
                by_repo = defaultdict(list)
                for task_id in common:
                    by_repo[tasks[task_id]["repo"]].append(scores[(receiver, condition)][task_id]["line_f1"] - scores[(receiver, base)][task_id]["line_f1"])
                mean, low, high = bootstrap(by_repo, args.resamples)
                print(f"   {condition} - {base}: {mean:+.3f} [{low:+.3f}, {high:+.3f}]")
        # by where the annotated files were visible before handoff
        groups = defaultdict(list)
        for task_id in common:
            if task_id in contexts:
                groups[visibility(tasks[task_id], contexts[task_id]["contexts"]["recent"], contexts[task_id]["contexts"]["full"])].append(task_id)
        for group, ids in sorted(groups.items()):
            cells = "  ".join(f"{c} {sum(scores[(receiver, c)][t]['line_f1'] for t in ids) / len(ids):.3f}" for c in conditions)
            print(f"   [{group}, {len(ids)} tasks] {cells}")


if __name__ == "__main__":
    main()
