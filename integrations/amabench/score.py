"""Accuracy per condition (the judge's yes rate), on the questions judged under
every condition. 95% intervals resample whole episodes (the 12 questions of
an episode share one trajectory). Breakdowns by question type (A recall,
B causal inference, C state updating, D state abstraction), domain, and
trajectory length.

    python -m integrations.amabench.score --episodes logs_ama/episodes.jsonl \
        --contexts logs_ama/contexts.jsonl --judged logs_ama/judged.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from integrations.hscbench.score import bootstrap

ORDER = ["trunc", "actions", "bm25", "random", "cpc", "full"]
PAIRS = [("cpc", "bm25"), ("cpc", "random"), ("cpc", "trunc"), ("cpc", "actions"), ("cpc", "full"),
         ("bm25", "random"), ("bm25", "full")]
LONG = 32_000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default="logs_ama/episodes.jsonl")
    parser.add_argument("--contexts", default="logs_ama/contexts.jsonl")
    parser.add_argument("--judged", default="logs_ama/judged.jsonl")
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()

    episodes = {e["episode_id"]: e for e in map(json.loads, Path(args.episodes).open(encoding="utf-8"))}
    meta = {qa["question_uuid"]: {"episode": e["episode_id"], "type": qa["type"], "domain": e["domain"],
                                   "length": "long (>=32k)" if e["full_tokens"] >= LONG else "short (<32k)",
                                   "full_tokens": e["full_tokens"]}
            for e in episodes.values() for qa in e["qa_pairs"]}
    tokens = defaultdict(list)
    if Path(args.contexts).exists():
        for row in map(json.loads, Path(args.contexts).open(encoding="utf-8")):
            if not row["error"]:
                tokens[row["condition"]].append(row["tokens"])
    scores = defaultdict(dict)  # (reader, condition) -> question -> score
    unparsed = defaultdict(int)
    for row in map(json.loads, Path(args.judged).open(encoding="utf-8")):
        if not row["error"] and row["question_uuid"] in meta:
            scores[(row["reader"], row["condition"])][row["question_uuid"]] = row["score"]
            unparsed[(row["reader"], row["condition"])] += not row["parsed"]

    for reader in sorted({key[0] for key in scores}):
        conditions = [c for c in ORDER if (reader, c) in scores] + sorted(c for r, c in scores if r == reader and c not in ORDER)
        common = sorted(set.intersection(*(set(scores[(reader, c)]) for c in conditions)))
        episodes_in = {meta[q]["episode"] for q in common}
        print(f"\n== reader {reader}: {len(common)} questions in {len(episodes_in)} episodes judged under every condition")
        print(f"   {'condition':9s} {'accuracy [95% CI]':26s} {'context tokens (median)':>24s}  unparsed")
        for condition in conditions:
            by_episode = defaultdict(list)
            for q in common:
                by_episode[meta[q]["episode"]].append(scores[(reader, condition)][q])
            mean, low, high = bootstrap(by_episode, args.resamples)
            kept = sorted(tokens[condition]) or sorted(meta[q]["full_tokens"] for q in common)
            print(f"   {condition:9s} {mean:.3f} [{low:.3f}, {high:.3f}]     {kept[len(kept) // 2]:>18d}  "
                  f"{unparsed[(reader, condition)]}")
        for a, b in PAIRS:
            if a in conditions and b in conditions:
                by_episode = defaultdict(list)
                for q in common:
                    by_episode[meta[q]["episode"]].append(scores[(reader, a)][q] - scores[(reader, b)][q])
                mean, low, high = bootstrap(by_episode, args.resamples)
                print(f"   {a} - {b}: {mean:+.3f} [{low:+.3f}, {high:+.3f}]")
        for field in ("type", "domain", "length"):
            groups = defaultdict(list)
            for q in common:
                groups[meta[q][field]].append(q)
            print(f"   by {field}:")
            for group, ids in sorted(groups.items()):
                cells = "  ".join(f"{c} {sum(scores[(reader, c)][q] for q in ids) / len(ids):.3f}" for c in conditions)
                print(f"     {str(group):14s} n={len(ids):4d}  {cells}")


if __name__ == "__main__":
    main()
