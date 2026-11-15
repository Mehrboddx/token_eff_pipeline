"""How many tokens an ideal per-question budget would save, from runs
already judged. No model calls.

Each question can be answered from a ladder of contexts, cheapest first
(e.g. CPC 2k, CPC 4k, full history). An ideal gate that knows which
context will answer correctly picks one per question. Three policies:

- match R: the cheapest level that does at least as well as reference
  level R on that question (R wrong -> cheapest level). Reported
  accuracy is R's own, so a cheaper level being right by chance where R
  was wrong is not counted.
- oracle: the cheapest level answered correctly, else the cheapest
  level. Accuracy is the union over levels, which includes answers that
  were right by chance (answering is sampled), so it is an upper bound.

Tokens are the compressed context's cl100k tokens; the full history is
rendered and counted here. A real gate would decide before answering,
so only the chosen level is paid.

Usage:
    python -m eval.adaptive_budget --dataset longmemeval_s --encoder mistral
    python -m eval.adaptive_budget --dataset locomo
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken

from eval.longmemeval import CompressionSettings, compress_question, load_items

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENCODING = tiktoken.get_encoding("cl100k_base")

# dataset -> encoder -> [(level name, run tag)], cheapest first; full history last.
LADDERS = {
    "longmemeval_s": {
        "mistral": [("2k", "run0003_compress_cpc-mistral_b2000"), ("4k", "run0001_compress_cpc-mistral_b4000"),
                    ("full", "run0039_full_fullhistory")],
        "llama": [("2k", "run0005_compress_cpc-llama_b2000"), ("4k", "run0004_compress_cpc-llama_b4000"),
                  ("full", "run0039_full_fullhistory")],
    },
    "locomo": {
        "mistral": [("4k", "run0021_compress_locomo_cpc-mistral_b4000"), ("full", "run0041_full_locomo_fullhistory")],
    },
}


def results(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if record.get("event") == "result":
            out[record["question_id"]] = record
    return out


def full_history_tokens(dataset, question_ids):
    """cl100k tokens of each rendered full history, cached in logs/ since
    rendering every history takes minutes."""
    cache_path = Path(f"logs/full_history_tokens_{dataset}.json")
    tokens = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = set(question_ids) - set(tokens)
    if missing:
        for item in load_items(dataset, limit=None):
            if item["question_id"] in missing:
                described, _, _, _ = compress_question(item, None, CompressionSettings())
                # disallowed_special=(): some histories contain "<|endoftext|>" as plain text.
                tokens[item["question_id"]] = sum(len(ENCODING.encode(entry["text"], disallowed_special=()))
                                                  for entry in described)
        cache_path.write_text(json.dumps(tokens))
    return tokens


def choose(levels, correct, rule, reference=None):
    """Index of the level picked for one question."""
    if rule == "oracle":
        return next((i for i, ok in enumerate(correct) if ok), 0)
    target = correct[reference]
    return next(i for i in range(len(levels)) if correct[i] >= target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s", choices=list(LADDERS))
    parser.add_argument("--encoder", default="mistral")
    args = parser.parse_args()

    ladder = LADDERS[args.dataset][args.encoder]
    names = [name for name, _ in ladder]
    judged = [results(f"logs/judge_runs/eval_longmemeval_{tag}.jsonl") for _, tag in ladder]
    compressed = [results(f"logs/eval_longmemeval_{tag}.jsonl") for _, tag in ladder]
    ids = sorted(set.intersection(*(set(j) for j in judged)))

    full_tokens = full_history_tokens(args.dataset, set(ids))
    tokens = {qid: [compressed[i][qid]["compressed_tokens"] if names[i] != "full" else full_tokens[qid]
                    for i in range(len(names))] for qid in ids}
    correct = {qid: [bool(j[qid].get("llm_judge_correct")) for j in judged] for qid in ids}
    types = {qid: judged[0][qid]["question_type"] for qid in ids}

    print(f"{args.dataset}, {args.encoder} encoder, {len(ids)} questions\n")
    print(f"{'policy':<22} {'accuracy':>8} {'mean tokens':>12} {'vs 4k':>7}   picks")
    fixed_4k = statistics.mean(tokens[q][names.index("4k")] for q in ids)
    for i, name in enumerate(names):
        mean = statistics.mean(tokens[q][i] for q in ids)
        print(f"{'fixed ' + name:<22} {100 * statistics.mean(correct[q][i] for q in ids):>8.1f} {mean:>12,.0f} "
              f"{mean / fixed_4k:>7.2f}")

    policies = [(f"match {names[r]}", "match", r) for r in range(1, len(names))] + [("oracle", "oracle", None)]
    picks_by_type = {}
    for label, rule, reference in policies:
        picks = {q: choose(names, correct[q], rule, reference) for q in ids}
        # match R: report R's own accuracy. Where R is wrong the cheapest
        # level is picked, and its lucky hits there are not counted.
        accuracy = (statistics.mean(correct[q][reference] for q in ids) if rule == "match"
                    else statistics.mean(correct[q][picks[q]] for q in ids))
        mean = statistics.mean(tokens[q][picks[q]] for q in ids)
        share = Counter(names[p] for p in picks.values())
        print(f"{label:<22} {100 * accuracy:>8.1f} {mean:>12,.0f} {mean / fixed_4k:>7.2f}   "
              + ", ".join(f"{n} {100 * share[n] / len(ids):.0f}%" for n in names))
        picks_by_type[label] = picks

    print(f"\nShare of questions answered at the cheapest level, policy 'match {names[-1]}', by type:")
    picks = picks_by_type[f"match {names[-1]}"]
    by_type = defaultdict(list)
    for q in ids:
        by_type[types[q]].append(picks[q])
    for qtype, values in sorted(by_type.items()):
        print(f"  {qtype:<28} n={len(values):<4} " + "  ".join(
            f"{n} {100 * sum(v == i for v in values) / len(values):3.0f}%" for i, n in enumerate(names)))


if __name__ == "__main__":
    main()
