"""Compare two judged runs on the questions both answered: overall accuracy,
the paired exact McNemar test, and accuracy per question type.

    python -m eval.compare_runs logs/judge_runs/<base>.jsonl logs/judge_runs/<variant>.jsonl
"""

import argparse
import json
from math import comb


def load(path):
    """question_id -> judged result row."""
    out = {}
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if row.get("event") == "result" and row.get("llm_judge_correct") is not None:
                out[row["question_id"]] = row
    return out


def mcnemar(better, worse):
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = better + worse
    if not n:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("variant")
    args = parser.parse_args()

    a, b = load(args.base), load(args.variant)
    common = sorted(set(a) & set(b))
    correct_a = sum(a[q]["llm_judge_correct"] for q in common)
    correct_b = sum(b[q]["llm_judge_correct"] for q in common)
    better = sum(b[q]["llm_judge_correct"] and not a[q]["llm_judge_correct"] for q in common)
    worse = sum(a[q]["llm_judge_correct"] and not b[q]["llm_judge_correct"] for q in common)
    print(
        f"n={len(common)}  base {100 * correct_a / len(common):.1f}  variant {100 * correct_b / len(common):.1f}  "
        f"better {better} / worse {worse}  McNemar p {mcnemar(better, worse):.3f}"
    )

    by_type = {}
    for q in common:
        counts = by_type.setdefault(a[q]["question_type"], [0, 0, 0])
        counts[0] += 1
        counts[1] += a[q]["llm_judge_correct"]
        counts[2] += b[q]["llm_judge_correct"]
    for question_type, (n, x, y) in sorted(by_type.items()):
        print(f"  {question_type:<28} n={n:<4} {100 * x / n:5.1f} -> {100 * y / n:5.1f}")


if __name__ == "__main__":
    main()
