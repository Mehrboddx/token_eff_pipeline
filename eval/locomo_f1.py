"""LoCoMo's original metric -- token F1 against the reference answer --
computed from answers already in judge logs. No model calls.

Scoring follows the official script (snap-research/locomo,
task_eval/evaluation.py): lowercase, strip punctuation, articles and
"and", Porter-stem, then token F1. Multi-hop (category 1) splits both
sides on commas and averages the best match per reference part;
open-domain (category 3) keeps only the reference before the first ';'.
Adversarial questions are not in our runs.

F1 rewards short answers: our agent answers in full sentences, so
absolute F1 is lower than with the short-phrase prompt of the original
paper. Comparisons between methods are fair because all share the same
answer prompt; mean answer length is printed to check that.

Usage:
    python -m eval.locomo_f1 logs/judge_runs/*locomo*.jsonl
"""

import json
import random
import re
import statistics
import string
import sys
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer

from eval.locomo import CATEGORIES

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEMMER = PorterStemmer()
NAME_TO_CATEGORY = {name: number for number, name in CATEGORIES.items()}
ORDER = ["single-hop", "multi-hop", "temporal", "open-domain"]


def normalize_answer(s):
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction, ground_truth):
    prediction_tokens = [STEMMER.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [STEMMER.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return statistics.mean(max(f1_score(p, g) for p in predictions) for g in ground_truths)


def score(question_type, prediction, reference):
    category = NAME_TO_CATEGORY[question_type]
    prediction = prediction or ""
    reference = str(reference)
    if category == 3:
        reference = reference.split(";")[0].strip()
    if category == 1:
        return multi_answer_f1(prediction, reference)
    return f1_score(prediction, reference)


def load(path):
    """question_id -> result record; a resumed run's later record wins."""
    results = {}
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if record.get("event") == "result":
            results[record["question_id"]] = record
    return results


def paired_bootstrap(a, b, rounds=10000, seed=0):
    """Two-sided p-value that mean(a - b) differs from 0."""
    diffs = [x - y for x, y in zip(a, b)]
    observed = statistics.mean(diffs)
    rng = random.Random(seed)
    centered = [d - observed for d in diffs]
    n = len(diffs)
    extreme = sum(abs(statistics.mean(rng.choices(centered, k=n))) >= abs(observed) for _ in range(rounds))
    return observed, (extreme + 1) / (rounds + 1)


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    runs = {path.stem: load(path) for path in paths}

    print(f"{'run':<70} {'n':>5} {'F1':>5} {'judge':>6} {'len':>5}  " + " ".join(f"{t[:6]:>6}" for t in ORDER))
    scored = {}
    for name, results in runs.items():
        rows = {qid: (r["question_type"], score(r["question_type"], r.get("predicted"), r["reference"]),
                      bool(r.get("llm_judge_correct")), len(normalize_answer(r.get("predicted") or "").split()))
                for qid, r in results.items() if r.get("question_type") in NAME_TO_CATEGORY}
        scored[name] = rows
        values = list(rows.values())
        by_type = {t: [v[1] for v in values if v[0] == t] for t in ORDER}
        print(f"{name:<70} {len(values):>5} {100 * statistics.mean(v[1] for v in values):>5.1f} "
              f"{100 * statistics.mean(v[2] for v in values):>6.1f} {statistics.mean(v[3] for v in values):>5.1f}  "
              + " ".join(f"{100 * statistics.mean(by_type[t]):>6.1f}" if by_type[t] else f"{'-':>6}" for t in ORDER))
    return scored


if __name__ == "__main__":
    main()
