"""
Score MODEL ANSWER vs. ground-truth Answers in run_long_bench.py /
evaluate_answers.py output files, using the same per-dataset metrics as the
official LongBench benchmark (https://github.com/THUDM/LongBench):

  - qa_f1_score      : English QA subsets (token-level F1 after normalization)
  - qa_f1_zh_score   : Chinese QA subsets (jieba-tokenized F1)
  - rouge_zh_score   : Chinese summarization subsets (jieba-tokenized ROUGE-L)

A dataset with multiple ground truths is scored against each and the max is
kept (same convention as the official eval script).

Usage:
    python score_answers.py --input_dir output
    python score_answers.py --input_dir output --csv eval_results.csv
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path

import jieba
from rouge import Rouge

ANSWER_MARKER = "MODEL ANSWER"


# ---------------------------------------------------------------- metrics --
# Reimplementation of THUDM/LongBench's metrics.py, trimmed to the metric
# functions actually used by the subsets we have output for.

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _normalize_zh_answer(s: str) -> str:
    cn_punctuation = ",。:?!、;’“”‘’《》()"
    all_punctuation = set(string.punctuation + cn_punctuation)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in all_punctuation)
    return "".join(s.split())


def _token_f1(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not prediction_tokens or not ground_truth_tokens:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    return _token_f1(
        _normalize_answer(prediction).split(),
        _normalize_answer(ground_truth).split(),
    )


def qa_f1_zh_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = [_normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    gt_tokens = [_normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    return _token_f1([t for t in pred_tokens if t], [t for t in gt_tokens if t])


_rouge = Rouge()


def rouge_zh_score(prediction: str, ground_truth: str) -> float:
    pred = " ".join(jieba.cut(prediction, cut_all=False))
    gt = " ".join(jieba.cut(ground_truth, cut_all=False))
    if not pred.strip() or not gt.strip():
        return 0.0
    try:
        return _rouge.get_scores([pred], [gt])[0]["rouge-l"]["f"]
    except (ValueError, RecursionError):
        return 0.0


DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
}


# ---------------------------------------------------------------- parsing --

def parse_scored_example(text: str) -> tuple[str, list[str], str] | None:
    """Return (dataset, ground_truths, model_answer), or None if unanswered."""
    if ANSWER_MARKER not in text:
        return None

    header, _, tail = text.partition(f"{ANSWER_MARKER} (")
    model_answer = tail.split(":\n", 1)[1].strip()

    dataset = None
    answers_line = ""
    for line in header.splitlines():
        if line.startswith("Dataset: "):
            dataset = line[len("Dataset: "):].strip()
        elif line.startswith("Answers: "):
            answers_line = line[len("Answers: "):].strip()

    ground_truths = [] if answers_line == "(none provided)" else [
        a.strip() for a in answers_line.split("; ") if a.strip()
    ]
    return dataset, ground_truths, model_answer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="output")
    p.add_argument("--csv", default=None, help="Optional path to write per-file scores as CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("*.txt"))

    rows = []  # (filename, dataset, score)
    unscored_no_answer = []
    unscored_no_metric = set()

    for path in paths:
        parsed = parse_scored_example(path.read_text(encoding="utf-8"))
        if parsed is None:
            unscored_no_answer.append(path.name)
            continue
        dataset, ground_truths, model_answer = parsed

        metric = DATASET2METRIC.get(dataset)
        if metric is None:
            unscored_no_metric.add(dataset)
            continue
        if not ground_truths:
            continue

        score = max(metric(model_answer, gt) for gt in ground_truths)
        rows.append((path.name, dataset, score))

    if not rows:
        print("No scoreable files found.")
        return

    by_dataset: dict[str, list[float]] = {}
    for _, dataset, score in rows:
        by_dataset.setdefault(dataset, []).append(score)

    print(f"{'Dataset':<20}{'N':>6}{'Mean score':>14}")
    print("-" * 40)
    for dataset in sorted(by_dataset):
        scores = by_dataset[dataset]
        print(f"{dataset:<20}{len(scores):>6}{sum(scores) / len(scores):>14.4f}")
    print("-" * 40)
    all_scores = [s for _, _, s in rows]
    print(f"{'OVERALL':<20}{len(all_scores):>6}{sum(all_scores) / len(all_scores):>14.4f}")

    if unscored_no_answer:
        print(f"\n{len(unscored_no_answer)} file(s) skipped (no MODEL ANSWER yet): "
              f"{unscored_no_answer[:5]}{'...' if len(unscored_no_answer) > 5 else ''}")
    if unscored_no_metric:
        print(f"\nSkipped dataset(s) with no metric implemented yet: {sorted(unscored_no_metric)}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "dataset", "score"])
            writer.writerows(rows)
        print(f"\nWrote per-file scores to {args.csv}")


if __name__ == "__main__":
    main()
