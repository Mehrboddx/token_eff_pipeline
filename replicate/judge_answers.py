"""
LLM-as-judge grading for run_long_bench.py / evaluate_answers.py output files
-- an alternative to score_answers.py's deterministic F1/ROUGE-L metrics, for
a second read on examples where exact-string/token overlap under- or
over-penalizes an answer that's factually correct but phrased differently.

This does NOT replace score_answers.py: LongBench's own published numbers use
F1/ROUGE, and that remains the metric of record. This exists to check where
the two disagree -- a real Gemini call is not free, so it's kept as a
separate opt-in pass rather than folded into scoring by default.

Resumable: a file that already has an "LLM JUDGE" line is skipped, so
re-running after a partial/interrupted pass only grades what's missing --
same convention as evaluate_answers.py.

Caveat: dureader is a summarization dataset (graded elsewhere by ROUGE-L, not
answer-matching) -- a binary CORRECT/INCORRECT verdict is a poor fit for it.
This script will still grade it if asked, but treat that dataset's LLM-judge
numbers with more skepticism than the other seven QA-style datasets.

Usage:
    python judge_answers.py --input_dir output
    python judge_answers.py --input_dir output --limit 20
    python judge_answers.py --input_dir output_gemini_compressor --csv judge_results.csv
"""

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ANSWER_MARKER = "MODEL ANSWER"
JUDGE_MARKER = "LLM JUDGE"
JUDGE_VERDICT_RE = re.compile(rf"{JUDGE_MARKER} \([^)]*\):\s*(CORRECT|INCORRECT)")

JUDGE_PROMPT_TEMPLATE = (
    "You are grading whether a model's answer correctly answers a question, "
    "given one or more acceptable reference answers, in the context of a "
    "long-context QA benchmark.\n\n"
    "Question: {question}\n"
    "Reference answer(s): {reference}\n"
    "Model's answer: {predicted}\n\n"
    "Grade the response as CORRECT if it conveys the same essential fact(s) as "
    "ANY ONE of the reference answers, even if worded differently, more "
    "concise or verbose, or given in a different language than the reference "
    "(source documents and references span English and Chinese). Minor extra "
    "detail beyond the reference is fine as long as the core answer is present "
    "and correct.\n\n"
    "Grade as INCORRECT if the answer states a different fact than every "
    "reference, is too vague or evasive to count as the specific answer asked "
    "for, or fails to answer the question at all.\n\n"
    "Respond with exactly one word: CORRECT or INCORRECT."
)


def parse_dataset(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("Dataset: "):
            return line[len("Dataset: "):].strip()
    return None


def parse_example(text: str) -> dict | None:
    """Return {dataset, question, ground_truths, model_answer}, or None if
    this file has no MODEL ANSWER yet (evaluate_answers.py hasn't run on it)."""
    if f"{ANSWER_MARKER} (" not in text:
        return None

    header, _, tail = text.partition("CONTEXT:\n")
    _, _, tail = tail.partition("\n\nINPUT:\n")
    question, _, tail = tail.partition(f"\n\n{ANSWER_MARKER} (")
    model_answer = tail.split(":\n", 1)[1].split("\n" + JUDGE_MARKER, 1)[0].strip()

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
    return {
        "dataset": dataset,
        "question": question.strip(),
        "ground_truths": ground_truths,
        "model_answer": model_answer,
    }


def llm_judge_grade(client, model, question, ground_truths, predicted):
    reference = "; ".join(ground_truths) if ground_truths else "(none provided)"
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, reference=reference, predicted=predicted)
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(temperature=0),
    )
    verdict = (response.text or "").strip().upper()
    return verdict.startswith("CORRECT"), verdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="output", help="Directory of evaluate_answers.py .txt outputs to grade in place.")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    p.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    p.add_argument("--limit", type=int, default=None, help="Only grade the first N not-yet-judged files (omit for all).")
    p.add_argument("--csv", default=None, help="Optional path to write per-file verdicts as CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("*.txt"))
    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    rows = []  # (filename, dataset, is_correct)
    skipped_unanswered = []
    skipped_no_reference = []
    processed = 0

    for path in paths:
        text = path.read_text(encoding="utf-8")
        dataset = parse_dataset(text)

        existing = JUDGE_VERDICT_RE.search(text)
        if existing:
            rows.append((path.name, dataset, existing.group(1) == "CORRECT"))
            continue

        parsed = parse_example(text)
        if parsed is None:
            skipped_unanswered.append(path.name)
            continue
        if not parsed["ground_truths"]:
            skipped_no_reference.append(path.name)
            continue

        try:
            is_correct, verdict = llm_judge_grade(
                client, args.model, parsed["question"], parsed["ground_truths"], parsed["model_answer"],
            )
        except Exception as exc:
            print(f"[{path.name}] ERROR: {exc}")
            continue

        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n{JUDGE_MARKER} ({args.model}): {verdict}\n")

        rows.append((path.name, dataset, is_correct))
        processed += 1
        print(f"[{processed}] {path.name} ({dataset}) -> {verdict}")

        if args.limit is not None and processed >= args.limit:
            break

    if not rows:
        print("No gradeable files found.")
        return

    by_dataset: dict[str, list[bool]] = {}
    for _, dataset, correct in rows:
        if dataset:
            by_dataset.setdefault(dataset, []).append(correct)

    print(f"\n{'Dataset':<20}{'N':>6}{'LLM-judge accuracy':>20}")
    print("-" * 46)
    for dataset in sorted(by_dataset):
        outcomes = by_dataset[dataset]
        print(f"{dataset:<20}{len(outcomes):>6}{sum(outcomes) / len(outcomes):>19.1%}")
    all_outcomes = [c for _, _, c in rows]
    print("-" * 46)
    print(f"{'OVERALL':<20}{len(all_outcomes):>6}{sum(all_outcomes) / len(all_outcomes):>19.1%}")

    if skipped_unanswered:
        print(f"\n{len(skipped_unanswered)} file(s) skipped (no MODEL ANSWER yet): "
              f"{skipped_unanswered[:5]}{'...' if len(skipped_unanswered) > 5 else ''}")
    if skipped_no_reference:
        print(f"{len(skipped_no_reference)} file(s) skipped (no reference answer to grade against): "
              f"{skipped_no_reference[:5]}{'...' if len(skipped_no_reference) > 5 else ''}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "dataset", "llm_judge_correct"])
            writer.writerows(rows)
        print(f"\nWrote per-file verdicts to {args.csv}")


if __name__ == "__main__":
    main()
