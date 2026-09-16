"""
Compare answer quality and compressed-context size between two
run_long_bench.py compressor runs: the local CPC/Llama-3.2-1B compressor
(output/, --compressor=cpc) and the hosted Gemini compressor
(output_gemini_compressor/, --compressor=gemini). Both directories hold
COMPRESSED context, not the original uncompressed document — this compares
one compressor against the other, not compression against no compression.
Answer quality uses the same per-dataset LongBench metrics as
score_answers.py.

Usage:
    python compare_outputs.py
    python compare_outputs.py --baseline_dir output --compressed_dir output_gemini_compressor
    python compare_outputs.py --csv compare_results.csv
"""

import argparse
from pathlib import Path

from score_answers import DATASET2METRIC

ANSWER_MARKER = "MODEL ANSWER"


def parse_example(text: str) -> dict | None:
    """Return {dataset, ground_truths, model_answer, context}, or None if unanswered."""
    if ANSWER_MARKER not in text:
        return None

    header, _, tail = text.partition("CONTEXT:\n")
    context, _, tail = tail.partition("\n\nINPUT:\n")
    _, _, tail = tail.partition(f"\n\n{ANSWER_MARKER} (")
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
    return {
        "dataset": dataset,
        "ground_truths": ground_truths,
        "model_answer": model_answer,
        "context": context.strip(),
    }


def score(dataset: str | None, ground_truths: list[str], model_answer: str) -> float | None:
    metric = DATASET2METRIC.get(dataset)
    if metric is None or not ground_truths:
        return None
    return max(metric(model_answer, gt) for gt in ground_truths)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", default="output",
                    help="CPC (local Llama-3.2-1B) compressed outputs (run_long_bench.py --compressor=cpc).")
    p.add_argument("--compressed_dir", default="output_gemini_compressor",
                    help="Gemini-compressed outputs (run_long_bench.py --compressor=gemini).")
    p.add_argument("--csv", default=None, help="Optional path to write per-file comparison as CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    compressed_dir = Path(args.compressed_dir)

    baseline_files = {p.name: p for p in baseline_dir.glob("*.txt")}
    compressed_files = {p.name: p for p in compressed_dir.glob("*.txt")}
    common = sorted(set(baseline_files) & set(compressed_files))
    only_compressed = sorted(set(compressed_files) - set(baseline_files))
    only_baseline = sorted(set(baseline_files) - set(compressed_files))

    rows = []
    skipped = []

    for name in common:
        base = parse_example(baseline_files[name].read_text(encoding="utf-8"))
        comp = parse_example(compressed_files[name].read_text(encoding="utf-8"))
        if base is None or comp is None:
            skipped.append(name)
            continue

        dataset = base["dataset"]
        ground_truths = base["ground_truths"]
        base_score = score(dataset, ground_truths, base["model_answer"])
        comp_score = score(dataset, ground_truths, comp["model_answer"])
        if base_score is None or comp_score is None:
            skipped.append(name)
            continue

        rows.append({
            "file": name,
            "dataset": dataset,
            "base_score": base_score,
            "comp_score": comp_score,
            "base_len": len(base["context"]),
            "comp_len": len(comp["context"]),
            "base_words": len(base["context"].split()),
            "comp_words": len(comp["context"].split()),
        })

    if not rows:
        print("No comparable, scoreable files found.")
        return

    def summarize(label: str, group: list[dict]) -> None:
        n = len(group)
        base_mean = sum(r["base_score"] for r in group) / n
        comp_mean = sum(r["comp_score"] for r in group) / n
        base_words_mean = sum(r["base_words"] for r in group) / n
        comp_words_mean = sum(r["comp_words"] for r in group) / n
        comp_ratio = sum(r["comp_len"] / r["base_len"] for r in group if r["base_len"]) / n
        print(f"{label:<20}{n:>5}{base_mean:>10.4f}{comp_mean:>10.4f}{comp_mean - base_mean:>+9.4f}"
              f"{base_words_mean:>10.0f}{comp_words_mean:>10.0f}{comp_ratio:>13.1%}")

    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    header = (f"{'Dataset':<20}{'N':>5}{'CPC score':>10}{'Gemini sc':>10}{'Delta':>9}"
              f"{'CPC words':>10}{'Gem words':>10}{'Gemini/CPC ctx':>13}")
    print(header)
    print("-" * len(header))
    for dataset in sorted(by_dataset):
        summarize(dataset, by_dataset[dataset])
    print("-" * len(header))
    summarize("OVERALL", rows)

    if skipped:
        print(f"\n{len(skipped)} file(s) skipped (unanswered or no metric): "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if only_compressed:
        print(f"{len(only_compressed)} file(s) only in {compressed_dir.name}: "
              f"{only_compressed[:5]}{'...' if len(only_compressed) > 5 else ''}")
    if only_baseline:
        print(f"{len(only_baseline)} file(s) only in {baseline_dir.name}: "
              f"{only_baseline[:5]}{'...' if len(only_baseline) > 5 else ''}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file", "dataset", "cpc_score", "gemini_score", "delta",
                "cpc_context_chars", "gemini_context_chars", "cpc_context_words",
                "gemini_context_words", "gemini_ctx_over_cpc_ctx",
            ])
            for r in rows:
                writer.writerow([
                    r["file"], r["dataset"], r["base_score"], r["comp_score"],
                    r["comp_score"] - r["base_score"], r["base_len"], r["comp_len"],
                    r["base_words"], r["comp_words"],
                    r["comp_len"] / r["base_len"] if r["base_len"] else "",
                ])
        print(f"\nWrote per-file comparison to {args.csv}")


if __name__ == "__main__":
    main()
