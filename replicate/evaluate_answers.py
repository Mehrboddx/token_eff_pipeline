"""
Feed each compressed LongBench example (output of run_long_bench.py) to
Gemini and append the model's answer to the same .txt file, right next to
the ground-truth "Answers:" line — so both are on disk together for a
later evaluation pass.

Resumable: a file that already has a "MODEL ANSWER" section is skipped, so
re-running after a partial/interrupted pass only fills in what's missing.

Usage:
    python evaluate_answers.py --input_dir output
    python evaluate_answers.py --input_dir output --limit 5 --model gemini-2.5-flash
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ANSWER_MARKER = "MODEL ANSWER"

SYSTEM_PROMPT = (
    "You answer questions using only the given context. "
    "Reply with just the answer itself — no explanation, no restating the question."
)


def parse_example(text: str) -> tuple[str, str]:
    """Return (context, question) from a run_long_bench.py output file."""
    _, _, body = text.partition("CONTEXT:\n")
    context, _, tail = body.partition("\n\nINPUT:\n")
    question, _, _ = tail.partition(f"\n\n{ANSWER_MARKER}")
    return context.strip(), question.strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        default="output",
        help="Directory of run_long_bench.py .txt outputs to evaluate in place.",
    )
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP project (defaults to GOOGLE_CLOUD_PROJECT from .env).",
    )
    p.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N not-yet-answered files (omit for all).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("*.txt"))
    print(input_dir)
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

    processed = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if ANSWER_MARKER in text:
            continue  # already answered, resumable skip

        context, question = parse_example(text)
        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        try:
            response = client.models.generate_content(
                model=args.model, contents=prompt, config=config
            )
            answer = (response.text or "").strip()
        except Exception as exc:
            print(f"[{path.name}] ERROR: {exc}")
            continue

        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n{ANSWER_MARKER} ({args.model}):\n{answer}\n")

        processed += 1
        print(f"[{processed}] {path.name} -> {answer[:80]!r}")

        if args.limit is not None and processed >= args.limit:
            break

    print(f"Done. Answered {processed} file(s).")


if __name__ == "__main__":
    main()
