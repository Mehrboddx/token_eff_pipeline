"""Evaluate the agentic system's context compression against LongMemEval.

Downloads xiaowu0162/LongMemEval's `longmemeval_s` split — 500 self-contained
long-memory QA items, each with its own embedded chat-history haystack (tens
of sessions, tens of thousands of words) plus a question that depends on
recalling one fact buried in it. Each item's haystack is replayed straight
into a fresh Memory (no model calls — these are pre-recorded turns, not a
live conversation), then Runner answers the real question exactly as
main.py's REPL would, just fed a benchmark haystack instead of interactive
input. This is the same Agent/Runner/Memory/TokenWise stack, so it exercises
the actual compression path, not a mock of it.

Usage:
    python -m eval.longmemeval --limit 25
    python -m eval.longmemeval --limit 25 --compare-baseline
    python -m eval.longmemeval --limit 25 --compressor cpc
"""

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types
from huggingface_hub import hf_hub_download

from core.agent import Agent
from core.cpcCompressor import CPCCompressor
from core.geminiCompressor import GeminiCompressor
from core.memory import Memory
from core.runner import Runner
from core.tokenWise import TokenWise
from prompts.prompt import universal_agent_prompt

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = "gemini-2.5-flash"
DATASET_REPO = "xiaowu0162/LongMemEval"
DATASET_FILE = "longmemeval_s"
LOGS_DIR = Path("logs")


def load_longmemeval(limit=None, seed=42, question_types=None):
    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    items = json.loads(Path(path).read_text(encoding="utf-8"))

    if question_types:
        items = [item for item in items if item["question_type"] in question_types]

    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)

    return items


def _parse_haystack_date(date_str):
    # "2023/05/20 (Sat) 19:50" -> "2023/05/20 19:50" (drop the weekday
    # parenthetical, which strptime has no directive for).
    cleaned = re.sub(r"\s*\([^)]*\)", "", date_str).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")


def seed_memory_from_haystack(item):
    """Replay one item's haystack sessions into a fresh Memory, in their
    original (chronological) order. These are pre-recorded turns, so they
    go straight into Memory via add_user_message/add_model_content rather
    than through Runner — replaying them through the model would be both
    wrong (they're not this agent's own words) and enormously expensive.

    Each turn is tagged with how many days before the question it happened,
    computed from haystack_dates and question_date. Without this,
    temporal-reasoning questions ("how many days ago...") are unanswerable
    in principle — sessions mostly use relative phrasing ("today", "last
    week"), so the model has no absolute date to anchor that to.

    Two things ruled out first because they actively regressed retrieval:
    - A separate "[Session date: ...]" marker sentence per session: short,
      and every one contains the word "date", so on any date-related query
      the word-overlap scorer (which normalizes by sentence length) ranks
      it above genuinely relevant but longer sentences. ~20 markers alone
      filled the entire compression budget on one test question, pushing
      out all real content.
    - Tagging real turns with the literal date string instead: same
      problem in a different shape — every haystack date shares the same
      year (and often nearby weekday abbreviations) with the injected
      "today's date is ..." question prefix, so literally every tagged
      sentence in the haystack got a free token-overlap match on "2023"
      regardless of actual relevance.
    A precomputed relative offset ("14 days before this question") avoids
    both: it doesn't share vocabulary with a typical query (so it doesn't
    inflate irrelevant sentences), and it only helps when the query
    actually is about elapsed time (sharing a word like "days"), which is
    exactly the case it needs to help."""
    memory = Memory()
    dates = item.get("haystack_dates") or []
    question_date = item.get("question_date")
    question_dt = None
    if question_date:
        try:
            question_dt = _parse_haystack_date(question_date)
        except ValueError:
            question_dt = None

    for session_index, session in enumerate(item["haystack_sessions"]):
        date_tag = ""
        if question_dt is not None and session_index < len(dates):
            try:
                days_before = (question_dt - _parse_haystack_date(dates[session_index])).days
                date_tag = f"(from {days_before} days before this question) "
            except ValueError:
                date_tag = ""

        for turn in session:
            content = (turn.get("content") or "").strip()
            if not content:
                continue

            tagged_content = date_tag + content
            if turn.get("role") == "user":
                memory.add_user_message(tagged_content)
            elif turn.get("role") == "assistant":
                memory.add_model_content(
                    types.Content(role="model", parts=[types.Part.from_text(text=tagged_content)])
                )

    return memory


def normalize(text):
    # Collapse to a space, not empty: stripping straight to "" merges words
    # that were only separated by punctuation/newlines with no surrounding
    # space (e.g. "conversations:\n\nWhen" -> "conversationswhen"), silently
    # destroying a word boundary and making a real match look like a miss.
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def grade(predicted, reference):
    """Correct if the normalized reference answer appears verbatim in the
    response (cheap, precise), OR if the response's normalized words are a
    superset of the reference's (catches the same fact stated in different
    word order or lightly paraphrased, e.g. "3rd of June" vs "June 3rd").
    Both are heuristics with no extra API calls — neither understands
    meaning, so a sufficiently different phrasing can still be missed."""
    normalized_reference = normalize(reference)
    normalized_predicted = normalize(predicted)

    if not normalized_reference:
        return False

    if normalized_reference in normalized_predicted:
        return True

    reference_words = normalized_reference.split()
    predicted_words = set(normalized_predicted.split())
    missing = sum(1 for word in reference_words if word not in predicted_words)
    # Long, sentence-style reference answers (e.g. "When you just started
    # ..., you led 4 engineers. Now, you lead 5 engineers") often get one
    # connective word swapped for a synonym ("now" -> "currently") even
    # when every substantive fact and number matches — tolerate exactly one
    # miss there. Short answers (a name, a count, a date) get no tolerance:
    # losing even one word out of two or three is much more likely to mean
    # the answer is actually different, not just reworded.
    tolerance = 1 if len(reference_words) >= 6 else 0
    return missing <= tolerance


def build_agent(project=PROJECT_ID, location=LOCATION):
    # No tools: this eval is specifically about whether compression
    # preserves the fact needed to answer, not about tool-use behavior —
    # leaving calculator/exit_pipeline in just adds an unrelated variable.
    return Agent(
        name="longmemeval-agent",
        model=MODEL,
        system_prompt=universal_agent_prompt,
        project=project,
        location=location,
        tools=None,
    )


def build_tokenwise(compressor: str, cpc_max_seq_length: int = 1536, cpc_preset: str = "llama",
                     project=PROJECT_ID, location=LOCATION) -> TokenWise:
    if compressor == "cpc":
        return TokenWise(model=CPCCompressor(preset=cpc_preset, max_seq_length=cpc_max_seq_length))

    if compressor == "gemini":
        return TokenWise(model=GeminiCompressor(project=project, location=location, model=MODEL))

    return TokenWise(use_openai_tokenizer=True)


NEEDS_QUESTION_DATE = {"temporal-reasoning", "knowledge-update"}


def answer_question(agent, item, tokenwise, compression_sentence_threshold, compression_token_budget, recent_turns):
    memory = seed_memory_from_haystack(item)
    runner = Runner(
        agent,
        memory,
        tokenwise=tokenwise,
        compression_sentence_threshold=compression_sentence_threshold,
        compression_token_budget=compression_token_budget,
        recent_turns=recent_turns,
    )
    question = item["question"]
    # Same reasoning as seed_memory_from_haystack's date tags: only add this
    # for question types that actually need "now" to reason about elapsed
    # time or the most recent version of a fact — the query is also what
    # TokenWise scores sentences against, so adding "today"/"date" to every
    # question would give every sentence that happens to mention a date a
    # free relevance boost on completely unrelated questions.
    if item.get("question_date") and item.get("question_type") in NEEDS_QUESTION_DATE:
        question = f"(Today's date is {item['question_date']}.) {question}"

    # Capture what the compressor actually produced for this question, via
    # the same Agent.context_monitor hook main.py's REPL uses for its own
    # logging. Without this, a FAIL only shows the model's answer -- it
    # can't distinguish "the needed fact never reached the model"
    # (a compression miss) from "the model had it and still got it wrong"
    # (a reasoning miss), which is exactly the distinction that mattered
    # when diagnosing the tie-break bug earlier. Eval agents have no
    # tools, so Runner.run() always does exactly one pass -- no need to
    # handle multiple context_monitor calls per question.
    captured = {}

    def capture_context(history, context_meta=None):
        captured["mode"] = (context_meta or {}).get("mode")
        captured["history"] = history

    agent.context_monitor = capture_context
    try:
        response = runner.run(question)
    finally:
        agent.context_monitor = None

    compressed_context = None
    if captured.get("mode") == "compressed_history" and captured.get("history"):
        first_item = captured["history"][0]
        if first_item.get("kind") == "compressed_context":
            compressed_context = first_item["text"]

    return response.text or "", captured.get("mode"), compressed_context


def summarize(name, rows):
    total = len(rows)
    passed = sum(row["correct"] for row in rows)
    print(f"\n{name}: {passed}/{total} ({passed / total:.1%})")

    by_type = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row["correct"])
    for question_type, outcomes in sorted(by_type.items()):
        print(f"  {question_type}: {sum(outcomes)}/{len(outcomes)} ({sum(outcomes) / len(outcomes):.1%})")


RUN_LOG_PATTERN = re.compile(r"eval_longmemeval_run(\d+)\.jsonl$")


def load_run_log(log_path):
    """Parse an existing run's JSONL log for --resume: the run's own
    run_start record (settings to reproduce the same item sample) and every
    result row logged so far (across however many prior sessions), whether
    the file was written by one run_start or a run_start plus one or more
    run_resume continuations."""
    if not log_path.exists():
        raise SystemExit(f"{log_path} does not exist")

    run_start = None
    results = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") in ("run_start", "run_resume") and run_start is None:
                run_start = record
            elif record.get("event") == "result":
                results.append(record)

    return run_start, results


def _gcs_blob(gcs_uri, filename):
    from google.cloud import storage

    assert gcs_uri.startswith("gs://"), "--log-gcs-uri must start with gs://"
    bucket_name, _, prefix = gcs_uri[len("gs://"):].partition("/")
    prefix = prefix.rstrip("/")
    blob_name = f"{prefix}/{filename}" if prefix else filename
    return storage.Client().bucket(bucket_name).blob(blob_name)


def _next_run_number(log_dir, gcs_uri=None):
    """Sequential run numbers instead of a random session code, so files
    sort chronologically and it's obvious at a glance which run is latest
    (ls/dir already sorts eval_longmemeval_run0009.jsonl after run0008,
    whereas two random hex codes tell you nothing about order).

    When gcs_uri is set, also checks the GCS prefix, not just the local
    directory: a fresh Vertex AI job container always starts with an empty
    local logs/ dir, so counting local files alone would hand out run0001
    on every single job launch -- silently colliding with (and overwriting)
    whatever an earlier job already wrote to that same GCS prefix."""
    existing = [
        int(match.group(1))
        for path in log_dir.glob("eval_longmemeval_run*.jsonl")
        if (match := RUN_LOG_PATTERN.match(path.name))
    ]

    if gcs_uri:
        from google.cloud import storage

        assert gcs_uri.startswith("gs://"), "--log-gcs-uri must start with gs://"
        bucket_name, _, prefix = gcs_uri[len("gs://"):].partition("/")
        list_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
        for blob in storage.Client().list_blobs(bucket_name, prefix=list_prefix):
            match = RUN_LOG_PATTERN.match(blob.name.rsplit("/", 1)[-1])
            if match:
                existing.append(int(match.group(1)))

    return max(existing, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=25, help="Number of questions to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-types", default=None, help="Comma-separated question_type filter.")
    parser.add_argument(
        "--token-budget", type=int, default=500,
        help="compression_token_budget. Haystacks run tens of thousands of words, so "
             "main.py's interactive default (50) is far too tight here.",
    )
    parser.add_argument("--sentence-threshold", type=int, default=20)
    parser.add_argument("--recent-turns", type=int, default=4)
    parser.add_argument(
        "--compressor",
        choices=["lexical", "gemini", "cpc"],
        default="lexical",
        help="'lexical' ranks sentences by query word-overlap (default, no extra "
             "model load). 'cpc' uses the local context-aware embedding model "
             "(replicate/cpc_compressor.py) -- slow to load, needs torch/transformers/peft. "
             "'gemini' calls a hosted Gemini model to do the compression itself.",
    )
    parser.add_argument(
        "--cpc-max-seq-length", type=int, default=1536,
        help="Per-chunk token budget for the CPC compressor's forward pass (only applies to "
             "--compressor cpc). The replicate/ default (6144) needs more VRAM than an 8GB "
             "card comfortably has for LongMemEval-sized haystacks -- once VRAM is nearly "
             "full, Windows silently pages GPU memory to system RAM, turning a ~1s forward "
             "pass into tens of minutes. 1536 keeps peak usage around 3GB (more, smaller "
             "chunks per haystack, but each one fast) -- raise it if you have a bigger GPU.",
    )
    parser.add_argument(
        "--cpc-preset",
        choices=["llama", "mistral"],
        default="llama",
        help="Which CPC base model + LoRA adapter to use (only applies to --compressor cpc; "
             "see core/cpcCompressor.py). 'llama' is unsloth/Llama-3.2-1B-Instruct -- runs on "
             "an 8GB laptop GPU. 'mistral' is mistralai/Mistral-7B-Instruct-v0.2 -- a full 7B "
             "model, well beyond an 8GB card; run it on Vertex AI instead (see eval/deploy.sh).",
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Also answer each question with no compression (the full raw haystack sent as "
             "context). Expensive: tens of thousands of tokens per call.",
    )
    parser.add_argument(
        "--log-dir", default=str(LOGS_DIR),
        help="Directory for the per-question JSONL log (one file per run, same convention as "
             "main.py's session logs). Written incrementally so a run can be inspected or "
             "recovered even if it's interrupted partway through.",
    )
    parser.add_argument(
        "--resume", type=int, default=None, metavar="RUN_NUMBER",
        help="Continue an existing run instead of starting a new one, e.g. --resume 10 "
             "continues eval_longmemeval_run0010.jsonl. Safe to Ctrl+C a long run and pick "
             "it back up later: this re-derives --limit/--seed/--question-types/--compressor/ "
             "etc. from that run's own run_start record (so it reproduces the exact same "
             "question sample regardless of what's passed this time), skips any "
             "(config, question) pair already logged, and keeps appending to the same file.",
    )
    parser.add_argument(
        "--project", default=PROJECT_ID,
        help="GCP project for Vertex AI. Defaults to GOOGLE_CLOUD_PROJECT, but a Vertex AI "
             "custom job's container isn't guaranteed to have that env var set, so it can be "
             "passed explicitly here instead (see eval/deploy.sh).",
    )
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument(
        "--log-gcs-uri", default=None, metavar="gs://bucket/prefix",
        help="Also mirror the JSONL log to GCS after every result, re-uploading the whole "
             "(small) file each time. For running on a Vertex AI custom job, whose local disk "
             "doesn't survive a restart or preemption -- without this, --resume has nothing to "
             "resume from if the job dies. Requires google-cloud-storage.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    existing_results = []
    if args.resume is not None:
        run_number = args.resume
        log_path = log_dir / f"eval_longmemeval_run{run_number:04d}.jsonl"

        if args.log_gcs_uri and not log_path.exists():
            # A fresh Vertex AI job container has no local disk history --
            # the log this resume needs only exists in GCS from the prior
            # (interrupted) attempt.
            blob = _gcs_blob(args.log_gcs_uri, log_path.name)
            if blob.exists():
                print(f"Local log missing; downloading from {args.log_gcs_uri}/{log_path.name}")
                blob.download_to_filename(str(log_path))

        run_start, existing_results = load_run_log(log_path)
        if run_start is None:
            raise SystemExit(f"--resume {run_number}: {log_path} has no run_start record to resume from")

        # The original run's own settings win: a resumed run must reproduce
        # the exact same item sample, or the "already completed" skip-set
        # would be checked against the wrong items.
        args.limit = run_start["limit"]
        args.seed = run_start["seed"]
        args.question_types = ",".join(run_start["question_types"]) if run_start.get("question_types") else None
        args.compressor = run_start.get("compressor", args.compressor)
        args.token_budget = run_start.get("token_budget", args.token_budget)
        args.sentence_threshold = run_start.get("sentence_threshold", args.sentence_threshold)
        args.recent_turns = run_start.get("recent_turns", args.recent_turns)
        if run_start.get("cpc_max_seq_length") is not None:
            args.cpc_max_seq_length = run_start["cpc_max_seq_length"]
        if run_start.get("cpc_preset") is not None:
            args.cpc_preset = run_start["cpc_preset"]
        args.compare_baseline = len(run_start.get("configs", [])) > 1
        print(f"Resuming run {run_number} ({log_path}): "
              f"{len(existing_results)} (config, question) results already logged")
    else:
        run_number = _next_run_number(log_dir, gcs_uri=args.log_gcs_uri)
        log_path = log_dir / f"eval_longmemeval_run{run_number:04d}.jsonl"
        print(f"Logging full per-question results to {log_path}")

    question_types = args.question_types.split(",") if args.question_types else None
    items = load_longmemeval(limit=args.limit, seed=args.seed, question_types=question_types)
    print(f"Loaded {len(items)} questions from {DATASET_REPO}:{DATASET_FILE}")

    agent = build_agent(project=args.project, location=args.location)

    tokenwise = build_tokenwise(
        args.compressor, cpc_max_seq_length=args.cpc_max_seq_length, cpc_preset=args.cpc_preset,
        project=args.project, location=args.location,
    )
    configs = [(f"compressed_{args.compressor}", tokenwise)]
    if args.compare_baseline:
        configs.append(("baseline_full_history", None))

    log_gcs_blob = _gcs_blob(args.log_gcs_uri, log_path.name) if args.log_gcs_uri else None

    already_done = {(row["config"], row["question_id"]) for row in existing_results}
    results = {name: [] for name, _ in configs}
    for row in existing_results:
        if row["config"] in results:
            results[row["config"]].append(row)

    with log_path.open("a", encoding="utf-8") as log_file:
        def log(record):
            log_file.write(json.dumps({"timestamp": datetime.now().isoformat(), **record}) + "\n")
            log_file.flush()
            if log_gcs_blob is not None:
                log_gcs_blob.upload_from_filename(str(log_path))

        log({
            "event": "run_resume" if args.resume is not None else "run_start",
            "run": run_number,
            "dataset": f"{DATASET_REPO}:{DATASET_FILE}",
            "limit": args.limit,
            "seed": args.seed,
            "question_types": question_types,
            "compressor": args.compressor,
            "cpc_preset": args.cpc_preset if args.compressor == "cpc" else None,
            "cpc_max_seq_length": args.cpc_max_seq_length if args.compressor == "cpc" else None,
            "token_budget": args.token_budget,
            "sentence_threshold": args.sentence_threshold,
            "recent_turns": args.recent_turns,
            "configs": [name for name, _ in configs],
        })

        for index, item in enumerate(items, start=1):
            pending_configs = [
                (name, tw) for name, tw in configs
                if (name, item["question_id"]) not in already_done
            ]
            if not pending_configs:
                continue  # every config for this question was already logged in a prior session

            print(f"\n[{index}/{len(items)}] {item['question_id']} ({item['question_type']}): {item['question'][:120]}")
            for name, tokenwise in pending_configs:
                try:
                    predicted, compression_mode, compressed_context = answer_question(
                        agent, item, tokenwise,
                        args.sentence_threshold, args.token_budget, args.recent_turns,
                    )
                except Exception as exc:
                    predicted, compression_mode, compressed_context = f"[ERROR: {exc}]", None, None

                correct = grade(predicted, item["answer"])
                row = {
                    "config": name,
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "reference": item["answer"],
                    "predicted": predicted,
                    "correct": correct,
                    "compression_mode": compression_mode,
                    "compressed_context": compressed_context,
                }
                results[name].append(row)
                log({"event": "result", **row})

                status = "PASS" if correct else "FAIL"
                print(f"  [{name}] {status} | ref: {item['answer']!r} | got: {predicted[:150]!r}")

        print("\n=== Summary ===")
        summary = {}
        for name, rows in results.items():
            summarize(name, rows)
            total = len(rows)
            passed = sum(row["correct"] for row in rows)
            by_type = {}
            for row in rows:
                by_type.setdefault(row["question_type"], []).append(row["correct"])
            summary[name] = {
                "total": total,
                "passed": passed,
                "by_type": {
                    question_type: {"total": len(outcomes), "passed": sum(outcomes)}
                    for question_type, outcomes in by_type.items()
                },
            }
        log({"event": "run_end", "summary": summary})

    print(f"\nFull results (including untruncated predictions) saved to {log_path}")


if __name__ == "__main__":
    main()
