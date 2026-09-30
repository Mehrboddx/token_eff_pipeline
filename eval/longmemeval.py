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
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Windows' console/redirect default codepage (cp1252) can't encode plenty of
# characters that show up in ordinary Gemini answers (curly quotes, em dashes,
# accented names) -- without this, a print() partway through a long run raises
# UnicodeEncodeError and kills the process even though the result was already
# written to the JSONL log.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from google.genai import types
from huggingface_hub import hf_hub_download

from core.agent import Agent
from core.cpcCompressor import BASE_MODELS as CPC_BASE_MODELS
from core.cpcCompressor import CPCCompressor, cpc_cache_key
from core.geminiCompressor import GeminiCompressor
from core.memory import Memory
from core.runner import Runner
from core.selection import SelectionConfig
from core.tokenWise import TokenWise
from eval.evidence import exact_evidence_recall, summarize_recall, text_matched_evidence_recall
from eval.score_cache import CachedScorer
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

    Each turn records how many days before the question it happened
    (from haystack_dates and question_date). Without this, temporal
    questions ("how many days ago...") are unanswerable in principle —
    sessions mostly use relative phrasing ("today", "last week").

    The date travels as metadata, not text: the raw history shown to the
    answering model (recent turns, the uncompressed baseline) gets it as a
    "(from N days before this question)" prefix, while the sentences a
    compressor scores never contain it -- the compressor's renderer adds it
    back once per selected session. Gluing the tag into the text used to
    let the sentence splitter cut it off into a standalone "sentence" that
    was almost free under the budget and scored well on any date-flavored
    query: 36-55% of the CPC compression budget on temporal-reasoning and
    knowledge-update questions went to bare tags.

    `session` is the session's position, never its id: LongMemEval ids
    for answer-bearing sessions start with "answer_", which would leak
    the label if anything ever rendered it."""
    memory = Memory()
    dates = item.get("haystack_dates") or []
    session_ids = item.get("haystack_session_ids") or []
    question_date = item.get("question_date")
    question_dt = None
    if question_date:
        try:
            question_dt = _parse_haystack_date(question_date)
        except ValueError:
            question_dt = None

    for session_index, session in enumerate(item["haystack_sessions"]):
        time_label = None
        if question_dt is not None and session_index < len(dates):
            try:
                # Calendar days, not elapsed 24h periods: flooring a
                # timedelta drops a day whenever the session's clock time
                # is later than the question's, turning "7 days between"
                # into 6 on temporal questions.
                days_before = (question_dt.date() - _parse_haystack_date(dates[session_index]).date()).days
                time_label = f"{days_before} days before this question"
            except ValueError:
                time_label = None
        prefix = f"(from {time_label}) " if time_label else ""

        for turn in session:
            content = (turn.get("content") or "").strip()
            if not content:
                continue

            meta = {
                "session": session_index,
                "time": time_label,
                "has_answer": bool(turn.get("has_answer")),
                "session_id": session_ids[session_index] if session_index < len(session_ids) else None,
            }
            if turn.get("role") == "user":
                memory.add_user_message(content, meta=meta, prefix=prefix)
            elif turn.get("role") == "assistant":
                memory.add_model_content(
                    types.Content(role="model", parts=[types.Part.from_text(text=prefix + content)]),
                    meta=meta,
                    scored_text=content,
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


def build_tokenwise(compressor: str, cpc_max_seq_length: int = 6144, cpc_preset: str = "llama",
                     project=PROJECT_ID, location=LOCATION, selection=None, score_cache=None,
                     cpc_attention="bidirectional"):
    if compressor == "none":
        return None

    if compressor == "gemini":
        return TokenWise(model=GeminiCompressor(project=project, location=location, model=MODEL), selection=selection)

    if compressor == "cpc":
        def factory():
            return CPCCompressor(preset=cpc_preset, max_seq_length=cpc_max_seq_length, attention=cpc_attention)

        model = (
            CachedScorer(factory, score_cache, cpc_cache_key(cpc_preset, cpc_max_seq_length, cpc_attention))
            if score_cache else factory()
        )
        return TokenWise(model=model, selection=selection)

    return TokenWise(selection=selection)


@dataclass
class CompressionSettings:
    sentence_threshold: int = 20
    token_budget: int = 500
    recent_turns: int = 4
    min_unit_words: int = 4


def make_runner(agent, memory, tokenwise, settings):
    return Runner(
        agent, memory, tokenwise=tokenwise,
        compression_sentence_threshold=settings.sentence_threshold,
        compression_token_budget=settings.token_budget,
        recent_turns=settings.recent_turns,
        min_unit_words=settings.min_unit_words,
    )


NEEDS_QUESTION_DATE = {"temporal-reasoning", "knowledge-update"}


def _question_with_date_tag(item):
    """What the answering model is asked. Only question types that need
    "now" (elapsed time, most recent version of a fact) get today's date --
    kept identical to earlier runs so answer-side prompts stay comparable.
    Compressors never score against this text; see retrieval_queries."""
    question = item["question"]
    if item.get("question_date") and item.get("question_type") in NEEDS_QUESTION_DATE:
        question = f"(Today's date is {item['question_date']}.) {question}"
    return question


def retrieval_queries(item, decompositions=None):
    """What compressors score against: the bare question (a date prefix
    would pull every date-bearing sentence up the ranking), plus any
    sub-queries from --decompositions for multi-query scoring."""
    return [item["question"], *((decompositions or {}).get(item["question_id"]) or [])]


def compression_info(runner, memory, item, mode):
    last = runner.last_compression
    return {
        "compressed_tokens": last["result"].tokens if last else None,
        "candidate_units": last["candidates"] if last else None,
        "evidence": exact_evidence_recall(memory, item, last, mode),
    }


def _compressed_context_from_described(described, mode):
    if mode == "compressed_history" and described:
        first = described[0]
        if first.get("kind") == "compressed_context":
            return first.get("text")
    return None


def compress_question(item, tokenwise, settings, decompositions=None, seeded_memory=None):
    """The compression half of answer_question, split out so it can run
    without ever touching the answering model -- see --stage compress.
    Lets compression (GPU-bound, e.g. Mistral) run on one machine, and
    answering (one Gemini API call, no GPU) run on another that has
    working Vertex AI credentials but no local GPU at all.

    `seeded_memory` lets a caller that compresses the same item many times
    (eval/ablate.py) seed it once; it's copied, never mutated.

    Returns (described_history, mode, question, info) -- described_history
    is Agent.describe_history()'s JSON-serializable [{role, kind, text}, ...]
    form, mode is "compressed_history"/"full_history", question is the
    (possibly date-tagged) text the answering model gets, and info holds
    compressed-token counts and evidence recall."""
    memory = seeded_memory.copy() if seeded_memory is not None else seed_memory_from_haystack(item)
    runner = make_runner(None, memory, tokenwise, settings)
    question = _question_with_date_tag(item)

    previous_sentences = memory.get_sentences()
    memory.add_user_message(question)
    history, mode = runner._build_history(question, previous_sentences, retrieval_queries(item, decompositions))

    return Agent.describe_history(history), mode, question, compression_info(runner, memory, item, mode)


def answer_from_compressed(agent, described_history):
    """The answering half of answer_question, split out so it can run on a
    different machine than the one that produced described_history (see
    --stage answer) -- one Gemini call, no compressor, no GPU. Rebuilds
    real Content objects from the described (role/kind/text) form that
    compress_question logged; the question itself is already the last
    entry (compress_question adds it to Memory before building history),
    so nothing else needs to be passed in."""
    history = [
        types.Content(role=entry["role"], parts=[types.Part.from_text(text=entry["text"])])
        for entry in described_history
    ]
    response = agent.call_model(history)
    return response.text or ""


def answer_question(agent, item, tokenwise, settings, decompositions=None):
    memory = seed_memory_from_haystack(item)
    runner = make_runner(agent, memory, tokenwise, settings)
    question = _question_with_date_tag(item)

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
        response = runner.run(question, retrieval_queries=retrieval_queries(item, decompositions))
    finally:
        agent.context_monitor = None

    mode = captured.get("mode")
    compressed_context = _compressed_context_from_described(captured.get("history"), mode)
    return response.text or "", mode, compressed_context, compression_info(runner, memory, item, mode)


def summarize(name, rows):
    total = len(rows)
    passed = sum(row["correct"] for row in rows)
    print(f"\n{name}: {passed}/{total} ({passed / total:.1%})")

    by_type = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row["correct"])
    for question_type, outcomes in sorted(by_type.items()):
        print(f"  {question_type}: {sum(outcomes)}/{len(outcomes)} ({sum(outcomes) / len(outcomes):.1%})")


# "eval_longmemeval_run0039.jsonl" or, since settings tags were added,
# "eval_longmemeval_run0039_compress_cpc-mistral_b500_w1.jsonl".
RUN_LOG_PATTERN = re.compile(r"eval_longmemeval_run(\d+)(?:_[A-Za-z0-9._-]+)?\.jsonl$")


def run_tag(stage, settings):
    """Short, human-readable summary of a run's settings for its file name.
    Only a convenience for browsing: the authoritative settings are the
    run_start record inside the file (what --resume reads)."""
    parts = [stage]
    compressor = settings.get("compressor")
    if compressor == "cpc":
        parts.append(f"cpc-{settings.get('cpc_preset')}")
        if settings.get("cpc_attention") == "causal":
            parts.append("causal")
    elif compressor == "none":
        parts.append("fullhistory")
    elif compressor:
        parts.append(compressor)
    if compressor not in (None, "none") and settings.get("token_budget"):
        parts.append(f"b{settings['token_budget']}")

    selection = settings.get("selection") or {}
    if selection.get("window"):
        parts.append(f"w{selection['window']}")
    if selection.get("pair_turns"):
        parts.append("pair")
    if selection.get("top_sessions"):
        parts.append(f"top{selection['top_sessions']}")
    if selection.get("mmr_lambda"):
        parts.append(f"mmr{selection['mmr_lambda']:g}{'s' if selection.get('mmr_similarity') == 'semantic' else ''}")
    if selection.get("recency_weight"):
        parts.append(f"rec{selection['recency_weight']:g}")
    if selection.get("lexical_weight"):
        parts.append(f"hyb{selection['lexical_weight']:g}")
    if settings.get("decompositions"):
        parts.append("mq")
    return re.sub(r"[^A-Za-z0-9._-]", "-", "_".join(parts))


def new_log_path(log_dir, run_number, stage, settings):
    return log_dir / f"eval_longmemeval_run{run_number:04d}_{run_tag(stage, settings)}.jsonl"


def find_run_log(log_dir, run_number, gcs_uri=None):
    """The existing log for `run_number`, whatever its settings tag --
    downloaded from the GCS mirror first if it only exists there (a fresh
    Vertex AI container starts with an empty local logs/)."""
    local = [path for path in log_dir.glob(f"eval_longmemeval_run{run_number:04d}*.jsonl")
             if (match := RUN_LOG_PATTERN.match(path.name)) and int(match.group(1)) == run_number]
    if local:
        return local[0]

    if gcs_uri:
        from google.cloud import storage

        bucket_name, _, prefix = gcs_uri[len("gs://"):].partition("/")
        list_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
        for blob in storage.Client().list_blobs(bucket_name, prefix=f"{list_prefix}eval_longmemeval_run{run_number:04d}"):
            name = blob.name.rsplit("/", 1)[-1]
            match = RUN_LOG_PATTERN.match(name)
            if match and int(match.group(1)) == run_number:
                path = log_dir / name
                print(f"Local log missing; downloading from {gcs_uri}/{name}")
                blob.download_to_filename(str(path))
                return path

    raise SystemExit(f"--resume {run_number}: no eval_longmemeval_run{run_number:04d}*.jsonl in {log_dir}"
                     + (f" or {gcs_uri}" if gcs_uri else ""))


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


def _resolve_source_settings(from_log_path, from_run_start):
    """The compressor/token_budget/etc. that actually produced from_log's
    content -- an 'answer' or 'judge' stage log has no such settings of its
    own (it only takes --from-log), so without this a log three stages deep
    (compress -> answer -> judge) would need each intermediate file opened
    by hand to find out what was actually run, exactly the problem this eval
    file's own logging was found to have (see conversation history: run
    0012-0026 needed manual backfilling for this). Chains through an
    already-resolved 'source_settings' on from_run_start when present (an
    answer log whose own from_log was a compress log) instead of stopping
    one hop short at that intermediate log's empty settings."""
    settings = dict(from_run_start["source_settings"]) if from_run_start.get("source_settings") else {
        key: from_run_start.get(key)
        for key in ("compressor", "cpc_preset", "cpc_max_seq_length", "cpc_attention", "token_budget",
                    "min_unit_words", "selection", "decompositions")
    }
    settings["resolved_from"] = str(from_log_path)
    return settings


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


JUDGE_PROMPT_TEMPLATE = (
    "You are grading whether an AI assistant's response correctly answers a "
    "question, given a reference answer.\n\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Assistant's response: {predicted}\n\n"
    "Grade the response as CORRECT if it conveys the same essential fact(s) as "
    "the reference answer, even if worded differently, abbreviated, or phrased "
    "in a different grammatical person (e.g. \"your sister\" vs \"my sister\" "
    "both refer to the same person from the assistant's point of view). Minor "
    "omissions of extra detail are fine as long as the core answer matches.\n\n"
    "If the reference answer says the information was not mentioned/available "
    "(an abstention case), grade the response as CORRECT if it also indicates "
    "it doesn't know or can't find that specific information -- it does not "
    "need to repeat any other distractor details the reference happens to "
    "mention.\n\n"
    "Grade as INCORRECT if the response states a different fact than the "
    "reference, confidently answers when it should have abstained, or fails "
    "to answer when the reference expects a specific answer.\n\n"
    "Respond with exactly one word: CORRECT or INCORRECT."
)


def llm_judge_grade(client, question, reference, predicted):
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, reference=reference, predicted=predicted)
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(temperature=0),
    )
    verdict = (response.text or "").strip().upper()
    return verdict.startswith("CORRECT"), verdict


def run_judge_stage(args, log_dir):
    """--stage judge: re-grade an already-answered log's predicted/reference
    pairs with an LLM judge instead of the heuristic text-match grader. Cheap
    -- one Gemini call per row, no compression or answering redone -- since
    every answered log already has everything a judge needs. Logs both
    verdicts side by side (heuristic_correct, llm_judge_correct) rather than
    replacing one with the other, so the two can be compared directly."""
    existing_results = []
    if args.resume is not None:
        run_number = args.resume
        log_path = find_run_log(log_dir, run_number, args.log_gcs_uri)
        run_start, existing_results = load_run_log(log_path)
        if run_start is None:
            raise SystemExit(f"--resume {run_number}: {log_path} has no run_start record to resume from")
        args.from_log = run_start.get("from_log", args.from_log)
        print(f"Resuming judge run {run_number} ({log_path}): "
              f"{len(existing_results)} results already logged")

    if not args.from_log:
        raise SystemExit("--stage judge requires --from-log <path to a --stage full/answer JSONL log>")

    from_log_path = Path(args.from_log)
    from_run_start, answered_rows = load_run_log(from_log_path)
    answered_rows = [row for row in answered_rows if "predicted" in row]
    if not answered_rows:
        raise SystemExit(f"{from_log_path} has no answered result rows (predicted/reference) to judge")
    print(f"Loaded {len(answered_rows)} answered questions from {from_log_path}")
    source_settings = _resolve_source_settings(from_log_path, from_run_start)

    if args.resume is None:
        run_number = _next_run_number(log_dir, gcs_uri=args.log_gcs_uri)
        log_path = new_log_path(log_dir, run_number, "judge", source_settings)
        print(f"Logging full per-question results to {log_path}")

    from google import genai
    from google.genai import types as genai_types
    judge_client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
        http_options=genai_types.HttpOptions(timeout=120_000),
    )
    log_gcs_blob = _gcs_blob(args.log_gcs_uri, log_path.name) if args.log_gcs_uri else None

    already_done = {(row["config"], row["question_id"]) for row in existing_results}
    results = {}
    for row in existing_results:
        results.setdefault(row["config"], []).append(row)

    with log_path.open("a", encoding="utf-8") as log_file:
        def log(record):
            log_file.write(json.dumps({"timestamp": datetime.now().isoformat(), **record}) + "\n")
            log_file.flush()
            if log_gcs_blob is not None:
                log_gcs_blob.upload_from_filename(str(log_path))

        log({
            "event": "run_resume" if args.resume is not None else "run_start",
            "run": run_number,
            "stage": "judge",
            "model": MODEL,
            "from_log": str(from_log_path),
            "source_settings": source_settings,
            "project": args.project,
            "location": args.location,
        })

        for index, arow in enumerate(answered_rows, start=1):
            key = (arow["config"], arow["question_id"])
            if key in already_done:
                continue

            print(f"\n[{index}/{len(answered_rows)}] {arow['question_id']} ({arow['question_type']}): "
                  f"{arow['question'][:120]}", flush=True)
            try:
                judge_correct, verdict = llm_judge_grade(
                    judge_client, arow["question"], arow["reference"], arow["predicted"],
                )
            except Exception as exc:
                judge_correct, verdict = None, f"[ERROR: {exc}]"

            row = {
                "config": arow["config"],
                "question_id": arow["question_id"],
                "question_type": arow["question_type"],
                "question": arow["question"],
                "reference": arow["reference"],
                "predicted": arow["predicted"],
                "heuristic_correct": arow.get("correct"),
                "llm_judge_correct": judge_correct,
                "llm_judge_verdict": verdict,
            }
            results.setdefault(row["config"], []).append(row)
            log({"event": "result", **row})

            agreement = "" if judge_correct == row["heuristic_correct"] else "  <-- DISAGREES with heuristic"
            status = "CORRECT" if judge_correct else "INCORRECT"
            print(f"  [{row['config']}] judge={status}{agreement} | "
                  f"heuristic={'PASS' if row['heuristic_correct'] else 'FAIL'}", flush=True)

        print("\n=== Summary (LLM judge vs. heuristic grader) ===")
        summary = {}
        for name, rows in results.items():
            total = len(rows)
            judge_passed = sum(1 for row in rows if row.get("llm_judge_correct"))
            heuristic_passed = sum(1 for row in rows if row.get("heuristic_correct"))
            print(f"\n{name}: heuristic {heuristic_passed}/{total} ({heuristic_passed / total:.1%})  "
                  f"vs  LLM-judge {judge_passed}/{total} ({judge_passed / total:.1%})")
            by_type = {}
            for row in rows:
                by_type.setdefault(row["question_type"], []).append(row)
            for question_type, qrows in sorted(by_type.items()):
                n = len(qrows)
                h = sum(1 for row in qrows if row.get("heuristic_correct"))
                j = sum(1 for row in qrows if row.get("llm_judge_correct"))
                print(f"  {question_type}: heuristic {h}/{n} ({h / n:.1%})  vs  judge {j}/{n} ({j / n:.1%})")
            summary[name] = {"total": total, "heuristic_passed": heuristic_passed, "llm_judge_passed": judge_passed}
        log({"event": "run_end", "summary": summary})

    print(f"\nFull results saved to {log_path}")


def run_answer_stage(args, log_dir):
    """--stage answer: read a --stage compress log's described histories
    and do just the Gemini call + grading for each -- no GPU, no
    compressor construction, needs only Vertex AI credentials. Mirrors the
    run-numbering/resume/GCS-mirroring shape of the main compress/full
    loop below, but iterates compress-stage rows instead of dataset items,
    since there's no dataset load or compressor to build here at all."""
    existing_results = []
    if args.resume is not None:
        run_number = args.resume
        log_path = find_run_log(log_dir, run_number, args.log_gcs_uri)
        run_start, existing_results = load_run_log(log_path)
        if run_start is None:
            raise SystemExit(f"--resume {run_number}: {log_path} has no run_start record to resume from")
        # --from-log only needs to be passed explicitly for a fresh answer
        # run -- a resumed one already has it recorded in its own
        # run_start, from when it was first started.
        args.from_log = run_start.get("from_log", args.from_log)
        print(f"Resuming answer run {run_number} ({log_path}): "
              f"{len(existing_results)} results already logged")

    if not args.from_log:
        raise SystemExit("--stage answer requires --from-log <path to a --stage compress JSONL log>")

    from_log_path = Path(args.from_log)
    from_run_start, compress_rows = load_run_log(from_log_path)
    if not compress_rows:
        raise SystemExit(f"{from_log_path} has no compress-stage result rows to answer")
    print(f"Loaded {len(compress_rows)} compressed questions from {from_log_path}")
    source_settings = _resolve_source_settings(from_log_path, from_run_start)

    if args.resume is None:
        run_number = _next_run_number(log_dir, gcs_uri=args.log_gcs_uri)
        log_path = new_log_path(log_dir, run_number, "answer", source_settings)
        print(f"Logging full per-question results to {log_path}")

    agent = build_agent(project=args.project, location=args.location)
    log_gcs_blob = _gcs_blob(args.log_gcs_uri, log_path.name) if args.log_gcs_uri else None

    already_done = {(row["config"], row["question_id"]) for row in existing_results}
    results = {}
    for row in existing_results:
        results.setdefault(row["config"], []).append(row)

    with log_path.open("a", encoding="utf-8") as log_file:
        def log(record):
            log_file.write(json.dumps({"timestamp": datetime.now().isoformat(), **record}) + "\n")
            log_file.flush()
            if log_gcs_blob is not None:
                log_gcs_blob.upload_from_filename(str(log_path))

        log({
            "event": "run_resume" if args.resume is not None else "run_start",
            "run": run_number,
            "stage": "answer",
            "model": MODEL,
            "from_log": str(from_log_path),
            "source_settings": source_settings,
            "project": args.project,
            "location": args.location,
        })

        for index, crow in enumerate(compress_rows, start=1):
            key = (crow["config"], crow["question_id"])
            if key in already_done:
                continue

            print(f"\n[{index}/{len(compress_rows)}] {crow['question_id']} ({crow['question_type']}): "
                  f"{crow['question'][:120]}", flush=True)
            try:
                predicted = answer_from_compressed(agent, crow["history"])
            except Exception as exc:
                predicted = f"[ERROR: {exc}]"

            correct = grade(predicted, crow["reference"])
            compressed_context = _compressed_context_from_described(crow.get("history"), crow.get("compression_mode"))
            row = {
                "config": crow["config"],
                "question_id": crow["question_id"],
                "question_type": crow["question_type"],
                "question": crow["question"],
                "reference": crow["reference"],
                "predicted": predicted,
                "correct": correct,
                "compression_mode": crow.get("compression_mode"),
                "compressed_context": compressed_context,
                "compressed_tokens": crow.get("compressed_tokens"),
                "evidence": crow.get("evidence"),
            }
            results.setdefault(row["config"], []).append(row)
            log({"event": "result", **row})

            status = "PASS" if correct else "FAIL"
            print(f"  [{row['config']}] {status} | ref: {crow['reference']!r} | got: {predicted[:150]!r}", flush=True)

        print("\n=== Summary ===")
        summary = {}
        for name, rows in results.items():
            summarize(name, rows)
            summarize_recall(name, rows)
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


def selection_from_args(args):
    return SelectionConfig(
        lexical_weight=args.lexical_weight,
        recency_weight=args.recency_weight,
        mmr_lambda=args.mmr_lambda,
        mmr_similarity=args.mmr_similarity,
        top_sessions=args.top_sessions,
        window=args.window,
        pair_turns=args.pair_turns,
    )


def load_decompositions(path):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe_evidence(info):
    evidence = (info or {}).get("evidence")
    if not evidence or not evidence["evidence_turns"]:
        return ""
    tokens = info.get("compressed_tokens")
    return (f"[evidence {evidence['evidence_turns_kept']}/{evidence['evidence_turns']} turns"
            f"{f', {tokens} tok' if tokens is not None else ''}]")


def run_recall_stage(args):
    """--stage recall: text-matched evidence recall for every row of an
    existing log -- including logs written before compressors recorded
    their selections, so old and new runs can be compared on the same
    footing. No model calls; only the dataset's evidence labels."""
    if not args.from_log:
        raise SystemExit("--stage recall requires --from-log <path to a compress/answer/full log>")

    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-4")
    from_log_path = Path(args.from_log)
    run_start, rows = load_run_log(from_log_path)
    recent_turns = (run_start or {}).get("recent_turns") or args.recent_turns
    items = {item["question_id"]: item for item in load_longmemeval()}
    print(f"Text-matched evidence recall for {from_log_path} ({len(rows)} rows, recent_turns={recent_turns})")

    by_config = {}
    for row in rows:
        mode = row.get("compression_mode")
        item = items.get(row["question_id"])
        if mode is None or item is None:
            continue
        context = row.get("compressed_context")
        if context is None:
            context = _compressed_context_from_described(row.get("history"), mode)
        context = (context or "").removeprefix("Relevant earlier context:\n")
        by_config.setdefault(row["config"], []).append({
            "question_type": row["question_type"],
            "evidence": text_matched_evidence_recall(item, context, mode, recent_turns),
            "compressed_tokens": len(encoding.encode(context, disallowed_special=())) if mode == "compressed_history" else None,
        })

    for name, measured in by_config.items():
        summarize_recall(f"{name} (text-matched)", measured)


DECOMPOSE_PROMPT = (
    "A retrieval system searches a user's long chat history with an AI assistant to answer "
    "the question below. Rewrite the question as 1 to 3 short, standalone search queries, one "
    "per distinct fact that must be found in the history (for example: each event whose date "
    "is needed, each item being counted or compared, the old and the new value of something "
    "that changed). If one fact suffices, return a single query. Do not answer the question.\n"
    "Return only a JSON array of strings.\n\nQuestion: {question}"
)


def run_decompose_stage(args):
    """--stage decompose: one Gemini call per sampled question, producing
    the sub-queries --decompositions feeds to multi-query scoring. Written
    after every question and resumable: already-decomposed ids are skipped."""
    if not args.decompositions:
        raise SystemExit("--stage decompose requires --decompositions <output JSON path>")
    if not args.project:
        raise SystemExit("--stage decompose calls Gemini: pass --project (or set GOOGLE_CLOUD_PROJECT)")

    from google import genai

    path = Path(args.decompositions)
    decompositions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    question_types = args.question_types.split(",") if args.question_types else None
    items = load_longmemeval(limit=args.limit, seed=args.seed, question_types=question_types)
    client = genai.Client(
        vertexai=True, project=args.project, location=args.location,
        http_options=types.HttpOptions(timeout=120_000),
    )
    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")

    for index, item in enumerate(items, start=1):
        if item["question_id"] in decompositions:
            continue
        prompt = DECOMPOSE_PROMPT.format(question=item["question"])
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config,
            )
            parsed = json.loads(response.text or "[]")
            queries = [q.strip() for q in parsed if isinstance(q, str) and q.strip()][:3]
        except Exception as exc:
            print(f"[{index}/{len(items)}] {item['question_id']}: failed ({exc}); skipping", flush=True)
            continue
        decompositions[item["question_id"]] = queries
        path.write_text(json.dumps(decompositions, indent=1), encoding="utf-8")
        print(f"[{index}/{len(items)}] {item['question'][:90]!r} -> {queries}", flush=True)

    print(f"\n{len(decompositions)} decompositions saved to {path}")


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
        choices=["lexical", "gemini", "cpc", "none"],
        default="lexical",
        help="'lexical' ranks sentences by query word-overlap (default, no extra "
             "model load). 'cpc' uses the local context-aware embedding model "
             "(replicate/cpc_compressor.py) -- slow to load, needs torch/transformers/peft. "
             "'gemini' calls a hosted Gemini model to do the compression itself. "
             "'none' disables compression entirely (tokenwise=None, full raw haystack sent "
             "as context every call) -- the uncompressed baseline, logged as "
             "'baseline_full_history' just like --compare-baseline's second config.",
    )
    parser.add_argument(
        "--cpc-max-seq-length", type=int, default=6144,
        help="Per-chunk token budget for the CPC compressor's forward pass (only applies to "
             "--compressor cpc). 6144 is the length the CPC LoRA was trained at "
             "(cpc-1.0-*.json), so most sessions fit in one chunk. On an 8GB card this "
             "can exhaust VRAM (Windows then pages GPU memory to system RAM, making each "
             "forward pass take minutes) -- pass 1536 there instead.",
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
        "--cpc-attention", choices=["bidirectional", "causal"], default="bidirectional",
        help="CPC encoder attention. 'bidirectional' is what the CPC LoRA was trained with; "
             "'causal' reproduces runs made before it was enforced (for the before/after comparison).",
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
    parser.add_argument(
        "--stage",
        choices=["full", "compress", "answer", "judge", "decompose", "recall"],
        default="full",
        help="'full' (default): compress and answer in one pass. 'compress': only run the "
             "compressor and log the resulting context -- no Gemini call, no GCP credentials "
             "needed at all -- for compressing (e.g. --compressor cpc --cpc-preset mistral) on "
             "a GPU machine that doesn't have Vertex AI access. 'answer': read a --stage "
             "compress log via --from-log and only do the Gemini call + grading -- no GPU or "
             "compressor needed -- for finishing the eval on a machine that does have Vertex AI "
             "access. 'judge': read a 'full' or 'answer' log via --from-log and re-grade its "
             "predicted/reference pairs with an LLM judge instead of the heuristic text-match "
             "grader -- cheap (one Gemini call per row, no compression or answering redone) and "
             "much less likely to mark a correct-but-differently-worded answer wrong. Left at "
             "the default ('full') with no --project/GOOGLE_CLOUD_PROJECT set, this "
             "automatically falls back to 'compress': Gemini isn't reachable anyway, so there's "
             "no reason to fail loudly instead of just doing the half that still works. "
             "'decompose': ask Gemini to split each sampled question into sub-queries and write "
             "them to --decompositions (for multi-query scoring). 'recall': text-matched "
             "evidence recall of an existing log's compressed contexts (--from-log) -- works on "
             "logs from before selections were recorded; no model calls.",
    )
    parser.add_argument(
        "--from-log", default=None, metavar="PATH",
        help="A --stage compress log to read compressed histories from (required for --stage "
             "answer), a --stage full/answer log to re-grade (required for --stage judge), or "
             "any compress/answer/full log to measure (--stage recall).",
    )
    selection = parser.add_argument_group(
        "selection", "How scored sentences become the compressed context (extractive compressors)."
    )
    selection.add_argument("--min-unit-words", type=int, default=4,
                           help="Merge sentence fragments shorter than this many words into a "
                                "neighbour from the same turn.")
    selection.add_argument("--lexical-weight", type=float, default=0.0,
                           help="Hybrid scoring: weight of BM25 in the fused score (0 = off).")
    selection.add_argument("--recency-weight", type=float, default=0.0,
                           help="Additive bonus (z units) for later sentences (0 = off).")
    selection.add_argument("--mmr", type=float, default=0.0, dest="mmr_lambda",
                           help="MMR redundancy penalty (z units per unit of similarity; 0 = off).")
    selection.add_argument("--mmr-similarity", choices=["lexical", "semantic"], default="lexical",
                           help="How MMR measures redundancy: word overlap, or cosine of CPC's own "
                                "sentence embeddings (centered on the candidate pool; --compressor cpc only).")
    selection.add_argument("--top-sessions", type=int, default=0,
                           help="Two-level selection: only select from the K best-scoring sessions (0 = off).")
    selection.add_argument("--window", type=int, default=0,
                           help="Also include +-N neighbouring sentences of the same turn.")
    selection.add_argument("--pair-turns", action="store_true",
                           help="Also include the best sentence of the partner turn (reply/prompt).")
    selection.add_argument("--decompositions", default=None, metavar="PATH",
                           help="JSON {question_id: [sub-queries]} (from --stage decompose): score "
                                "against each sub-query too, max-pooled.")
    selection.add_argument("--score-cache", default=None, metavar="DIR",
                           help="CPC only: cache scores on disk keyed by the exact candidate units, "
                                "so re-runs with different selection settings skip the GPU entirely.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        # A resumed run must continue in whatever stage it originally
        # started in, regardless of what --stage defaults to or whether
        # --project happens to be set this time -- peek at its own
        # run_start before deciding how to dispatch below.
        peek_start, _ = load_run_log(find_run_log(log_dir, args.resume, args.log_gcs_uri))
        if peek_start and peek_start.get("stage"):
            args.stage = peek_start["stage"]
    elif args.stage == "full" and not args.project:
        print(
            "No --project (and no GOOGLE_CLOUD_PROJECT env var) -- Gemini isn't reachable, so "
            "only running the compressor. Pass --project (or set GOOGLE_CLOUD_PROJECT) to also "
            "answer in one pass, or answer this run later on a machine that has it with:\n"
            "  python -m eval.longmemeval --stage answer --from-log <this run's log path> --project <project>\n"
        )
        args.stage = "compress"

    if args.stage == "answer":
        run_answer_stage(args, log_dir)
        return

    if args.stage == "judge":
        run_judge_stage(args, log_dir)
        return

    if args.stage == "recall":
        run_recall_stage(args)
        return

    if args.stage == "decompose":
        run_decompose_stage(args)
        return

    existing_results = []
    if args.resume is not None:
        run_number = args.resume
        log_path = find_run_log(log_dir, run_number, args.log_gcs_uri)
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
        for key, value in (run_start.get("selection") or {}).items():
            setattr(args, key, value)
        for key in ("min_unit_words", "decompositions", "score_cache", "cpc_attention"):
            if key in run_start:
                setattr(args, key, run_start[key])
        args.compare_baseline = len(run_start.get("configs", [])) > 1
        print(f"Resuming run {run_number} ({log_path}): "
              f"{len(existing_results)} (config, question) results already logged")
    else:
        run_number = _next_run_number(log_dir, gcs_uri=args.log_gcs_uri)
        log_path = new_log_path(log_dir, run_number, args.stage, {
            "compressor": args.compressor,
            "cpc_preset": args.cpc_preset,
            "cpc_attention": args.cpc_attention,
            "token_budget": args.token_budget,
            "selection": asdict(selection_from_args(args)),
            "decompositions": args.decompositions,
        })
        print(f"Logging full per-question results to {log_path}")

    question_types = args.question_types.split(",") if args.question_types else None
    items = load_longmemeval(limit=args.limit, seed=args.seed, question_types=question_types)
    print(f"Loaded {len(items)} questions from {DATASET_REPO}:{DATASET_FILE}")

    # --stage compress never calls the model, so it never needs Vertex AI
    # credentials -- skip building the Agent entirely rather than fail on a
    # missing/invalid project.
    agent = build_agent(project=args.project, location=args.location) if args.stage != "compress" else None

    selection_config = selection_from_args(args)
    settings = CompressionSettings(args.sentence_threshold, args.token_budget, args.recent_turns, args.min_unit_words)
    decompositions = load_decompositions(args.decompositions)
    tokenwise = build_tokenwise(
        args.compressor, cpc_max_seq_length=args.cpc_max_seq_length, cpc_preset=args.cpc_preset,
        project=args.project, location=args.location, selection=selection_config, score_cache=args.score_cache,
        cpc_attention=args.cpc_attention,
    )
    primary_name = "baseline_full_history" if args.compressor == "none" else f"compressed_{args.compressor}"
    configs = [(primary_name, tokenwise)]
    if args.compare_baseline and primary_name != "baseline_full_history":
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
            "stage": args.stage,
            # The model used wherever this run calls Gemini: answering
            # (build_agent), compression (build_tokenwise's "gemini" branch),
            # and/or LLM-judge grading -- all three share this one constant
            # today, but recording it here means a future change to MODEL
            # doesn't silently make older logs ambiguous about which model
            # produced them.
            "model": MODEL,
            "dataset": f"{DATASET_REPO}:{DATASET_FILE}",
            "limit": args.limit,
            "seed": args.seed,
            "question_types": question_types,
            "compressor": args.compressor,
            "cpc_preset": args.cpc_preset if args.compressor == "cpc" else None,
            "cpc_base_model": CPC_BASE_MODELS.get(args.cpc_preset) if args.compressor == "cpc" else None,
            "cpc_max_seq_length": args.cpc_max_seq_length if args.compressor == "cpc" else None,
            "cpc_attention": args.cpc_attention if args.compressor == "cpc" else None,
            "token_budget": args.token_budget,
            "sentence_threshold": args.sentence_threshold,
            "recent_turns": args.recent_turns,
            "min_unit_words": args.min_unit_words,
            "selection": asdict(selection_config),
            "decompositions": args.decompositions,
            "score_cache": args.score_cache,
            "configs": [name for name, _ in configs],
        })

        for index, item in enumerate(items, start=1):
            pending_configs = [
                (name, tw) for name, tw in configs
                if (name, item["question_id"]) not in already_done
            ]
            if not pending_configs:
                continue  # every config for this question was already logged in a prior session

            print(f"\n[{index}/{len(items)}] {item['question_id']} ({item['question_type']}): {item['question'][:120]}", flush=True)
            for name, tokenwise in pending_configs:
                if args.stage == "compress":
                    try:
                        described, mode, question, info = compress_question(item, tokenwise, settings, decompositions)
                    except Exception as exc:
                        described, mode, question, info = [], None, f"[ERROR: {exc}]", {}

                    row = {
                        "config": name,
                        "question_id": item["question_id"],
                        "question_type": item["question_type"],
                        "question": question,
                        "reference": item["answer"],
                        "compression_mode": mode,
                        "history": described,
                        **info,
                    }
                    results[name].append(row)
                    log({"event": "result", **row})
                    print(f"  [{name}] compressed (mode={mode}) {describe_evidence(info)}", flush=True)
                    continue

                try:
                    predicted, compression_mode, compressed_context, info = answer_question(
                        agent, item, tokenwise, settings, decompositions,
                    )
                except Exception as exc:
                    predicted, compression_mode, compressed_context, info = f"[ERROR: {exc}]", None, None, {}

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
                    **info,
                }
                results[name].append(row)
                log({"event": "result", **row})

                status = "PASS" if correct else "FAIL"
                print(f"  [{name}] {status} {describe_evidence(info)} | ref: {item['answer']!r} | "
                      f"got: {predicted[:150]!r}", flush=True)

        print("\n=== Summary ===")
        if args.stage == "compress":
            compressed_counts = {name: len(rows) for name, rows in results.items()}
            recall = {}
            for name, count in compressed_counts.items():
                print(f"{name}: compressed {count} questions")
                recall[name] = summarize_recall(name, results[name])
            print(f"\nAnswer these later (needs Vertex AI credentials, no GPU) with:")
            print(f"  python -m eval.longmemeval --stage answer --from-log {log_path} --project <project>")
            log({"event": "run_end", "compressed": compressed_counts, "evidence_recall": recall})
        else:
            summary = {}
            for name, rows in results.items():
                summarize(name, rows)
                summarize_recall(name, rows)
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
