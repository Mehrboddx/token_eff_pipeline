"""Answer-model latency: Gemini answering from the full history vs from a
compressed context, on the same questions. Gemini calls only, no local
model, so it can run anywhere with Vertex AI credentials.

The compressed contexts are read from an existing compress-stage log;
the full history is rebuilt from the dataset. Each question is answered
once from each, alternating which goes first so drift in API latency
falls on both sides equally. Calls are sequential.

Usage:
    python -m eval.answer_latency --dataset longmemeval_s --limit 50 \
        --compressed-log logs/eval_longmemeval_run0001_compress_cpc-mistral_b4000.jsonl \
        --out logs/answer_latency_lme.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime

from google.genai import types

from eval.longmemeval import CompressionSettings, build_agent, compress_question, load_items

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def compressed_histories(path):
    histories = {}
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if record.get("event") == "result" and record.get("history"):
            histories[record["question_id"]] = record["history"]
    return histories


def to_contents(described):
    return [types.Content(role=entry["role"], parts=[types.Part.from_text(text=entry["text"])])
            for entry in described]


def timed_call(agent, described):
    start = time.perf_counter()
    response = agent.call_model(to_contents(described))
    seconds = time.perf_counter() - start
    usage = getattr(response, "usage_metadata", None)
    return {
        "seconds": round(seconds, 3),
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s", choices=["longmemeval_s", "locomo"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compressed-log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    compressed = compressed_histories(args.compressed_log)
    items = [item for item in load_items(args.dataset, limit=args.limit, seed=args.seed)
             if item["question_id"] in compressed]
    agent = build_agent()
    print(f"{len(items)} questions from {args.dataset}")

    rows = []
    with open(args.out, "a", encoding="utf-8") as out:
        for index, item in enumerate(items):
            full, _, _, _ = compress_question(item, None, CompressionSettings())
            contexts = [("full", full), ("compressed", compressed[item["question_id"]])]
            if index % 2:
                contexts.reverse()
            for name, described in contexts:
                try:
                    result = timed_call(agent, described)
                except Exception as error:  # one failed call shouldn't end the run
                    result = {"error": str(error)[:300]}
                row = {"event": "answer_latency", "dataset": args.dataset, "question_id": item["question_id"],
                       "context": name, **result}
                rows.append(row)
                out.write(json.dumps({"timestamp": datetime.now().isoformat(), **row}) + "\n")
                out.flush()
            print(f"{index + 1}/{len(items)}", flush=True)

    groups = defaultdict(list)
    for row in rows:
        if "seconds" in row:
            groups[row["context"]].append(row)
    for name, group in groups.items():
        print(f"{name:<11} n={len(group)} median {statistics.median(r['seconds'] for r in group):.2f}s "
              f"median prompt tokens {statistics.median(r['prompt_tokens'] or 0 for r in group):.0f}")


if __name__ == "__main__":
    main()
