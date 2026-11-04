"""Compression latency per question, on one GPU, for each compressor in
the paper. Two measurements per question:

- cold: every encoder cache cleared first, so the whole history is
  encoded -- the cost of compressing a history seen for the first time;
- incremental (CPC with our units only): the history minus its newest
  session is compressed first, untimed, then the full history is timed.
  Only chunks that changed are re-encoded -- what an agent pays on each
  call once its earlier sessions are cached (change 13).

Storing the history (sentence splitting, done once when a message
arrives) is timed separately as seed_seconds and is not part of either.
Model loading is timed once per method. Run with nothing else on the
GPU; the first --warmup questions of each method are run but not logged.

Usage (on the GPU machine):
    python -m eval.latency --dataset longmemeval_s --limit 50 --out logs/latency_lme.jsonl
    python -m eval.latency --dataset locomo --limit 30 --out logs/latency_locomo.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime

import tiktoken

from core.selection import SelectionConfig
from eval.longmemeval import (
    CompressionSettings,
    build_tokenwise,
    compress_question,
    load_items,
    seed_memory_from_haystack,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# name -> (build_tokenwise kwargs, CompressionSettings kwargs, incremental?)
METHODS = {
    "ours-mistral": ({"compressor": "cpc", "cpc_preset": "mistral"}, {}, True),
    "ours-llama": ({"compressor": "cpc", "cpc_preset": "llama"}, {}, True),
    "paper-mistral": ({"compressor": "cpc", "cpc_preset": "mistral"}, {"paper": True}, False),
    "provence": ({"compressor": "provence", "provence": {}}, {}, False),
    "dycp": ({"compressor": "dycp", "dycp": {}}, {}, False),
}

# Item keys that hold one entry per session, trimmed together.
SESSION_KEYS = ("haystack_sessions", "haystack_dates", "haystack_session_ids", "session_time_labels")

ENCODING = tiktoken.get_encoding("cl100k_base")


def _torch():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def timed(fn):
    torch = _torch()
    if torch:
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if torch:
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def cpc_cache(tokenwise):
    """The CPC encoder's per-chunk embedding cache, or None for other compressors."""
    inner = getattr(tokenwise.model, "_compressor", None)
    return getattr(inner, "embedding_cache", None)


def clear_cache(tokenwise):
    inner = getattr(tokenwise.model, "_compressor", None)
    cache = cpc_cache(tokenwise)
    if cache is not None:
        inner.embedding_cache = type(cache)(cache.max_bytes)


def without_last_session(item):
    trimmed = dict(item)
    for key in SESSION_KEYS:
        if item.get(key) is not None:
            trimmed[key] = item[key][:-1]
    return trimmed


def history_tokens(item, settings):
    """cl100k tokens of the full history as the full-history baseline
    renders it (labels, date tags, captions) -- the size the paper reports."""
    described, _, _, _ = compress_question(item, None, CompressionSettings(token_budget=settings.token_budget))
    return sum(len(ENCODING.encode(entry["text"])) for entry in described)


def measure(item, tokenwise, settings, incremental):
    """One row per measurement for this item."""
    memory, seed_seconds = timed(lambda: seed_memory_from_haystack(item))
    base = {
        "question_id": item["question_id"],
        "question_type": item.get("question_type"),
        "sessions": len(item["haystack_sessions"]),
        "history_tokens": history_tokens(item, settings),
        "seed_seconds": round(seed_seconds, 4),
    }

    clear_cache(tokenwise)
    (_, _, _, info), seconds = timed(lambda: compress_question(item, tokenwise, settings, seeded_memory=memory))
    rows = [{**base, "mode": "cold", "seconds": round(seconds, 4),
             "compressed_tokens": info.get("compressed_tokens"), "units": info.get("candidate_units")}]

    if incremental and len(item["haystack_sessions"]) > 1:
        clear_cache(tokenwise)
        compress_question(without_last_session(item), tokenwise, settings)
        cache = cpc_cache(tokenwise)
        hits, misses = cache.hits, cache.misses
        (_, _, _, info), seconds = timed(lambda: compress_question(item, tokenwise, settings, seeded_memory=memory))
        rows.append({**base, "mode": "incremental", "seconds": round(seconds, 4),
                     "compressed_tokens": info.get("compressed_tokens"), "units": info.get("candidate_units"),
                     "chunks_reused": cache.hits - hits, "chunks_encoded": cache.misses - misses})
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["mode"])].append(row)
    print(f"\n{'method':<15} {'mode':<12} {'n':>4} {'median s':>9} {'p90 s':>8} {'ms / 1k hist tok':>17}")
    for (method, mode), group in sorted(groups.items()):
        seconds = sorted(row["seconds"] for row in group)
        per_k = statistics.median(1000 * row["seconds"] / (row["history_tokens"] / 1000) for row in group)
        p90 = seconds[min(len(seconds) - 1, int(0.9 * len(seconds)))]
        print(f"{method:<15} {mode:<12} {len(group):>4} {statistics.median(seconds):>9.3f} {p90:>8.3f} {per_k:>17.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s", choices=["longmemeval_s", "locomo"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", required=True, help="JSONL file, one row per measurement (appended).")
    args = parser.parse_args()

    items = load_items(args.dataset, limit=args.limit, seed=args.seed)
    torch = _torch()
    gpu = torch.cuda.get_device_name(0) if torch else "cpu"
    print(f"{len(items)} questions from {args.dataset} on {gpu}")

    rows = []
    with open(args.out, "a", encoding="utf-8") as out:
        def write(record):
            out.write(json.dumps({"timestamp": datetime.now().isoformat(), **record}) + "\n")
            out.flush()

        for method in args.methods:
            build_kwargs, settings_kwargs, incremental = METHODS[method]
            settings = CompressionSettings(token_budget=args.token_budget, **settings_kwargs)
            tokenwise, load_seconds = timed(lambda: build_tokenwise(selection=SelectionConfig(), **build_kwargs))
            if torch:
                torch.cuda.reset_peak_memory_stats()
            for item in items[:args.warmup]:
                measure(item, tokenwise, settings, incremental)

            for index, item in enumerate(items, 1):
                for row in measure(item, tokenwise, settings, incremental):
                    row = {"event": "latency", "dataset": args.dataset, "method": method, "gpu": gpu,
                           "token_budget": args.token_budget, **row}
                    rows.append(row)
                    write(row)
                print(f"{method} {index}/{len(items)} {rows[-1]['mode']} {rows[-1]['seconds']:.2f}s", flush=True)

            peak = torch.cuda.max_memory_allocated() / 2**30 if torch else None
            write({"event": "method_done", "dataset": args.dataset, "method": method, "gpu": gpu,
                   "load_seconds": round(load_seconds, 2), "peak_gpu_gib": round(peak, 2) if peak else None})
            print(f"{method}: load {load_seconds:.1f}s, peak GPU memory {peak or 0:.1f} GiB")
            del tokenwise
            if torch:
                import gc
                gc.collect()
                torch.cuda.empty_cache()

    summarize(rows)


if __name__ == "__main__":
    main()
