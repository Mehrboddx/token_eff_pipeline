"""Where the time goes inside one cold CPC compression: GPU encoding vs
CPU work (sentence splitting, chunk packing, tokenizer calls, pooling).

Wraps the CPC compressor's internal steps with timers, then runs the
paper-faithful and our version on the same questions, cold (cache
cleared). Timers on nested steps overlap -- chunk packing includes the
token counting it does -- so the stages are not meant to sum; `total` is
the whole compress_question call.

Usage (on the GPU machine):
    python -m eval.profile_cpc --dataset longmemeval_s --limit 50 --out logs/profile_cpc_lme.jsonl
"""

import argparse
import functools
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime

from core.selection import SelectionConfig
from eval.latency import METHODS, _torch, clear_cache, timed
from eval.longmemeval import CompressionSettings, build_tokenwise, compress_question, load_items, seed_memory_from_haystack

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Compressor methods to time. _embed_with_spans is the only one on the GPU.
STAGES = {
    "_embed_with_spans": "gpu_encode",
    "_split_sentences": "sentence_split",
    "_expand_oversized": "expand_oversized",
    "_build_chunks": "chunk_packing",       # paper path
    "_pack_group": "chunk_packing",         # our path
    "_span_average": "span_pooling",
    "_count_tokens": "token_counting",
}


def instrument(compressor, totals):
    torch = _torch()
    for method, stage in STAGES.items():
        original = getattr(compressor, method, None)
        if original is None:
            continue

        def wrapper(*args, _original=original, _stage=stage, **kwargs):
            if torch and _stage == "gpu_encode":
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                if torch and _stage == "gpu_encode":
                    torch.cuda.synchronize()
                totals[_stage] += time.perf_counter() - start
                totals[_stage + "_calls"] += 1

        setattr(compressor, method, functools.wraps(original)(wrapper))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s", choices=["longmemeval_s", "locomo"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--methods", nargs="+", default=["paper-mistral", "ours-mistral"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = load_items(args.dataset, limit=args.limit, seed=args.seed)
    torch = _torch()
    gpu = torch.cuda.get_device_name(0) if torch else "cpu"
    print(f"{len(items)} questions from {args.dataset} on {gpu}")

    summary = defaultdict(lambda: defaultdict(list))
    with open(args.out, "a", encoding="utf-8") as out:
        for method in args.methods:
            build_kwargs, settings_kwargs, _ = METHODS[method]
            settings = CompressionSettings(token_budget=args.token_budget, **settings_kwargs)
            tokenwise = build_tokenwise(selection=SelectionConfig(), **build_kwargs)
            totals = defaultdict(float)
            instrument(tokenwise.model._compressor, totals)

            for index, item in enumerate(items[:args.warmup] + items):
                memory = seed_memory_from_haystack(item)
                clear_cache(tokenwise)
                # clear_cache swaps the cache object, not the compressor, so the timers stay in place.
                totals.clear()
                _, seconds = timed(lambda: compress_question(item, tokenwise, settings, seeded_memory=memory))
                if index < args.warmup:
                    continue
                stages = {key: round(value, 4) for key, value in totals.items()}
                stages["cpu_other"] = round(seconds - totals["gpu_encode"], 4)
                row = {"event": "profile", "dataset": args.dataset, "method": method, "gpu": gpu,
                       "question_id": item["question_id"], "total": round(seconds, 4), **stages}
                out.write(json.dumps({"timestamp": datetime.now().isoformat(), **row}) + "\n")
                out.flush()
                for key, value in row.items():
                    if isinstance(value, float):
                        summary[method][key].append(value)
                print(f"{method} {index - args.warmup + 1}/{len(items)} total {seconds:.1f}s "
                      f"gpu {totals['gpu_encode']:.1f}s", flush=True)

            del tokenwise
            if torch:
                import gc
                gc.collect()
                torch.cuda.empty_cache()

    for method, columns in summary.items():
        print(f"\n{method} (medians, seconds)")
        for key, values in sorted(columns.items()):
            if not key.endswith("_calls"):
                print(f"  {key:<18} {statistics.median(values):8.3f}")


if __name__ == "__main__":
    main()
