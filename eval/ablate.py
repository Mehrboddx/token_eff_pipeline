"""Selection ablations over LongMemEval, measured by evidence recall.

Seeds each haystack once, builds its compression candidates once (the same
Runner.compression_candidates the pipeline uses), then compresses it under
every selection config in the grid with the same TokenWise path -- so each
config is scored on identical inputs, and scoring itself (CPC's GPU pass)
is shared: with --score-cache, a question scored once is never re-encoded,
and a fully cached run never loads the model.

Usage:
    python -m eval.ablate --compressor lexical --limit 500
    python -m eval.ablate --compressor cpc --cpc-preset llama --score-cache logs/score_cache
    python -m eval.ablate --compressor cpc --score-cache logs/score_cache --decompositions logs/decompositions.json

Semantic-MMR configs need CPC sentence embeddings; with --score-cache the
top candidates' embeddings are stored beside the scores on first use.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from core.selection import SelectionConfig
from eval.evidence import exact_evidence_recall, summarize_recall
from eval.longmemeval import (
    build_tokenwise,
    load_decompositions,
    load_longmemeval,
    retrieval_queries,
    seed_memory_from_haystack,
    CompressionSettings,
    make_runner,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def grid(has_decompositions, has_embeddings):
    base = SelectionConfig()
    configs = [
        ("base", base, False),
        ("window1", replace(base, window=1), False),
        ("pair", replace(base, pair_turns=True), False),
        ("window1+pair", replace(base, window=1, pair_turns=True), False),
        ("top3", replace(base, top_sessions=3), False),
        ("top5", replace(base, top_sessions=5), False),
        ("top10", replace(base, top_sessions=10), False),
        ("mmr0.5", replace(base, mmr_lambda=0.5), False),
        ("mmr1", replace(base, mmr_lambda=1.0), False),
        ("mmr2", replace(base, mmr_lambda=2.0), False),
        *([
            ("mmr0.5-semantic", replace(base, mmr_lambda=0.5, mmr_similarity="semantic"), False),
            ("mmr1-semantic", replace(base, mmr_lambda=1.0, mmr_similarity="semantic"), False),
            ("mmr2-semantic", replace(base, mmr_lambda=2.0, mmr_similarity="semantic"), False),
        ] if has_embeddings else []),
        ("recency0.25", replace(base, recency_weight=0.25), False),
        ("recency0.5", replace(base, recency_weight=0.5), False),
        ("recency1", replace(base, recency_weight=1.0), False),
        ("hybrid0.25", replace(base, lexical_weight=0.25), False),
        ("hybrid0.5", replace(base, lexical_weight=0.5), False),
    ]
    if has_decompositions:
        configs += [("multiquery", base, True), ("multiquery+hybrid0.25", replace(base, lexical_weight=0.25), True)]
    return configs


def parse_configs(spec, has_decompositions):
    """--configs name=field:value,field:value;... for custom combinations
    (e.g. after the single-factor grid shows which knobs help)."""
    configs = []
    for part in filter(None, (piece.strip() for piece in spec.split(";"))):
        name, _, fields = part.partition("=")
        values = {}
        multi_query = False
        for assignment in filter(None, fields.split(",")):
            key, _, raw = assignment.partition(":")
            if key == "multiquery":
                multi_query = raw.lower() in ("1", "true", "yes")
                continue
            current = getattr(SelectionConfig(), key)
            values[key] = raw.lower() in ("1", "true", "yes") if isinstance(current, bool) else type(current)(raw)
        if multi_query and not has_decompositions:
            raise SystemExit(f"config {name!r} uses multiquery but no --decompositions were given")
        configs.append((name, SelectionConfig(**values), multi_query))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compressor", choices=["lexical", "cpc"], default="lexical")
    parser.add_argument("--cpc-preset", choices=["llama", "mistral"], default="llama")
    parser.add_argument("--cpc-max-seq-length", type=int, default=6144)
    parser.add_argument("--cpc-attention", choices=["bidirectional", "causal"], default="bidirectional")
    parser.add_argument("--score-cache", default=None, metavar="DIR")
    parser.add_argument("--decompositions", default=None, metavar="PATH")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-types", default=None)
    parser.add_argument("--token-budget", type=int, default=500)
    parser.add_argument("--sentence-threshold", type=int, default=20)
    parser.add_argument("--recent-turns", type=int, default=4)
    parser.add_argument("--min-unit-words", type=int, default=4)
    parser.add_argument("--configs", default=None,
                        help="Custom configs instead of the default single-factor grid, e.g. "
                             "\"a=window:1,pair_turns:true;b=top_sessions:5,multiquery:true\".")
    parser.add_argument("--out", default=None, help="Write the summary table as JSON here.")
    args = parser.parse_args()

    decompositions = load_decompositions(args.decompositions)
    configs = (
        parse_configs(args.configs, bool(decompositions)) if args.configs
        else grid(bool(decompositions), has_embeddings=args.compressor == "cpc")
    )
    settings = CompressionSettings(args.sentence_threshold, args.token_budget, args.recent_turns, args.min_unit_words)
    tokenwise = build_tokenwise(
        args.compressor, cpc_max_seq_length=args.cpc_max_seq_length, cpc_preset=args.cpc_preset,
        score_cache=args.score_cache, cpc_attention=args.cpc_attention,
    )

    question_types = args.question_types.split(",") if args.question_types else None
    items = load_longmemeval(limit=args.limit, seed=args.seed, question_types=question_types)
    print(f"{len(items)} questions x {len(configs)} configs, budget {args.token_budget}, compressor {args.compressor}")

    rows = {name: [] for name, _, _ in configs}
    started = time.monotonic()
    for index, item in enumerate(items, start=1):
        memory = seed_memory_from_haystack(item)
        compressible = len(memory.get_sentences()) > settings.sentence_threshold
        memory.add_user_message(item["question"])
        runner = make_runner(None, memory, tokenwise, settings)
        candidates = runner.compression_candidates() if compressible else None

        for name, config, multi_query in configs:
            if candidates is None:
                evidence = exact_evidence_recall(memory, item, None, "full_history")
                rows[name].append({"question_type": item["question_type"], "evidence": evidence, "compressed_tokens": None})
                continue
            units, cutoff = candidates
            queries = retrieval_queries(item, decompositions if multi_query else None)
            tokenwise.selection = config
            result = tokenwise.compress_units(queries[0], settings.token_budget, units, extra_queries=queries[1:])
            last = {"cutoff_index": cutoff, "result": result, "candidates": len(units)}
            rows[name].append({
                "question_type": item["question_type"],
                "evidence": exact_evidence_recall(memory, item, last, "compressed_history"),
                "compressed_tokens": result.tokens,
            })

        if index % 25 == 0 or index == len(items):
            elapsed = time.monotonic() - started
            print(f"  {index}/{len(items)} questions ({elapsed:.0f}s, ~{elapsed / index * (len(items) - index):.0f}s left)",
                  flush=True)

    summaries = {}
    for name, config, multi_query in configs:
        summaries[name] = {
            "selection": asdict(config),
            "multi_query": multi_query,
            "recall": summarize_recall(name, rows[name], verbose=False),
        }

    focus = ["temporal-reasoning", "knowledge-update", "multi-session"]
    print(f"\n{'config':24} {'turn':>6} {'all-ev':>6} {'sess':>6} {'tokens':>6}   " + "  ".join(f"{t[:10]:>10}" for t in focus))
    for name, summary in summaries.items():
        overall = summary["recall"]["overall"]
        by_type = summary["recall"]["by_type"]
        cells = "  ".join(f"{(by_type.get(t) or {}).get('turn_recall') or 0:10.1%}" for t in focus)
        print(f"{name:24} {overall['turn_recall']:6.1%} {overall['all_evidence_kept']:6.1%} "
              f"{overall['session_recall']:6.1%} {overall['mean_tokens']:6.0f}   {cells}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "configs": summaries}, indent=1), encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
