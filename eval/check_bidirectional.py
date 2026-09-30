"""Sanity check that CPC really runs with bidirectional attention, then an
A/B of bidirectional vs. causal on real LongMemEval questions.

1. Mask test: embed two sentences that differ only in their LAST word.
   Under causal attention, earlier tokens can't see later ones, so their
   embeddings are identical; under bidirectional attention they differ.
2. Evidence recall of plain top-k selection at one budget, both modes,
   on the same questions (no Gemini calls).

The model is loaded once and switched between modes in place.

Usage:
    python -m eval.check_bidirectional --cpc-preset mistral
    python -m eval.check_bidirectional --cpc-preset mistral --limit 50 --token-budget 500
"""

import argparse
import sys

import torch

from core.cpcCompressor import CPCCompressor
from core.tokenWise import TokenWise
from eval.evidence import exact_evidence_recall, summarize_recall
from eval.longmemeval import CompressionSettings, load_longmemeval, make_runner, seed_memory_from_haystack

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def set_attention(adapter, mode):
    inner = adapter._compressor
    inner.attention = mode
    for module in inner.model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = mode == "causal"
    inner.embedding_cache = type(inner.embedding_cache)(inner.embedding_cache.max_bytes)


@torch.no_grad()
def mask_test(adapter):
    inner = adapter._compressor
    a, _ = inner._embed_with_spans("The meeting with the design team moved to Thursday afternoon")
    b, _ = inner._embed_with_spans("The meeting with the design team moved to Friday morning")
    shared = min(a.shape[0], b.shape[0]) - 3
    return (a[1:shared] - b[1:shared]).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpc-preset", choices=["llama", "mistral"], default="mistral")
    parser.add_argument("--cpc-max-seq-length", type=int, default=6144)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=500)
    args = parser.parse_args()

    adapter = CPCCompressor(preset=args.cpc_preset, max_seq_length=args.cpc_max_seq_length)

    print("\n== 1. Mask test (max |diff| of early-token embeddings when only the last words change)")
    results = {}
    for mode in ("causal", "bidirectional"):
        set_attention(adapter, mode)
        results[mode] = mask_test(adapter)
        print(f"  {mode:13}: {results[mode]:.6f}")
    # Causal should be ~0 (bf16 kernels on different lengths can leave tiny
    # noise); bidirectional must be clearly larger.
    ok = results["bidirectional"] > max(10 * results["causal"], 1e-2)
    print("  PASS: bidirectional attention is active" if ok else
          "  FAIL: expected ~0 for causal and a clear difference for bidirectional")

    print(f"\n== 2. Evidence recall, plain top-k, budget {args.token_budget}, {args.limit} questions")
    items = load_longmemeval(limit=args.limit, seed=args.seed)
    settings = CompressionSettings(token_budget=args.token_budget)
    tokenwise = TokenWise(model=adapter)
    rows = {"causal": [], "bidirectional": []}
    for index, item in enumerate(items, start=1):
        memory = seed_memory_from_haystack(item)
        memory.add_user_message(item["question"])
        units, cutoff = make_runner(None, memory, tokenwise, settings).compression_candidates()
        for mode in rows:
            set_attention(adapter, mode)
            result = tokenwise.compress_units(item["question"], settings.token_budget, units)
            last = {"cutoff_index": cutoff, "result": result, "candidates": len(units)}
            rows[mode].append({
                "question_type": item["question_type"],
                "evidence": exact_evidence_recall(memory, item, last, "compressed_history"),
                "compressed_tokens": result.tokens,
            })
        print(f"  {index}/{len(items)}", flush=True)
    for mode, mode_rows in rows.items():
        summarize_recall(f"CPC {args.cpc_preset} ({mode})", mode_rows)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
