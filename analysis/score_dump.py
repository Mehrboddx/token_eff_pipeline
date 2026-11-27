"""Dump every question's candidate units and both scorers' cached scores, so
selection ideas can be tried offline (numpy only, no model, no GPU).

Same capture as analysis.fusion_analysis: the units and budget of a
--stage compress run are rebuilt exactly, and CPC's and Provence's scores
come from the score cache (a cache miss raises; no model is ever loaded).

One JSON line per question (gzip):
  question_id, question_type, budget,
  units: [[entry_index, session, role, tokens], ...]   chronological, as scored
  cpc, provence: [[score per unit], ...] one list per query
  header_tokens: {session: tokens of its "[Earlier session, ...]" header}
  label_tokens: {role: tokens of its "User:" / "Assistant:" label}
  evidence: entry indices of the has_answer turns among the scored units
  evidence_total: all has_answer turns (some sit in the raw recent window)
  text: {unit id: text} for the 400 best units under the fused score

    python -m analysis.score_dump --dataset longmemeval_s --budget 4000 \
        --score-cache logs/score_cache --out logs/analysis/scores_longmemeval_s.jsonl.gz
"""

import argparse
import gzip
import json
from pathlib import Path

from core.console import utf8_console
from core.selection import SelectionConfig, fuse_scores
from core.units import role_label, session_header
from eval.longmemeval import CompressionSettings, build_tokenwise, compress_question, load_items, seed_memory_from_haystack

utf8_console()


def refuse_to_load():
    raise RuntimeError("score cache miss: this script never loads a model")


def dump_question(item, tokenwise, budget, weight, text_top):
    captured = {}
    original = tokenwise.compress_units

    def capture(query, token_budget, units, extra_queries=()):
        units = list(units)
        queries = [query, *dict.fromkeys(q for q in extra_queries if q and q != query)]
        captured.update(units=units, queries=queries,
                        primary=tokenwise.model.score_units(units, queries),
                        secondary=tokenwise.secondary.score_units(units, queries))
        return original(query, token_budget, units, extra_queries)

    tokenwise.compress_units = capture
    try:
        memory = seed_memory_from_haystack(item)
        _, mode, _, _ = compress_question(item, tokenwise, CompressionSettings(token_budget=budget), seeded_memory=memory)
    finally:
        tokenwise.compress_units = original
    if mode != "compressed_history" or not captured.get("units"):
        return None
    units = captured["units"]
    evidence = {i for i, meta in enumerate(memory.entry_meta) if meta.get("has_answer")}
    fused = fuse_scores(captured["primary"], SelectionConfig(provence_weight=weight), secondary=captured["secondary"])
    top = sorted(range(len(units)), key=lambda i: fused[i], reverse=True)[:text_top]
    count = tokenwise.count_tokens
    return {
        "question_id": item["question_id"],
        "question_type": item.get("question_type"),
        "question": item.get("question"),
        "budget": budget,
        "units": [[u.entry_index, u.session, u.role, u.tokens] for u in units],
        "cpc": [[round(s, 5) for s in q] for q in captured["primary"]],
        "provence": [[round(s, 5) for s in q] for q in captured["secondary"]],
        "header_tokens": {str(u.session): count(session_header(u)) for u in units},
        "label_tokens": {u.role: count(role_label(u)) for u in units},
        "evidence": sorted(evidence & {u.entry_index for u in units}),
        "evidence_total": len(evidence),
        "text": {str(i): units[i].text for i in top},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--score-cache", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text-top", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenwise = build_tokenwise(compressor="cpc", cpc_preset="mistral",
                                selection=SelectionConfig(provence_weight=args.weight), score_cache=args.score_cache)
    assert tokenwise is not None
    tokenwise.model._factory = refuse_to_load
    tokenwise.secondary._factory = refuse_to_load
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for index, item in enumerate(load_items(args.dataset, limit=args.limit), 1):
            row = dump_question(item, tokenwise, args.budget, args.weight, args.text_top)
            if row is not None:
                out.write(json.dumps(row) + "\n")
                written += 1
            if index % 50 == 0:
                print(f"{index} questions, {written} written", flush=True)
    print(f"done: {written} questions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
