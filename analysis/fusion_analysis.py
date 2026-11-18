"""Why does CPC + Provence fusion keep more evidence than either scorer?
No model is loaded: both scorers' sentence scores come from the score
cache, and a cache miss raises.

For each question, the units and budget of the fusion run are rebuilt
exactly (same code path as --stage compress), then selection is re-run
under three orderings of the same units at the same budget:
  cpc       CPC's z-scores alone (fusion weight 0)
  provence  Provence's z-scores alone
  fused     (1-W) z_CPC + W z_Provence, as in the fusion runs
and we record which evidence turns (has_answer) each one keeps.

Reports:
  - recall per ordering, and the 2x2 table of evidence turns by whether
    CPC alone and Provence alone keep them, with what fusion keeps in
    each cell (are the two scorers' misses different?);
  - rank agreement: Spearman correlation of the two scorers per question,
    and the best percentile rank of each evidence turn under each scorer;
  - precision: share of selected tokens that come from evidence turns.

    python -m analysis.fusion_analysis --dataset locomo --budgets 4000 2000 --score-cache logs/score_cache
    python -m analysis.fusion_analysis --dataset longmemeval_s --budgets 4000 1000 --score-cache logs/score_cache
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from core.console import utf8_console
from core.selection import SelectionConfig, fuse_scores, select_units
from eval.longmemeval import (
    CompressionSettings,
    build_tokenwise,
    compress_question,
    load_items,
    seed_memory_from_haystack,
)

utf8_console()

ORDERINGS = ("cpc", "provence", "fused")


def refuse_to_load():
    raise RuntimeError("score cache miss: this script never loads a model")


def ranks(values):
    """Rank 0 = best; ties broken by position, as selection does."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i), reverse=True)
    out = [0] * len(values)
    for rank, index in enumerate(order):
        out[index] = rank
    return out


def spearman(a, b):
    ra, rb = np.array(ranks(a), dtype=float), np.array(ranks(b), dtype=float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse_question(item, tokenwise, budget, weight):
    captured = {}
    original = tokenwise.compress_units

    def capture(query, token_budget, units, extra_queries=()):
        units = list(units)
        queries = [query, *dict.fromkeys(q for q in extra_queries if q and q != query)]
        captured.update(
            units=units,
            budget=token_budget,
            primary=tokenwise.model.score_units(units, queries),
            secondary=tokenwise.secondary.score_units(units, queries),
        )
        return original(query, token_budget, units, extra_queries)

    tokenwise.compress_units = capture
    try:
        memory = seed_memory_from_haystack(item)
        _, mode, _, info = compress_question(
            item, tokenwise, CompressionSettings(token_budget=budget), seeded_memory=memory
        )
    finally:
        tokenwise.compress_units = original
    if mode != "compressed_history" or not captured.get("units"):
        return None

    units = captured["units"]
    evidence = {i for i, meta in enumerate(memory.entry_meta) if meta.get("has_answer")}
    candidate_turns = {unit.entry_index for unit in units}
    scored_evidence = sorted(evidence & candidate_turns)  # turns in the recent window are always kept

    base = SelectionConfig()
    scores = {
        "cpc": fuse_scores(captured["primary"], base),
        "provence": fuse_scores(captured["secondary"], base),
        "fused": fuse_scores(
            captured["primary"], SelectionConfig(provence_weight=weight), secondary=captured["secondary"]
        ),
    }
    row = {
        "question_id": item["question_id"],
        "question_type": item.get("question_type"),
        "units": len(units),
        "evidence_turns": len(evidence),
        "scored_evidence": len(scored_evidence),
        "turns": scored_evidence,
        "run_evidence_kept": (info.get("evidence") or {}).get("evidence_turns_kept"),
        "spearman": spearman(scores["cpc"], scores["provence"]),
    }
    tokens = [tokenwise.count_tokens(unit.text) for unit in units]
    for name in ORDERINGS:
        selected = set(select_units(units, scores[name], captured["budget"], tokenwise.count_tokens, base))
        kept_turns = {units[i].entry_index for i in selected}
        rank = ranks(scores[name])
        best = {}
        for i, unit in enumerate(units):
            if unit.entry_index in evidence:
                best[unit.entry_index] = min(best.get(unit.entry_index, len(units)), rank[i])
        row[name] = {
            "kept": [turn for turn in scored_evidence if turn in kept_turns],
            "best_pct": [round(100 * best[turn] / len(units), 2) for turn in scored_evidence],
            "selected_tokens": sum(tokens[i] for i in selected),
            "evidence_tokens": sum(tokens[i] for i in selected if units[i].entry_index in evidence),
        }
    return row


def summarize(rows, budget):
    # Sanity check: re-running the fused selection must reproduce what the
    # fusion run kept (scored evidence kept + evidence in the recent window).
    checked = [row for row in rows if row and row["run_evidence_kept"] is not None]
    match = sum(
        len(row["fused"]["kept"]) + row["evidence_turns"] - row["scored_evidence"] == row["run_evidence_kept"]
        for row in checked
    )
    print(f"\n[check] fused selection reproduces the run's evidence count on {match}/{len(checked)} questions")
    rows = [row for row in rows if row and row["scored_evidence"]]
    total = sum(row["scored_evidence"] for row in rows)
    print(f"\n=== budget {budget}: {len(rows)} questions with evidence among the scored units, {total} evidence turns")
    for name in ORDERINGS:
        kept = sum(len(row[name]["kept"]) for row in rows)
        sel = sum(row[name]["selected_tokens"] for row in rows)
        ev = sum(row[name]["evidence_tokens"] for row in rows)
        pct = [p for row in rows for p in row[name]["best_pct"]]
        print(
            f"  {name:9s} recall {100 * kept / total:5.1f}   evidence share of kept tokens {100 * ev / sel:5.2f}%"
            f"   median best rank pct {statistics.median(pct):5.2f}"
        )

    cells = {}
    for row in rows:
        cpc, prov, fused = (set(row[name]["kept"]) for name in ORDERINGS)
        for turn in row["turns"]:
            key = ("CPC" if turn in cpc else "-") + "/" + ("Prov" if turn in prov else "-")
            n, f = cells.get(key, (0, 0))
            cells[key] = (n + 1, f + (turn in fused))
    print("  evidence turns by who keeps them alone (CPC/Provence) -> share fusion keeps:")
    for key in ("CPC/Prov", "CPC/-", "-/Prov", "-/-"):
        n, f = cells.get(key, (0, 0))
        print(f"    {key:9s} {n:5d} ({100 * n / total:4.1f}%)   fusion keeps {f:5d} ({100 * f / max(n, 1):5.1f}%)")

    rho = [row["spearman"] for row in rows if row["spearman"] is not None]
    print(
        f"  Spearman(CPC, Provence) per question: median {statistics.median(rho):.2f}, "
        f"IQR {np.percentile(rho, 25):.2f}-{np.percentile(rho, 75):.2f}"
    )
    turns = [(c, p) for row in rows for c, p in zip(row["cpc"]["best_pct"], row["provence"]["best_pct"])]
    if len(turns) > 2:
        corr = np.corrcoef(np.array(turns).T)[0, 1]
        print(f"  evidence turns' best-rank percentiles, CPC vs Provence: Pearson {corr:.2f}")

    types = sorted({row["question_type"] for row in rows if row["question_type"]})
    if types:
        print("  recall by question type (cpc / provence / fused):")
        for qtype in types:
            sub = [row for row in rows if row["question_type"] == qtype]
            n = sum(row["scored_evidence"] for row in sub)
            vals = [100 * sum(len(row[name]["kept"]) for row in sub) / n for name in ORDERINGS]
            print(f"    {qtype:28s} n={len(sub):4d}  " + " / ".join(f"{v:5.1f}" for v in vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s")
    parser.add_argument("--budgets", type=int, nargs="+", default=[4000])
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--score-cache", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenwise = build_tokenwise(
        compressor="cpc",
        cpc_preset="mistral",
        selection=SelectionConfig(provence_weight=args.weight),
        score_cache=args.score_cache,
    )
    assert tokenwise is not None  # always built with a compressor here
    tokenwise.model._factory = refuse_to_load
    tokenwise.secondary._factory = refuse_to_load
    items = load_items(args.dataset, limit=args.limit)
    for budget in args.budgets:
        path = Path(f"logs/fusion_analysis_{args.dataset}_b{budget}.jsonl")
        rows = []
        with path.open("w", encoding="utf-8") as out:
            for index, item in enumerate(items, 1):
                if index % 100 == 0:
                    print(f"budget {budget}: {index}/{len(items)}", flush=True)
                row = analyse_question(item, tokenwise, budget, args.weight)
                if row is not None:
                    out.write(json.dumps(row) + "\n")
                    rows.append(row)
        summarize(rows, budget)
        print(f"  rows written to {path}")


if __name__ == "__main__":
    main()
