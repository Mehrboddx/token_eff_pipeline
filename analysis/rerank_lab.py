"""Offline selection lab: try ranking functions over the dumped units and
cached scores (analysis.score_dump), with the same greedy budget packing as
core.selection.select_units (sentence tokens + each new session header + each
new turn label), and measure what each keeps. numpy only, no model.

Metrics per ranker and budget:
  recall      share of scored evidence turns with at least one kept sentence
  all kept    share of questions whose every scored evidence turn is kept
  ev share    share of kept sentence tokens that come from evidence turns
              (precision: how much of the budget the reader must wade through)

    python -m analysis.rerank_lab --dump logs/analysis/scores_longmemeval_s.jsonl.gz --budgets 500 1000 2000 4000
"""

import argparse
import gzip
import json
from collections import defaultdict

import numpy as np


def z(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    return (values - values.mean()) / std if std > 1e-12 else np.zeros_like(values)


def ranks(values):
    order = np.lexsort((np.arange(len(values)), -np.asarray(values)))  # best first, ties by position
    out = np.empty(len(values), dtype=float)
    out[order] = np.arange(len(values))
    return out


def select(row, scores, budget):
    """Greedy by score; a unit that does not fit is skipped and the next one tried."""
    units = row["units"]
    header, label = row["header_tokens"], row["label_tokens"]
    order = np.lexsort((-np.arange(len(scores)), -np.asarray(scores)))  # ties: later unit first, as select_units
    used, sessions, entries, kept = 0, set(), set(), []
    for index in order:
        if used >= budget:
            break
        entry, session, role, tokens = units[index]
        cost = (
            tokens
            + (header[str(session)] if session not in sessions else 0)
            + (label[role] if entry not in entries else 0)
        )
        if used + cost > budget:
            continue
        used += cost
        sessions.add(session)
        entries.add(entry)
        kept.append(int(index))
    return kept


# ---- rankers: row -> one score per unit ------------------------------------------------------


def fused(row, weight=0.25):
    per_query = [(1 - weight) * z(c) + weight * z(p) for c, p in zip(row["cpc"], row["provence"])]
    return np.max(per_query, axis=0)


def cpc(row):
    return np.max([z(c) for c in row["cpc"]], axis=0)


def provence(row):
    return np.max([z(p) for p in row["provence"]], axis=0)


def rrf(row, k=60, weight=0.25):
    """Reciprocal rank fusion (Cormack et al. 2009), weighted like the z fusion."""
    c, p = cpc(row), provence(row)
    return (1 - weight) / (k + ranks(c)) + weight / (k + ranks(p))


def group_max(row, scores, key):
    """For every unit, the best score among the other units of its group (turn or session)."""
    groups = defaultdict(list)
    for index, unit in enumerate(row["units"]):
        groups[key(unit)].append(index)
    best = np.full(len(scores), -np.inf)
    for members in groups.values():
        values = scores[members]
        if len(members) == 1:
            continue
        top2 = np.sort(values)[-2:]
        for index, value in zip(members, values):
            best[index] = top2[0] if value == top2[1] else top2[1]
    return np.where(np.isfinite(best), best, scores)


def turn_smooth(row, alpha=0.25):
    """Score + alpha * the best other sentence of the same turn: a sentence in a turn that
    holds a strong match is more likely to belong to the same fact (asides, second clauses)."""
    s = fused(row)
    return s + alpha * np.maximum(group_max(row, s, key=lambda u: u[0]), s.min())


def session_prior(row, beta=0.25, top=3):
    """Score + beta * z(session relevance), session relevance = mean of its `top` best scores:
    a hierarchical prior (relevant sessions hold the other instances of a fact)."""
    s = fused(row)
    sessions = defaultdict(list)
    for index, unit in enumerate(row["units"]):
        sessions[unit[1]].append(s[index])
    rel = {key: float(np.mean(sorted(values)[-top:])) for key, values in sessions.items()}
    zs = dict(zip(rel, z(list(rel.values()))))
    return s + beta * np.array([zs[unit[1]] for unit in row["units"]])


def with_upr(row, gamma=0.5):
    """Fused score, plus gamma * z(UPR question likelihood) on the units UPR re-scored
    (analysis.upr_scores: the top few hundred under the fused score)."""
    s = fused(row).copy()
    upr = row.get("upr") or {}
    if upr:
        ids = np.array([int(i) for i in upr])
        s[ids] += gamma * z([upr[str(i)] for i in ids])
    return s


RANKERS = {
    "cpc": cpc,
    "provence": provence,
    "fused": fused,
    "fused W0.15": lambda r: fused(r, 0.15),
    "fused W0.35": lambda r: fused(r, 0.35),
    "rrf": rrf,
    "rrf W0.5": lambda r: rrf(r, weight=0.5),
    "turn smooth 0.1": lambda r: turn_smooth(r, 0.1),
    "turn smooth 0.25": turn_smooth,
    "turn smooth 0.5": lambda r: turn_smooth(r, 0.5),
    "session prior 0.25": session_prior,
    "session prior 0.5": lambda r: session_prior(r, 0.5),
    "session prior 1.0": lambda r: session_prior(r, 1.0),
    "upr 0.25": lambda r: with_upr(r, 0.25),
    "upr 0.5": with_upr,
    "upr 1.0": lambda r: with_upr(r, 1.0),
}


def evaluate(rows, rankers, budgets):
    table = {}
    for name in rankers:
        for budget in budgets:
            kept_turns = total = all_kept = n = 0
            ev_tokens = sel_tokens = 0
            by_type = defaultdict(lambda: [0, 0])
            for row in rows:
                evidence = set(row["evidence"])
                if not evidence:
                    continue
                kept = select(row, rankers[name](row), budget)
                turns = {row["units"][i][0] for i in kept}
                hit = len(evidence & turns)
                kept_turns += hit
                total += len(evidence)
                all_kept += hit == len(evidence)
                n += 1
                by_type[row["question_type"]][0] += hit
                by_type[row["question_type"]][1] += len(evidence)
                sel_tokens += sum(row["units"][i][3] for i in kept)
                ev_tokens += sum(row["units"][i][3] for i in kept if row["units"][i][0] in evidence)
            table[(name, budget)] = {
                "recall": 100 * kept_turns / total,
                "all_kept": 100 * all_kept / n,
                "ev_share": 100 * ev_tokens / max(sel_tokens, 1),
                "by_type": {t: 100 * a / b for t, (a, b) in by_type.items()},
            }
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    parser.add_argument("--rankers", nargs="*", default=None)
    parser.add_argument("--by-type", action="store_true")
    parser.add_argument("--upr", default=None, help="analysis.upr_scores output; only questions it covers are used")
    args = parser.parse_args()
    rows = [json.loads(line) for line in gzip.open(args.dump, "rt", encoding="utf-8")]
    if args.upr:
        upr = {r["question_id"]: r["upr"] for r in map(json.loads, gzip.open(args.upr, "rt", encoding="utf-8"))}
        rows = [dict(row, upr=upr[row["question_id"]]) for row in rows if row["question_id"] in upr]
    rankers = {
        k: v
        for k, v in RANKERS.items()
        if (not args.rankers or k in args.rankers) and (args.upr or not k.startswith("upr"))
    }
    table = evaluate(rows, rankers, args.budgets)
    print(f"{len(rows)} questions; recall / all evidence kept / evidence share of kept tokens (%)")
    print(f"{'ranker':<20}" + "".join(f"{b:>22}" for b in args.budgets))
    for name in rankers:
        cells = "".join(
            f"{table[(name, b)]['recall']:>8.1f}{table[(name, b)]['all_kept']:>7.1f}{table[(name, b)]['ev_share']:>7.1f}"
            for b in args.budgets
        )
        print(f"{name:<20}{cells}")
    if args.by_type:
        types = sorted(next(iter(table.values()))["by_type"])
        for budget in args.budgets:
            print(f"\nrecall by type at {budget}:  " + "  ".join(t[:14] for t in types))
            for name in rankers:
                print(f"  {name:<20}" + "".join(f"{table[(name, budget)]['by_type'][t]:>16.1f}" for t in types))


if __name__ == "__main__":
    main()
