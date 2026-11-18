"""Per-question budgets chosen before answering, simulated on runs already
judged. No LLM calls and no CPC model: scores come from the score cache,
and a cache miss raises instead of loading the model.

Each question has a ladder of judged answers, cheapest first (e.g. 2k,
4k, full history). A gate picks one rung per question from information
available before answering; accuracy and tokens are then read from the
judged run at that rung. Two gates:

- rule: the sentences that stand out in CPC's z-normalized scores
  (z >= tau) are what the question needs; pick the smallest budget that
  holds their tokens, else the full history.
- learned: a logistic regression per rung predicts whether that rung
  answers at least as well as the reference rung, from CPC's score
  profile and cues in the question's wording. The cheapest rung whose
  predicted probability clears theta is picked.

tau and theta are tuned on training questions only -- the cheapest
setting that keeps the reference rung's accuracy there -- and every
reported number is on held-out questions (2 folds for the rule, 5 for
the learned gate; LoCoMo folds keep each conversation in one fold).

Two stages. On the GPU machine, compute each question's features (CPC
scores, from the score cache where present, else from the model) into
logs/gate_features_<dataset>.json:
    python -m analysis.score_gate --dataset longmemeval_s --features-only --score-cache logs/score_cache
    python -m analysis.score_gate --dataset locomo --features-only --score-cache logs/score_cache
Then, anywhere with the judge logs and that file (no model is loaded):
    python -m analysis.score_gate --dataset longmemeval_s
    python -m analysis.score_gate --dataset locomo
"""

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.adaptive_budget import LADDERS, full_history_tokens, results
from core.console import utf8_console
from core.selection import SelectionConfig, zscore
from eval.longmemeval import CompressionSettings, build_tokenwise, compress_question, load_items

utf8_console()

TAUS = [x / 4 for x in range(4, 21)]  # 1.0 .. 5.0
THETAS = [x / 20 for x in range(1, 20)]  # 0.05 .. 0.95
Z_LEVELS = (1.5, 2.0, 2.5, 3.0, 4.0)

# Cues in the question's wording; all are known before answering.
CUES = {
    "temporal": r"\b(when|date|day|days|week|weeks|month|months|year|ago|before|after|first|last|recently|how long)\b",
    "count": r"\b(how many|how much|total|number of|all the|list)\b",
    "assistant": r"\b(you (said|told|suggested|recommended|mentioned|gave|provided|listed)|your (suggestion|recommendation|answer))\b",
    "preference": r"\b(recommend|suggest|should i|any (tips|ideas|advice))\b",
}


def refuse_to_load():
    raise RuntimeError("score cache miss: this script never loads the CPC model")


def features_for(dataset, ids, score_cache, budget, allow_model=False):
    """question_id -> features, computed once and cached in logs/. ids=None
    means every question of the dataset. Without allow_model, a question
    whose scores aren't cached raises instead of loading CPC."""
    path = Path(f"logs/gate_features_{dataset}.json")
    cached = json.loads(path.read_text()) if path.exists() else {}
    missing = None if ids is None else set(ids) - set(cached)
    if missing is not None and not missing:
        return cached

    # Checked before building anything: without a score cache, building the
    # CPC compressor loads the model at once.
    if not allow_model and score_cache is None:
        raise SystemExit(
            f"{path} is missing questions; compute features on the GPU machine with --features-only first."
        )
    tokenwise = build_tokenwise(
        compressor="cpc", cpc_preset="mistral", selection=SelectionConfig(), score_cache=score_cache
    )
    assert tokenwise is not None  # always built with a compressor here
    if not allow_model:
        # The score cache builds CPC lazily, on the first miss: make a miss fail instead.
        tokenwise.model._factory = refuse_to_load
    captured = {}
    original = tokenwise.model.score_units

    def capture(units, queries):
        scores = original(units, queries)
        captured["z"] = zscore(scores[0])
        captured["tokens"] = [tokenwise.count_tokens(unit.text) for unit in units]
        return scores

    tokenwise.model.score_units = capture
    items = load_items(dataset, limit=None)
    for index, item in enumerate(items, 1):
        if item["question_id"] in cached or (missing is not None and item["question_id"] not in missing):
            continue
        if index % 50 == 0:
            print(f"features {index}/{len(items)}", flush=True)
            path.write_text(json.dumps(cached))  # checkpoint, so a stopped run resumes
        captured.clear()
        compress_question(item, tokenwise, CompressionSettings(token_budget=budget))
        if not captured:  # short history, not compressed
            cached[item["question_id"]] = None
            continue
        pairs = sorted(zip(captured["z"], captured["tokens"]), reverse=True)
        z = [value for value, _ in pairs]

        def top(k, z=z):
            return z[min(k, len(z)) - 1]

        cached[item["question_id"]] = {
            "need": {str(tau): sum(t for value, t in pairs if value >= tau) for tau in TAUS},
            "units": len(z),
            "history_tokens": sum(t for _, t in pairs),
            "z1": z[0],
            "z5": top(5),
            "z10": top(10),
            "z20": top(20),
            "z50": top(50),
            **{f"count_z{level}": sum(value >= level for value in z) for level in Z_LEVELS},
        }
    path.write_text(json.dumps(cached))
    return cached


def vector(features, question):
    """Feature vector for the learned gate. Log-scaled counts and sizes."""
    q = question.lower()
    if features is None:
        return [0.0] * (11 + len(Z_LEVELS) + len(CUES))
    return [
        np.log1p(features["units"]),
        np.log1p(features["history_tokens"]),
        features["z1"],
        features["z5"],
        features["z10"],
        features["z20"],
        features["z50"],
        features["z1"] - features["z10"],
        np.log1p(features["need"]["2.0"]),
        np.log1p(features["need"]["3.0"]),
        *[np.log1p(features[f"count_z{level}"]) for level in Z_LEVELS],
        len(q.split()),
        *[float(bool(re.search(pattern, q))) for pattern in CUES.values()],
    ]


def rule_pick(features, tau, budgets):
    need = features["need"][str(tau)] if features else 0
    return next((i for i, budget in enumerate(budgets) if need <= budget), len(budgets))


def cascade(probabilities, theta, reference):
    """Cheapest rung below the reference predicted good enough, else the reference."""
    return next((i for i, p in enumerate(probabilities) if p >= theta), reference)


def evaluate(ids, picks, correct, tokens):
    return (statistics.mean(correct[q][picks[q]] for q in ids), statistics.mean(tokens[q][picks[q]] for q in ids))


def mix_frontier(ids, correct, tokens, names):
    """Fixed budgets on the upper concave hull of (mean tokens, accuracy):
    the best accuracy any random mix of two fixed budgets reaches at each
    token level. A budget below the hull (e.g. one beaten by a cheaper one)
    is never worth mixing in."""
    points = sorted(
        (statistics.mean(tokens[q][i] for q in ids), statistics.mean(correct[q][i] for q in ids), i)
        for i in range(len(names))
    )
    hull = []
    for point in points:
        if hull and point[1] <= hull[-1][1]:
            continue  # more tokens, no better: never on the frontier
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            if (y2 - y1) * (point[0] - x1) <= (point[1] - y1) * (x2 - x1):  # hull[-1] on or below the chord
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def versus_mix(ids, picks, correct, tokens, names, resamples=10_000, seed=0):
    """The gate against the best random mix of two fixed budgets with the
    same mean tokens: (mix accuracy, gate - mix, bootstrap p). Per question
    the mix scores (1-w)*correct[i] + w*correct[j]; the difference is
    resampled by question, with w held fixed."""
    _, mean = evaluate(ids, picks, correct, tokens)
    hull = mix_frontier(ids, correct, tokens, names)
    fixed = {i: x for x, _, i in hull}
    order = [i for _, _, i in hull]
    if mean <= fixed[order[0]]:
        return None  # cheaper than every fixed budget: no mix to compare with
    if mean >= fixed[order[-1]]:
        i = j = order[-1]  # beyond the frontier: the best fixed budget is cheaper and at least as good
        w = 0.0
    else:
        k = next(k for k in range(len(order) - 1) if fixed[order[k]] <= mean <= fixed[order[k + 1]])
        i, j = order[k], order[k + 1]
        w = (mean - fixed[i]) / (fixed[j] - fixed[i])
    diff = np.array([correct[q][picks[q]] - ((1 - w) * correct[q][i] + w * correct[q][j]) for q in ids])
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), (resamples, len(diff)))].mean(axis=1)
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    accuracy, _ = evaluate(ids, picks, correct, tokens)
    return accuracy - diff.mean(), diff.mean(), p


def report(label, ids, picks, correct, tokens, names, reference):
    accuracy, mean = evaluate(ids, picks, correct, tokens)
    better = sum(correct[q][picks[q]] and not correct[q][reference] for q in ids)
    worse = sum(correct[q][reference] and not correct[q][picks[q]] for q in ids)
    share = Counter(names[picks[q]] for q in ids)
    mix = versus_mix(ids, picks, correct, tokens, names)
    mix_text = f"; random mix {100 * mix[0]:.1f} ({100 * mix[1]:+.1f}, p {mix[2]:.3f})" if mix else ""
    print(
        f"  {label:<12} accuracy {100 * accuracy:5.1f}, mean tokens {mean:>7,.0f}, "
        f"vs fixed {names[reference]} {better} better / {worse} worse{mix_text}; picks "
        + ", ".join(f"{n} {100 * share[n] / len(ids):.0f}%" for n in names)
    )


def fit_rungs(X, labels, rows, reference):
    """One classifier per rung below the reference: P(rung is enough)."""
    # Imported here: the GPU image has no scikit-learn, and --features-only doesn't need it.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = []
    for rung in range(reference):
        y = labels[rows, rung]
        if y.min() == y.max():  # one class only: predict it
            models.append(float(y[0]))
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        models.append(model.fit(X[rows], y))
    return models


def predict(models, X, rows):
    return (
        np.column_stack(
            [np.full(len(rows), m) if isinstance(m, float) else m.predict_proba(X[rows])[:, 1] for m in models]
        )
        if models
        else np.zeros((len(rows), 0))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s", choices=list(LADDERS))
    parser.add_argument("--score-cache", default=None)
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="GPU machine: compute features for every question (loads CPC on cache misses) and stop.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Rung whose accuracy the gates must keep (default: 4k on LongMemEval, full on LoCoMo).",
    )
    args = parser.parse_args()

    if args.features_only:
        features = features_for(args.dataset, None, args.score_cache, budget=4000, allow_model=True)
        print(f"{len(features)} questions -> logs/gate_features_{args.dataset}.json")
        return

    ladder = LADDERS[args.dataset]["mistral"]
    names = [name for name, _ in ladder]
    budgets = [int(float(name[:-1]) * 1000) for name in names if name != "full"]
    reference = names.index(args.reference or ("full" if args.dataset == "locomo" else "4k"))
    judged = [results(f"logs/judge_runs/eval_longmemeval_{tag}.jsonl") for _, tag in ladder]
    compressed = [results(f"logs/eval_longmemeval_{tag}.jsonl") for _, tag in ladder]
    ids = sorted(set.intersection(*(set(j) for j in judged)))

    full = full_history_tokens(args.dataset, set(ids))
    tokens = {
        q: [compressed[i][q]["compressed_tokens"] if names[i] != "full" else full[q] for i in range(len(names))]
        for q in ids
    }
    correct = {q: [bool(j[q].get("llm_judge_correct")) for j in judged] for q in ids}
    features = features_for(args.dataset, ids, args.score_cache, budgets[-1])
    print(f"{args.dataset}: {len(ids)} questions, reference rung {names[reference]}\n")

    print("Fixed budgets")
    for i, name in enumerate(names):
        report(f"fixed {name}", ids, {q: i for q in ids}, correct, tokens, names, reference)

    # Rule gate, 2 folds by question id.
    fold = {q: int(hashlib.sha1(q.encode()).hexdigest(), 16) % 2 for q in ids}
    rule_picks, taus = {}, []
    for held_out in (0, 1):
        tune = [q for q in ids if fold[q] != held_out]
        target, _ = evaluate(tune, {q: reference for q in tune}, correct, tokens)
        options = []
        for tau in TAUS:
            picks = {q: rule_pick(features[q], tau, budgets) for q in tune}
            accuracy, mean = evaluate(tune, picks, correct, tokens)
            if accuracy >= target:
                options.append((mean, tau))
        # No tau keeps the reference's accuracy: fall back to the reference itself.
        tau = min(options)[1] if options else None
        taus.append(tau)
        rule_picks.update(
            {q: reference if tau is None else rule_pick(features[q], tau, budgets) for q in ids if fold[q] == held_out}
        )
    print(f"\nRule gate (held-out; tau per fold {taus}, None = falls back to {names[reference]})")
    report("rule", ids, rule_picks, correct, tokens, names, reference)
    print("\nRule gate trade-off (no fitting beyond tau itself)")
    for tau in TAUS[::2]:
        report(
            f"tau {tau}", ids, {q: rule_pick(features[q], tau, budgets) for q in ids}, correct, tokens, names, reference
        )

    # Learned gate, 5 folds; LoCoMo folds by conversation, LongMemEval by question.
    question_text = {q: judged[0][q]["question"] for q in ids}
    X = np.array([vector(features[q], question_text[q]) for q in ids], dtype=float)
    labels = np.array([[correct[q][i] >= correct[q][reference] for i in range(len(names))] for q in ids], dtype=int)
    groups = [q.split("_")[0] for q in ids]
    from sklearn.model_selection import GroupKFold

    learned_picks, thetas = {}, []
    out_of_fold = np.zeros((len(ids), reference))  # each question predicted by a model that never saw it
    for train, test in GroupKFold(n_splits=5).split(X, groups=groups):
        # theta from out-of-fold predictions inside the training part only.
        inner = np.zeros((len(train), reference))
        for inner_train, inner_test in GroupKFold(n_splits=5).split(X[train], groups=[groups[i] for i in train]):
            models = fit_rungs(X, labels, train[inner_train], reference)
            inner[inner_test] = predict(models, X, train[inner_test])
        train_ids = [ids[i] for i in train]
        target, _ = evaluate(train_ids, {q: reference for q in train_ids}, correct, tokens)
        options = []
        for theta in THETAS:
            picks = {q: cascade(inner[k], theta, reference) for k, q in enumerate(train_ids)}
            accuracy, mean = evaluate(train_ids, picks, correct, tokens)
            if accuracy >= target:
                options.append((mean, theta))
        theta = min(options)[1] if options else 1.0
        thetas.append(theta)
        models = fit_rungs(X, labels, train, reference)
        probabilities = predict(models, X, test)
        out_of_fold[test] = probabilities
        learned_picks.update({ids[i]: cascade(probabilities[k], theta, reference) for k, i in enumerate(test)})
    print(f"\nLearned gate (held-out; theta per fold {thetas})")
    report("learned", ids, learned_picks, correct, tokens, names, reference)
    print("\nLearned gate trade-off (out-of-fold predictions, theta swept)")
    for theta in THETAS[::2]:
        report(
            f"theta {theta:.2f}",
            ids,
            {q: cascade(out_of_fold[k], theta, reference) for k, q in enumerate(ids)},
            correct,
            tokens,
            names,
            reference,
        )

    ideal = {q: next((i for i in range(len(names)) if correct[q][i] >= correct[q][reference]), reference) for q in ids}
    print("\nIdeal gate (knows the verdicts; upper bound)")
    report("ideal", ids, ideal, correct, tokens, names, reference)

    # Which features the learned gate relies on (fitted on all questions, cheapest rung).
    if reference > 0:
        model = fit_rungs(X, labels, np.arange(len(ids)), reference)[0]
        if not isinstance(model, float):
            names_f = (
                [
                    "log units",
                    "log history tokens",
                    "z1",
                    "z5",
                    "z10",
                    "z20",
                    "z50",
                    "z1-z10",
                    "log need z>=2",
                    "log need z>=3",
                ]
                + [f"log count z>={level}" for level in Z_LEVELS]
                + ["question words"]
                + [f"cue: {c}" for c in CUES]
            )
            weights = model[-1].coef_[0]
            print(f"\nLargest standardized weights for 'P({names[0]} is enough)':")
            for weight, name in sorted(zip(weights, names_f), key=lambda pair: -abs(pair[0]))[:8]:
                print(f"  {weight:+.2f}  {name}")


if __name__ == "__main__":
    main()
