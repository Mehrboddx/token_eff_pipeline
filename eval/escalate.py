"""Self-escalating memory, simulated offline: answer from a small budget,
and only if the answer signals that the needed memory was missing, answer
again from the next larger budget (up to full history).

The signal comes from the answer alone: a Gemini classifier (temperature 0)
sees the question and the answer -- never the reference or the context --
and says FOUND or MISSING. Verdicts are cached in
logs/sufficiency_{dataset}.json, so re-running is free.

Cost is what an agent would pay: the tokens of every budget it tried, not
only the last one. Each policy is compared with the best random mix of two
fixed budgets at the same total tokens (the frontier from eval.score_gate).

Usage (this PC; Gemini only, no GPU):
    python -m eval.escalate --dataset longmemeval_s
    python -m eval.escalate --dataset locomo
"""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eval.adaptive_budget import LADDERS, full_history_tokens, results
from eval.score_gate import mix_frontier

SUFFICIENCY_PROMPT = """You check an assistant's answer to a question about a user's past conversations with it. You see only the question and the answer, not the conversations.

Reply MISSING if the answer says or implies that the information needed was not available, incomplete or not found: for example it says something was not mentioned, asks the user for details it should already know, gives only general advice where the question asks about the user's own situation, or hedges between possibilities.
Reply FOUND if the answer commits to a specific answer taken from the conversations, even if it might be wrong.

Question: {question}
Answer: {answer}

Reply with one word: FOUND or MISSING."""


def classify(client, question, answer):
    from google.genai import types
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part.from_text(
            text=SUFFICIENCY_PROMPT.format(question=question, answer=answer[:4000]))])],
        config=types.GenerateContentConfig(temperature=0),
    )
    return "MISSING" not in (response.text or "").upper()


def sufficiency(dataset, ladder, judged, ids, project, location, workers=8):
    """{rung name: {question_id: found?}} for every compressed rung, cached."""
    path = Path(f"logs/sufficiency_{dataset}.json")
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [(name, q) for (name, _), rows in zip(ladder, judged) if name != "full"
            for q in ids if q not in cache.get(name, {})]
    if todo:
        from google import genai
        client = genai.Client(vertexai=True, project=project, location=location)
        rows_by_rung = {name: rows for (name, _), rows in zip(ladder, judged)}

        def work(task):
            name, q = task
            row = rows_by_rung[name][q]
            try:
                return name, q, classify(client, row["question"], row.get("predicted") or "")
            except Exception as exc:  # retried on the next run
                print(f"  {name} {q}: {exc}")
                return name, q, None

        print(f"classifying {len(todo)} answers")
        with ThreadPoolExecutor(workers) as pool:
            for done, (name, q, found) in enumerate(pool.map(work, todo), 1):
                if found is not None:
                    cache.setdefault(name, {})[q] = found
                if done % 200 == 0:
                    path.write_text(json.dumps(cache))
                    print(f"  {done}/{len(todo)}", flush=True)
        path.write_text(json.dumps(cache))
    return cache


def escalate(ids, start, names, found, correct, tokens, cap=None):
    """Try rungs from `start` upwards, stop at the first FOUND answer (or at
    `cap`, default the last rung). Returns per-question (correct, total tokens, final rung)."""
    last = len(names) - 1 if cap is None else cap
    out = {}
    for q in ids:
        spent, rung = 0, start
        while True:
            spent += tokens[q][rung]
            if rung == last or found[names[rung]].get(q, True):
                break
            rung += 1
        out[q] = (correct[q][rung], spent, rung)
    return out


def versus_frontier(ids, outcome, correct, tokens, names, resamples=10_000, seed=0):
    """Escalation vs the best random mix of two fixed budgets at the same mean total tokens."""
    mean = statistics.mean(spent for _, spent, _ in outcome.values())
    hull = mix_frontier(ids, correct, tokens, names)
    xs = [x for x, _, _ in hull]
    if mean <= xs[0]:
        return None
    if mean >= xs[-1]:
        i = j = hull[-1][2]
        w = 0.0
    else:
        k = next(k for k in range(len(hull) - 1) if xs[k] <= mean <= xs[k + 1])
        i, j = hull[k][2], hull[k + 1][2]
        w = (mean - xs[k]) / (xs[k + 1] - xs[k])
    diff = np.array([outcome[q][0] - ((1 - w) * correct[q][i] + w * correct[q][j]) for q in ids])
    boot = diff[np.random.default_rng(seed).integers(0, len(diff), (resamples, len(diff)))].mean(axis=1)
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return statistics.mean(o[0] for o in outcome.values()) - diff.mean(), diff.mean(), p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="longmemeval_s", choices=list(LADDERS))
    parser.add_argument("--project", default="cs-poc-pjtbxc0qllmcraowi6w1wmi")
    parser.add_argument("--location", default="us-central1")
    args = parser.parse_args()

    ladder = LADDERS[args.dataset]["mistral"]
    names = [name for name, _ in ladder]
    judged = [results(f"logs/judge_runs/eval_longmemeval_{tag}.jsonl") for _, tag in ladder]
    compressed = [results(f"logs/eval_longmemeval_{tag}.jsonl") for _, tag in ladder]
    ids = sorted(set.intersection(*(set(j) for j in judged)))
    full = full_history_tokens(args.dataset, set(ids))
    tokens = {q: [compressed[i][q]["compressed_tokens"] if names[i] != "full" else full[q]
                  for i in range(len(names))] for q in ids}
    correct = {q: [bool(j[q].get("llm_judge_correct")) for j in judged] for q in ids}
    found = sufficiency(args.dataset, ladder, judged, ids, args.project, args.location)

    print(f"\n{args.dataset}: {len(ids)} questions\n")
    print("How often each budget's answer is classified FOUND, and its accuracy when FOUND / MISSING")
    for i, name in enumerate(names[:-1]):
        f = [q for q in ids if found[name].get(q, True)]
        m = [q for q in ids if not found[name].get(q, True)]
        acc = lambda qs: 100 * statistics.mean(correct[q][i] for q in qs) if qs else float("nan")
        print(f"  {name:<5} FOUND {100 * len(f) / len(ids):5.1f}%  acc {acc(f):5.1f} | MISSING acc {acc(m):5.1f}")

    def show(label, outcome):
        accuracy = 100 * statistics.mean(o[0] for o in outcome.values())
        mean = statistics.mean(o[1] for o in outcome.values())
        final = {name: 100 * sum(o[2] == i for o in outcome.values()) / len(ids) for i, name in enumerate(names)}
        mix = versus_frontier(ids, outcome, correct, tokens, names)
        mix_text = f"  best mix {100 * mix[0]:.1f} ({100 * mix[1]:+.1f}, p {mix[2]:.3f})" if mix else ""
        print(f"  {label:<22} accuracy {accuracy:5.1f}, mean tokens {mean:>7,.0f}{mix_text}; ends at "
              + ", ".join(f"{n} {v:.0f}%" for n, v in final.items() if v))

    print("\nFixed budgets")
    for i, name in enumerate(names):
        show(f"fixed {name}", {q: (correct[q][i], tokens[q][i], i) for q in ids})
    print("\nEscalate on MISSING (tokens = every budget tried)")
    for start in range(len(names) - 1):
        show(f"from {names[start]} to full", escalate(ids, start, names, found, correct, tokens))
        if len(names) - 2 > start:
            show(f"from {names[start]} to {names[-2]}",
                 escalate(ids, start, names, found, correct, tokens, cap=len(names) - 2))


if __name__ == "__main__":
    main()
