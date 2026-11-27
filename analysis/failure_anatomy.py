"""Where CPC-Mem (CPC-Mistral + Provence fusion) loses LongMemEval questions.

No model calls: reads a compress run (the context the reader saw), its judged
answers, and LongMemEval's labels (which turns hold the answer), and sorts every
question the compressed reader got wrong into one cause:

  dropped turn     an answer-bearing turn has no sentence in the context
  thinned turn     every answer-bearing turn kept something, but the sentences
                   that carry the answer's words did not survive
  answer present   the evidence is in the context and the reader still erred
                   (counting, dates, updates, refusing on partial context)

and pairs it against the full-history reader on the same questions: losses that
compression caused (full right, compressed wrong) versus wins it gave (the
reverse), by question type, and by cause.

    python -m analysis.failure_anatomy \
        --compress logs/eval_longmemeval_run0053_compress_cpc-mistral_b4000_fuse0.25.jsonl \
        --judged logs/judge_runs/eval_longmemeval_run0053_compress_cpc-mistral_b4000_fuse0.25_t0.jsonl \
        --full logs/judge_runs/eval_longmemeval_run0039_full_fullhistory.jsonl \
        --dump logs/analysis/anatomy_0053.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval.evidence import haystack_entries
from eval.longmemeval import load_longmemeval

_WORD = re.compile(r"[a-z0-9]+")
STOP = set(
    "a an the of to in on at for and or but is are was were be been i you my your me it its this that with "
    "as by from did do does have has had what which who whom when where how many much did not no yes".split()
)


def words(text):
    return _WORD.findall(str(text).lower())


def content_words(text):
    return [w for w in words(text) if w not in STOP]


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(words(p)) >= 3]


def norm(text):
    return " ".join(words(text))


def load_rows(path, key="question_id"):
    rows = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row.get("event") == "result":
            rows[row[key]] = row
    return rows


def context_text(row):
    return "\n".join(turn.get("text") or "" for turn in row.get("history") or [])


def sentence_coverage(turn_text, context_norm):
    """Share of the turn's sentences found verbatim (normalised) in the context."""
    sents = sentences(turn_text)
    if not sents:
        return 1.0, []
    kept = [s for s in sents if norm(s) and norm(s) in context_norm]
    return len(kept) / len(sents), [s for s in sents if s not in kept]


def answer_terms(reference):
    """Content words of the reference answer (numbers kept); empty for long rubric answers."""
    terms = content_words(reference)
    return terms if 0 < len(terms) <= 6 else []


def classify(item, row, context):
    context_norm = norm(context)
    entries = haystack_entries(item)
    evidence = [e for e in entries if e[3]]
    if not evidence:  # abstention questions carry no answer turn
        return "no evidence label", {}
    coverage = [sentence_coverage(e[2], context_norm) for e in evidence]
    dropped = sum(1 for share, _ in coverage if share == 0)
    terms = answer_terms(item["answer"])
    evidence_words = set(content_words(" ".join(e[2] for e in evidence)))
    located = [t for t in terms if t in evidence_words]  # answer words the evidence actually spells out
    context_words = set(words(context))
    missing_terms = [t for t in located if t not in context_words]
    detail = {
        "evidence_turns": len(evidence),
        "dropped_turns": dropped,
        "sentence_coverage": round(sum(s for s, _ in coverage) / len(coverage), 3),
        "answer_terms": terms,
        "missing_answer_terms": missing_terms,
        "missing_sentences": [s for _, miss in coverage for s in miss][:6],
    }
    if dropped:
        return "dropped turn", detail
    if missing_terms:
        return "thinned turn", detail
    return "answer present", detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--dump", default=None)
    args = parser.parse_args()

    items = {item["question_id"]: item for item in load_longmemeval()}
    compress = load_rows(args.compress)
    judged = load_rows(args.judged)
    full = load_rows(args.full)

    ids = [qid for qid in judged if qid in compress]
    causes = Counter()
    by_type = defaultdict(Counter)
    paired = Counter()
    paired_cause = defaultdict(Counter)
    coverage_right, coverage_wrong = [], []
    dump = []
    for qid in ids:
        item, row, verdict = items[qid], compress[qid], judged[qid]
        ok = bool(verdict.get("llm_judge_correct"))
        full_ok = bool(full[qid]["llm_judge_correct"]) if qid in full else None
        qtype = item["question_type"] + ("_abs" if qid.endswith("_abs") else "")
        cause, detail = classify(item, row, context_text(row))
        if detail:
            (coverage_right if ok else coverage_wrong).append(detail["sentence_coverage"])
        by_type[qtype]["n"] += 1
        by_type[qtype]["right"] += ok
        if full_ok is not None:
            paired[(full_ok, ok)] += 1
        if not ok:
            causes[cause] += 1
            by_type[qtype][cause] += 1
            if full_ok is not None:
                paired_cause["full right" if full_ok else "full wrong"][cause] += 1
        dump.append({"question_id": qid, "type": qtype, "question": item["question"], "reference": item["answer"],
                     "predicted": verdict.get("predicted"), "correct": ok, "full_correct": full_ok,
                     "cause": cause if not ok else None, "tokens": row.get("compressed_tokens"), **detail})

    n = len(ids)
    right = sum(1 for d in dump if d["correct"])
    print(f"{n} questions, compressed right {right} ({100 * right / n:.1f}%)")
    print("\nwrong answers by cause:")
    for cause, count in causes.most_common():
        print(f"  {cause:<18} {count:>3}")
    print("\npaired with full history (full, compressed):")
    for key in [(True, True), (True, False), (False, True), (False, False)]:
        print(f"  full {'right' if key[0] else 'wrong'} / compressed {'right' if key[1] else 'wrong'}: {paired[key]}")
    for side, counter in paired_cause.items():
        print(f"  compressed wrong, {side}: " + ", ".join(f"{c} {k}" for c, k in counter.most_common()))
    print("\nby type (n, accuracy, causes of the wrong ones):")
    for qtype, counter in sorted(by_type.items()):
        rest = ", ".join(f"{k} {v}" for k, v in counter.items() if k not in ("n", "right"))
        print(f"  {qtype:<30} n={counter['n']:<4} {100 * counter['right'] / counter['n']:5.1f}%  {rest}")
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    print(f"\nevidence sentence coverage: right answers {mean(coverage_right):.3f}, wrong {mean(coverage_wrong):.3f}")
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as handle:
            for record in dump:
                handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
