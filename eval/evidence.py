"""Evidence recall: did the turns LongMemEval labels as answer-bearing
(`has_answer`) survive compression? Measures the compressor on its own,
without an answering model or a grader in the loop.

Two variants:
  - exact: from the units a compressor actually selected (new runs);
  - text-matched: from the rendered compressed context alone, so it also
    works on logs written before selections were recorded, and on any
    compressor whose output preserves source wording."""

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(text):
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def _answer_session_indices(item):
    answer_ids = set(item.get("answer_session_ids") or [])
    return {index for index, sid in enumerate(item["haystack_session_ids"]) if sid in answer_ids}


def _counts(evidence_kept, evidence_total, sessions_kept, sessions_total):
    return {
        "evidence_turns": evidence_total,
        "evidence_turns_kept": evidence_kept,
        "answer_sessions": sessions_total,
        "answer_sessions_kept": sessions_kept,
    }


def exact_evidence_recall(memory, item, last_compression, mode):
    """Retained = turns with at least one selected sentence, plus every
    turn after the cutoff (the recent window is passed through raw)."""
    evidence = [index for index, meta in enumerate(memory.entry_meta) if meta.get("has_answer")]
    answer_sessions = _answer_session_indices(item)

    if mode != "compressed_history":
        return _counts(len(evidence), len(evidence), len(answer_sessions), len(answer_sessions))
    if last_compression is None or last_compression["result"].selected is None:
        return None

    cutoff = last_compression["cutoff_index"]
    retained = {unit.entry_index for unit in last_compression["result"].selected}
    retained.update(range(cutoff, len(memory.entry_meta)))
    retained_sessions = {memory.entry_meta[index].get("session") for index in retained}
    return _counts(
        sum(1 for index in evidence if index in retained),
        len(evidence),
        sum(1 for session in answer_sessions if session in retained_sessions),
        len(answer_sessions),
    )


def haystack_entries(item):
    """(session_index, role, content, has_answer) per turn, in the exact
    order and with the same empty-turn skipping as seed_memory_from_haystack,
    so entry positions line up with Memory's history indices."""
    entries = []
    for session_index, session in enumerate(item["haystack_sessions"]):
        for turn in session:
            content = (turn.get("content") or "").strip()
            if content and turn.get("role") in ("user", "assistant"):
                entries.append((session_index, turn["role"], content, bool(turn.get("has_answer"))))
    return entries


def recent_cutoff(entries, recent_turns):
    """Entry index where the raw recent window starts, counting the
    question itself as the final user turn (as Runner does); None when
    there are too few turns for compression to apply at all."""
    boundaries = [index for index, entry in enumerate(entries) if entry[1] == "user"] + [len(entries)]
    if len(boundaries) <= recent_turns:
        return None
    return boundaries[-recent_turns]


def _shingles(words, size):
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def text_matched_evidence_recall(item, compressed_context, mode, recent_turns, shingle_size=6):
    """A turn counts as kept if any `shingle_size`-word run of it appears in
    the compressed context (whole-text match for shorter turns), or if it
    falls inside the raw recent window."""
    entries = haystack_entries(item)
    evidence = [index for index, entry in enumerate(entries) if entry[3]]
    answer_sessions = _answer_session_indices(item)
    cutoff = recent_cutoff(entries, recent_turns)

    if mode != "compressed_history" or cutoff is None:
        return _counts(len(evidence), len(evidence), len(answer_sessions), len(answer_sessions))

    context = _normalize(compressed_context or "")
    context_shingles = _shingles(context.split(), shingle_size)

    def kept(index):
        if index >= cutoff:
            return True
        words = _normalize(entries[index][2]).split()
        if len(words) < shingle_size:
            return bool(words) and f" {' '.join(words)} " in f" {context} "
        return not _shingles(words, shingle_size).isdisjoint(context_shingles)

    kept_turns = {index for index in range(len(entries)) if (entries[index][3] or entries[index][0] in answer_sessions) and kept(index)}
    kept_sessions = {entries[index][0] for index in kept_turns}
    return _counts(
        sum(1 for index in evidence if index in kept_turns),
        len(evidence),
        sum(1 for session in answer_sessions if session in kept_sessions),
        len(answer_sessions),
    )


def summarize_recall(name, rows, verbose=True):
    """Macro-averaged over questions: turn recall (questions with labelled
    evidence turns only), all-evidence-kept rate, answer-session recall,
    and mean compressed-context tokens."""
    measured = [row for row in rows if row.get("evidence")]
    if not measured:
        return None

    def stats(subset):
        with_turns = [row["evidence"] for row in subset if row["evidence"]["evidence_turns"]]
        with_sessions = [row["evidence"] for row in subset if row["evidence"]["answer_sessions"]]
        tokens = [row["compressed_tokens"] for row in subset if row.get("compressed_tokens") is not None]
        return {
            "n": len(subset),
            "turn_recall": (
                sum(e["evidence_turns_kept"] / e["evidence_turns"] for e in with_turns) / len(with_turns)
                if with_turns else None
            ),
            "all_evidence_kept": (
                sum(e["evidence_turns_kept"] == e["evidence_turns"] for e in with_turns) / len(with_turns)
                if with_turns else None
            ),
            "session_recall": (
                sum(e["answer_sessions_kept"] / e["answer_sessions"] for e in with_sessions) / len(with_sessions)
                if with_sessions else None
            ),
            "mean_tokens": sum(tokens) / len(tokens) if tokens else None,
        }

    def fmt(value, pct=True):
        if value is None:
            return "  n/a"
        return f"{value:6.1%}" if pct else f"{value:6.0f}"

    overall = stats(measured)
    by_type = {}
    for row in measured:
        by_type.setdefault(row["question_type"], []).append(row)
    summary = {"overall": overall, "by_type": {t: stats(subset) for t, subset in sorted(by_type.items())}}
    if not verbose:
        return summary

    print(f"\n{name} -- evidence recall (n={overall['n']})")
    print(f"  {'':28} {'turn':>6} {'all-ev':>6} {'sess':>6} {'tokens':>6}")
    for label, s in [("overall", overall), *summary["by_type"].items()]:
        print(f"  {label:28} {fmt(s['turn_recall'])} {fmt(s['all_evidence_kept'])} "
              f"{fmt(s['session_recall'])} {fmt(s['mean_tokens'], pct=False)}")
    return summary
