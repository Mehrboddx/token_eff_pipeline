"""Re-render a paper-faithful CPC run in CPC-Mem's dated layout.

Takes a --paper-cpc compress log and rewrites every compressed context so
the sentences the paper's CPC kept are shown exactly as CPC-Mem shows its
own: in time order, one "[Earlier session, ...]" header per session, one
speaker label per turn, " … " where a turn's sentences were skipped. The
selection is not touched -- no sentence is added or dropped, nothing is
re-encoded -- so answering the new log against the original isolates
layout from selection.

The paper's CPC returns its kept sentences joined in transcript order, so
they are recovered by re-splitting the same transcript with the same
segmenter and matching it, in order, against the joined text. Each kept
sentence then gets the session, date and speaker of the transcript line it
came from. "[Session: ...]" date lines that were kept are dropped (the
header now carries the date), and so are the "Speaker:" prefixes the
paper's turn lines start with (the turn label now carries it).

The output is an ordinary compress-stage log, so it is answered and judged
like any other run. It needs no GPU and no Vertex AI credentials.

Usage:
    python -m eval.rerender_paper --from-log logs/eval_longmemeval_run0031_compress_cpc-mistral_b4000_paper.jsonl --run 47
    python -m eval.rerender_paper --from-log logs/eval_longmemeval_run0032_compress_locomo_cpc-mistral_b4000_paper.jsonl --run 48

then, as for any compress log:
    python -m eval.longmemeval --stage answer --from-log logs/<new log> --project <project>
    python -m eval.longmemeval --stage judge --from-log logs/gemini_runs/<new log> --project <project>
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pysbd
import tiktoken

from core.units import Unit, render_units
from eval.longmemeval import (
    ROLE_LABELS,
    dataset_from_label,
    load_items,
    load_run_log,
    paper_transcript,
    seed_memory_from_haystack,
)

PREFIX = "Relevant earlier context:\n"
_SESSION_LINE = re.compile(r"\[Session: [^\]]*\]")


def transcript_lines(memory, turn_dates=False):
    """paper_transcript's lines, each with the turn it belongs to: (text,
    char_start, info), where info is None for a "[Session: ...]" line and
    otherwise holds the turn's entry index, role, session, date, speaker
    and the length of its "Speaker: " prefix. Built the same way, and
    checked against paper_transcript, so offsets line up exactly."""
    lines = []
    current_session = object()
    current_entry = None
    for entry in memory.get_sentence_entries():
        meta = entry["meta"] or {}
        if entry["entry_index"] == current_entry:
            lines[-1][0] += " " + entry["text"]
            continue
        current_entry = entry["entry_index"]
        if meta.get("session") != current_session:
            current_session = meta.get("session")
            if meta.get("time") and not turn_dates:
                lines.append([f"[Session: {meta['time']}]", None])
        label = meta.get("speaker") or ROLE_LABELS.get(entry["role"], entry["role"])
        date = f"[{meta['time']}] " if turn_dates and meta.get("time") else ""
        prefix = f"{date}{label}: "
        lines.append([prefix + entry["text"], {
            "entry_index": entry["entry_index"],
            "role": entry["role"],
            "session": meta.get("session"),
            "time": meta.get("time"),
            "meta": meta,
            "prefix_len": len(prefix),
        }])

    text = "\n".join(line for line, _ in lines)
    assert text == paper_transcript(memory, turn_dates), "transcript rebuild differs from paper_transcript"
    out, start = [], 0
    for line, info in lines:
        out.append((line, start, info))
        start += len(line) + 1
    return text, out


def split_sentences(segmenter, text):
    """The paper CPC's own split: pysbd char spans, whitespace-stripped."""
    out = []
    for span in segmenter.segment(text):
        stripped = span.sent.strip()
        if not stripped:
            continue
        start = span.start + (len(span.sent) - len(span.sent.lstrip()))
        out.append((stripped, start, start + len(stripped)))
    return out


def match_kept(sentences, joined):
    """Indices of the sentences whose in-order " ".join is `joined`.
    Greedy, backtracking when it gets stuck; None if nothing fits.
    Exact duplicates can make the match ambiguous -- the earliest
    occurrence wins, the same text either way."""
    kept, pos, index = [], 0, 0
    stack = []
    while pos < len(joined):
        found = False
        while index < len(sentences):
            sent = sentences[index][0]
            end = pos + len(sent)
            if joined.startswith(sent, pos) and (end == len(joined) or joined[end] == " "):
                stack.append((index, pos))
                kept.append(index)
                pos, index = end + 1, index + 1
                found = True
                break
            index += 1
        if found:
            continue
        if not stack:
            return None
        index, pos = stack.pop()
        kept.pop()
        index += 1
    return kept


def kept_units(sentences, kept, lines):
    """Units for the kept sentences, one per transcript line each covers.
    Session date lines and "Speaker: " prefixes are left out; ids are
    positions in the transcript, so render_units' " … " marks real gaps."""
    pieces = []  # (sentence index, line index, text) over the whole transcript
    line_index = 0
    for sent_index, (_, start, end) in enumerate(sentences):
        while lines[line_index][1] + len(lines[line_index][0]) < start:
            line_index += 1
        li = line_index
        while li < len(lines) and lines[li][1] < end:
            line, line_start, info = lines[li]
            lo, hi = max(start, line_start), min(end, line_start + len(line))
            if info is not None:
                lo = max(lo, line_start + info["prefix_len"])
            piece = line[lo - line_start:hi - line_start].strip()
            if piece and info is not None:
                pieces.append((sent_index, li, piece))
            li += 1

    keep = set(kept)
    units = []
    for position, (sent_index, li, piece) in enumerate(pieces):
        if sent_index not in keep:
            continue
        info = lines[li][2]
        units.append(Unit(id=position, text=piece, role=info["role"], entry_index=info["entry_index"],
                          session=info["session"], time=info["time"], meta=info["meta"]))
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from-log", required=True, help="a --paper-cpc compress-stage log")
    parser.add_argument("--run", type=int, required=True, help="run number for the new log")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--limit", type=int, help="only the first N questions (with --dry-run, for a check)")
    parser.add_argument("--dry-run", action="store_true", help="write nothing; print one re-rendered context")
    args = parser.parse_args()

    run_start, rows = load_run_log(Path(args.from_log))
    if not run_start.get("paper_cpc"):
        raise SystemExit(f"{args.from_log} is not a --paper-cpc run")
    turn_dates = run_start.get("paper_dates") == "turn"
    rows = list({(row["config"], row["question_id"]): row for row in rows}.values())
    if args.limit is not None:
        if not args.dry_run:
            raise SystemExit("--limit is for checks only; use it with --dry-run")
        rows = rows[:args.limit]

    dataset = dataset_from_label(run_start.get("dataset"))
    limit = run_start.get("limit")
    items = load_items(dataset, limit=int(limit) if limit not in (None, "None") else None,
                       seed=int(run_start.get("seed", 42)), question_types=run_start.get("question_types"))
    items = {item["question_id"]: item for item in items}

    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    encoding = tiktoken.encoding_for_model("gpt-4")
    count_tokens = lambda text: len(encoding.encode(text, disallowed_special=()))

    source = Path(args.from_log).name
    match = re.match(r"eval_longmemeval_run\d+_(.*)\.jsonl$", source)
    log_path = Path(args.log_dir) / f"eval_longmemeval_run{args.run:04d}_{match.group(1)}-rerender.jsonl"
    if log_path.exists():
        raise SystemExit(f"{log_path} already exists")

    stats = {"questions": 0, "unmatched": 0, "date_lines_dropped": 0, "tokens_before": 0, "tokens_after": 0}
    out_rows = []
    for row in rows:
        context = row["history"][0]
        assert context.get("kind") == "compressed_context" and context["text"].startswith(PREFIX)
        joined = context["text"][len(PREFIX):]

        memory = seed_memory_from_haystack(items[row["question_id"]])
        transcript, lines = transcript_lines(memory, turn_dates)
        sentences = split_sentences(segmenter, transcript)
        kept = match_kept(sentences, joined)
        stats["questions"] += 1
        if kept is None:
            # Never silently fall back to the original layout: the row
            # would then measure nothing. Stop and report instead.
            stats["unmatched"] += 1
            print(f"  {row['question_id']}: kept sentences not found in the transcript")
            continue

        stats["date_lines_dropped"] += sum(1 for i in kept if _SESSION_LINE.fullmatch(sentences[i][0]))
        text = render_units(kept_units(sentences, kept, lines))
        before, after = count_tokens(joined), count_tokens(text)
        stats["tokens_before"] += before
        stats["tokens_after"] += after

        history = [dict(context, text=PREFIX + text), *row["history"][1:]]
        out_rows.append(dict(row, history=history, compressed_tokens=after))

    n = stats["questions"]
    if stats["unmatched"]:
        raise SystemExit(f"{stats['unmatched']} of {n} questions could not be matched; no log written")

    if args.dry_run:
        print(out_rows[0]["history"][0]["text"][:3000])
    else:
        with log_path.open("w", encoding="utf-8") as log_file:
            def log(record):
                record = {key: value for key, value in record.items() if key != "timestamp"}
                log_file.write(json.dumps({"timestamp": datetime.now().isoformat(), **record}) + "\n")

            log(dict(run_start, run=args.run, rerender_of=run_start.get("run"),
                     rerender_from_log=str(args.from_log), paper_render="dated"))
            for row in out_rows:
                log(row)

    print(f"{n} questions re-rendered" + ("" if args.dry_run else f" -> {log_path}"))
    print(f"  mean tokens: {stats['tokens_before'] / n:.0f} as joined, {stats['tokens_after'] / n:.0f} re-rendered")
    print(f"  kept session date lines dropped: {stats['date_lines_dropped']}")


if __name__ == "__main__":
    main()
