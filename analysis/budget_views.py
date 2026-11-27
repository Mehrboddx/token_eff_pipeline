"""Reading the same memory at two budgets (Gemini only, no GPU).

Failure anatomy on LongMemEval: with every evidence turn kept at both budgets, CPC-Mem's
1k excerpt is read better than its 4k one (88.7 vs 85.9, 441 questions) -- the extra 3k
tokens are near-miss competitors (same topic, other event, older value) -- while 4k keeps
evidence 1k drops. The two budgets are complementary (right at either: 90.8%). Three ways to
get both, all applied the same way to every question:

  tiered      one context: the 1k excerpt under "Most relevant to the question", then the 4k
              excerpt under "Wider context". Writes a compress-stage log, answered and judged
              with the usual stages (costs ~1k more prompt tokens than 4k).
  cascade     answer from 1k; if the answer itself says the memory was missing (the
              analysis.escalate FOUND/MISSING classifier on question + answer), use the 4k
              answer. Needs both budgets answered already; only the classifier calls are new.
  adjudicate  a third call that sees the 4k context and both drafts (1k and 4k answers) and
              writes the final answer. Writes an answer-stage log for --stage judge.

    python -m analysis.budget_views tiered --small <1k compress log> --large <4k compress log> --out <log>
    python -m analysis.budget_views cascade --small-answers <1k judged> --large-answers <4k judged>
    python -m analysis.budget_views adjudicate --large <4k compress log> --small-answers <1k answers> \
        --large-answers <4k answers> --out logs/gemini_runs/<name>.jsonl
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

PREFIX = "Relevant earlier context:\n"

ADJUDICATE = """You have now answered this question twice from your memory of this user: once from a short excerpt that keeps only the sentences most relevant to the question, and once from the longer excerpt shown above. The short excerpt is less cluttered but can miss things; the longer one is more complete but also holds unrelated, similar-looking or outdated details.

Draft from the short excerpt:
{small}

Draft from the longer excerpt:
{large}

Check both drafts against the earlier context above, then reply to the question as you would to the user, with the final answer only.

Question: {question}"""


def rows(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row.get("event") == "result":
            out[row["question_id"]] = row
    return out


def first_line(path):
    return json.loads(open(path, encoding="utf-8").readline())


def body(history):
    text = history[0]["text"]
    return text[len(PREFIX) :] if text.startswith(PREFIX) else text


def tiered(args):
    small, large = rows(args.small), rows(args.large)
    start = first_line(args.large)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    **start,
                    "timestamp": datetime.now().isoformat(),
                    "view": "tiered",
                    "small": str(args.small),
                    "large": str(args.large),
                }
            )
            + "\n"
        )
        for qid, row in large.items():
            history = [dict(turn) for turn in row["history"]]
            if (
                row.get("compression_mode") == "compressed_history"
                and qid in small
                and small[qid].get("compression_mode") == "compressed_history"
            ):
                history[0]["text"] = (
                    PREFIX
                    + "Most relevant to the question:\n"
                    + body(small[qid]["history"])
                    + "\n\nWider context (more of the same memory; it repeats the lines above and adds "
                    "less directly relevant ones):\n" + body(row["history"])
                )
            out.write(json.dumps({**row, "history": history}) + "\n")
    print(f"wrote {len(large)} questions to {args.out}")


def cascade(args):
    from google import genai

    from analysis.escalate import classify

    small, large = rows(args.small_answers), rows(args.large_answers)
    ids = [qid for qid in small if qid in large]
    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client = genai.Client(vertexai=True, project=args.project, location="us-central1")
    todo = [qid for qid in ids if qid not in cache]
    with ThreadPoolExecutor(8) as pool:
        for qid, found in zip(
            todo, pool.map(lambda q: classify(client, small[q]["question"], small[q]["predicted"]), todo)
        ):
            cache[qid] = found
    cache_path.write_text(json.dumps(cache))
    right = lambda r: bool(r.get("llm_judge_correct"))  # noqa: E731
    tok = lambda r: float(r.get("compressed_tokens") or 0)  # noqa: E731
    found = [q for q in ids if cache[q]]
    picked = {q: (small[q] if cache[q] else large[q]) for q in ids}
    acc = 100 * sum(right(picked[q]) for q in ids) / len(ids)
    cost = sum(tok(small[q]) + (0 if cache[q] else tok(large[q])) for q in ids) / len(ids)
    print(f"{len(ids)} questions; 1k answers FOUND {len(found)}, MISSING {len(ids) - len(found)}")
    print(
        f"  FOUND right {100 * sum(right(small[q]) for q in found) / max(len(found), 1):.1f}%, "
        f"MISSING right at 1k {100 * sum(right(small[q]) for q in ids if not cache[q]) / max(len(ids) - len(found), 1):.1f}% "
        f"vs at 4k {100 * sum(right(large[q]) for q in ids if not cache[q]) / max(len(ids) - len(found), 1):.1f}%"
    )
    for name, view in (("1k", small), ("4k", large)):
        print(
            f"  {name}: {100 * sum(right(view[q]) for q in ids) / len(ids):.1f}%  {sum(tok(view[q]) for q in ids) / len(ids):.0f} tokens"
        )
    print(f"  cascade 1k -> 4k: {acc:.1f}%  {cost:.0f} tokens (both budgets' tokens counted when it escalates)")
    better = sum(1 for q in ids if right(picked[q]) and not right(large[q]))
    worse = sum(1 for q in ids if right(large[q]) and not right(picked[q]))
    print(f"  vs 4k: {better} better / {worse} worse")
    # a random mix of the two fixed budgets with the cascade's escalation share
    share = (len(ids) - len(found)) / len(ids)
    rng = random.Random(0)
    mixes = [sum(right(large[q] if rng.random() < share else small[q]) for q in ids) / len(ids) for _ in range(2000)]
    print(f"  random mix with the same 4k share ({100 * share:.0f}%): {100 * sum(mixes) / len(mixes):.1f}%")


def adjudicate(args):
    from google.genai import types

    from eval.longmemeval import build_agent

    large, small_ans, large_ans = rows(args.large), rows(args.small_answers), rows(args.large_answers)
    ids = [qid for qid in large if qid in small_ans and qid in large_ans]
    done = rows(args.out) if Path(args.out).exists() else {}
    agent = build_agent(project=args.project, reader="memory", temperature=0.0)

    def run(qid):
        row = large[qid]
        history = [
            types.Content(role=t["role"], parts=[types.Part.from_text(text=t["text"])]) for t in row["history"][:-1]
        ]
        prompt = ADJUDICATE.format(
            small=small_ans[qid]["predicted"], large=large_ans[qid]["predicted"], question=row["history"][-1]["text"]
        )
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        try:
            response = agent.call_model(history)
            meta = getattr(response, "usage_metadata", None)
            return (
                qid,
                response.text or "",
                {
                    "prompt_tokens": getattr(meta, "prompt_token_count", None),
                    "thinking_tokens": getattr(meta, "thoughts_token_count", None),
                },
            )
        except Exception as exc:  # retried on the next run
            return qid, f"[ERROR: {exc}]", {}

    todo = [qid for qid in ids if qid not in done or str(done[qid].get("predicted", "")).startswith("[ERROR")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as out:
        if not done:
            out.write(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "run_start",
                        "stage": "answer",
                        "view": "adjudicate",
                        "from_log": str(args.large),
                        "small": str(args.small_answers),
                        "large": str(args.large_answers),
                        "reader": "memory",
                        "answer_temperature": 0.0,
                    }
                )
                + "\n"
            )
        with ThreadPoolExecutor(args.workers) as pool:
            for count, (qid, reply, usage) in enumerate(pool.map(run, todo), 1):
                base = large_ans[qid]
                record = {
                    key: base.get(key)
                    for key in (
                        "config",
                        "question_id",
                        "question_type",
                        "question",
                        "reference",
                        "compression_mode",
                        "compressed_tokens",
                        "evidence",
                    )
                }
                out.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event": "result",
                            **record,
                            "predicted": reply,
                            "correct": None,
                            "reader": "memory+adjudicate",
                            "usage": usage,
                        }
                    )
                    + "\n"
                )
                out.flush()
                if count % 50 == 0:
                    print(f"{count}/{len(todo)}", flush=True)
    print(f"done -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("tiered")
    p.add_argument("--small", required=True)
    p.add_argument("--large", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("cascade")
    p.add_argument("--small-answers", required=True)
    p.add_argument("--large-answers", required=True)
    p.add_argument("--cache", default="logs/analysis/sufficiency_fusion_1k_memory.json")
    p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    p = sub.add_parser("adjudicate")
    p.add_argument("--large", required=True)
    p.add_argument("--small-answers", required=True)
    p.add_argument("--large-answers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    args = parser.parse_args()
    {"tiered": tiered, "cascade": cascade, "adjudicate": adjudicate}[args.command](args)


if __name__ == "__main__":
    main()
