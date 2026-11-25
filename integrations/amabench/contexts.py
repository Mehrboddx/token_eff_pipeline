"""What the reader sees for each question, at one token budget (cl100k).

    trunc    the reference long-context truncation (first 70% + last 30% of the
             trajectory's tokens), cut to the budget; the same for every question
    bm25     the reference BM25 (one document per step, whitespace tokens), steps
             taken by score while they fit, shown in step order
    actions  the agent's actions alone, latest first while they fit, in step order
    random   the compressor service with a random ranking (same units and layout as cpc)
    cpc      the compressor service: CPC + Provence fusion, query = the question

Excerpts from the service keep each step's own number ([step N, ...]); exact
repeats are kept, since the same line at two steps can answer different questions.
The full trajectory (condition `full`) is assembled by the reader step.

    python -m integrations.amabench.contexts --episodes logs_ama/episodes.jsonl \
        --out logs_ama/contexts.jsonl --budget 4000 --url http://127.0.0.1:8765
"""

import argparse
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from integrations.amabench.data import _encoding, count_tokens, step_text
from integrations.hscbench.trajectories import cap_observation

CONDITIONS = ("trunc", "bm25", "actions", "random", "cpc")
CUT_MARK = "\n[... middle of the trajectory cut ...]\n"


def step_block(step: dict) -> str:
    return f"Step {step.get('turn_idx', 0)}:\n{step_text(step)}\n"


def truncate(text: str, budget: int) -> str:
    """First 70% and last 30% of the budget, as the reference LongContextMethod does."""
    ids = _encoding().encode(text, disallowed_special=())
    if len(ids) <= budget:
        return text
    room = budget - count_tokens(CUT_MARK)
    head = int(room * 0.7)
    return _encoding().decode(ids[:head]) + CUT_MARK + _encoding().decode(ids[len(ids) - (room - head):])


def pack(blocks: list[str], order: list[int], budget: int) -> str:
    """Blocks taken in `order` while they fit, then shown in their original order."""
    kept, used = [], 0
    for index in order:
        cost = count_tokens(blocks[index]) + 1
        if used + cost <= budget:
            kept.append(index)
            used += cost
    return "\n".join(blocks[index] for index in sorted(kept))


def bm25_context(trajectory: list[dict], question: str, budget: int) -> str:
    from rank_bm25 import BM25Okapi

    blocks = [step_block(step) for step in trajectory]
    scores = BM25Okapi([block.lower().split() for block in blocks]).get_scores(question.lower().split())
    return pack(blocks, sorted(range(len(blocks)), key=lambda i: -scores[i]), budget)


def actions_context(trajectory: list[dict], budget: int) -> str:
    blocks = [f"Step {step.get('turn_idx', 0)}: Action: {step.get('action') or ''}" for step in trajectory]
    return pack(blocks, list(range(len(blocks) - 1, -1, -1)), budget)


def service_entries(trajectory: list[dict]) -> list[dict]:
    return [{"role": "tool", "label": "action and observation", "step": step.get("turn_idx", 0),
             "text": f"Action: {step.get('action') or ''}\nObservation: {cap_observation(step.get('observation') or '')}"}
            for step in trajectory]


def service_context(trajectory: list[dict], question: str, budget: int, url: str, extra: dict) -> dict:
    """One service call, refitted (at most 4 times) until the rendered excerpt, labels
    included, is within the budget."""
    entries = service_entries(trajectory)
    asked = budget
    for _ in range(4):
        payload = {"entries": entries, "query": question, "budget": asked, "dedupe": False, **extra}
        request = urllib.request.Request(url.rstrip("/") + "/compress", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1800) as reply:
            result = json.loads(reply.read())
        tokens = count_tokens(result["text"])
        if tokens <= budget:
            break
        asked -= tokens - budget
    return {"text": result["text"], "ms": result.get("ms"), "units_in": result.get("units_in"),
            "units_kept": result.get("units_kept")}


def build(episode: dict, qa: dict, condition: str, budget: int, url: str) -> dict:
    trajectory, question = episode["trajectory"], qa["question"]
    row = {"episode_id": episode["episode_id"], "question_uuid": qa["question_uuid"], "condition": condition,
           "budget": budget, "error": ""}
    try:
        if condition == "trunc":
            row["text"] = truncate("\n".join(step_block(step) for step in trajectory), budget)
        elif condition == "bm25":
            row["text"] = bm25_context(trajectory, question, budget)
        elif condition == "actions":
            row["text"] = actions_context(trajectory, budget)
        elif condition == "random":
            row.update(service_context(trajectory, question, budget, url, {"selector": "random", "seed": 0}))
        elif condition == "cpc":
            row.update(service_context(trajectory, question, budget, url, {}))
        else:
            raise ValueError(f"unknown condition {condition}")
    except Exception as caught:  # recorded, and retried on the next run
        row.update(text="", error=f"{type(caught).__name__}: {caught}"[:500])
    row["tokens"] = count_tokens(row["text"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default="logs_ama/episodes.jsonl")
    parser.add_argument("--out", default="logs_ama/contexts.jsonl")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--workers", type=int, default=2, help="the service scores one request at a time")
    args = parser.parse_args()

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(row["question_uuid"], row["condition"]) for row in map(json.loads, out.open(encoding="utf-8"))
                if not row["error"]}
    episodes = [json.loads(line) for line in Path(args.episodes).open(encoding="utf-8")]
    # longest trajectories last, so a slow one does not hold up the rest
    jobs = [(episode, qa, condition) for episode in sorted(episodes, key=lambda e: e["full_tokens"])
            for qa in episode["qa_pairs"] for condition in args.conditions.split(",")
            if (qa["question_uuid"], condition) not in done]
    lock = threading.Lock()

    def run(job):
        row = build(*job, args.budget, args.url)
        with lock, out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        for count, row in enumerate(pool.map(run, jobs), 1):
            if count % 50 == 0 or row["error"]:
                print(f"[{count}/{len(jobs)}] {row['condition']} tokens {row['tokens']} ms {row.get('ms')} "
                      f"{'ERROR ' + row['error'] if row['error'] else 'ok'}", flush=True)


if __name__ == "__main__":
    main()
