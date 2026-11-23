"""The receiving model takes over each task under each handoff condition and
predicts the relevant files and line spans (HSC-Bench repository track).

Model settings follow the paper (Appendix A): temperature 0, top-p 1, an
output limit of 1,024 tokens, minimal reasoning for Gemini.

    python -m integrations.hscbench.receive --tasks logs_hsc/tasks.jsonl \
        --contexts logs_hsc/contexts.jsonl --out logs_hsc/predictions.jsonl
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from integrations.hscbench.llm import complete

SYSTEM = (
    "You are a software engineer taking over an unfinished task from another agent. You cannot run "
    "commands. From the issue and the handoff left by the previous agent, identify the code locations "
    "in the repository that are relevant to resolving the issue."
)

PROMPT = """## Issue
{issue}

## Handoff from the previous agent
{context}

List the relevant code locations as JSON only, in this form:
{{"locations": [{{"file": "path/relative/to/the/repository/root", "spans": [[start_line, end_line], ...]}}]}}
At most 12 files and 12 spans per file; line numbers are 1-based and inclusive."""


def parse(text: str) -> list[dict] | None:
    """Predicted locations, or None when the answer cannot be parsed (scored as empty)."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    locations = data.get("locations") if isinstance(data, dict) else None
    if not isinstance(locations, list):
        return None
    out = []
    for location in locations[:12]:
        if not isinstance(location, dict) or not isinstance(location.get("file"), str):
            continue
        spans = []
        for span in (location.get("spans") or [])[:12]:
            try:
                start, end = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError):
                continue
            if start >= 1 and end >= start:
                spans.append([start, end])
        out.append({"file": location["file"], "spans": spans})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="logs_hsc/tasks.jsonl")
    parser.add_argument("--contexts", default="logs_hsc/contexts.jsonl")
    parser.add_argument("--out", default="logs_hsc/predictions.jsonl")
    parser.add_argument("--receiver", default="vertex_ai/gemini-3.5-flash-lite")
    parser.add_argument("--reasoning", default="minimal")
    parser.add_argument("--conditions", default="recent,state,summary,cpc,cpc_2x,full")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    issues = {task["task_id"]: task["issue"] for task in map(json.loads, Path(args.tasks).open(encoding="utf-8"))}
    contexts = [json.loads(line) for line in Path(args.contexts).open(encoding="utf-8")]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(row["task_id"], row["condition"], row["receiver"]) for row in map(json.loads, out.open(encoding="utf-8"))}
    jobs = [(row, condition) for row in contexts for condition in args.conditions.split(",")
            if (row["task_id"], condition, args.receiver) not in done]
    lock = threading.Lock()

    def run(job):
        row, condition = job
        prompt = PROMPT.format(issue=issues[row["task_id"]], context=row["contexts"][condition])
        result = complete(args.receiver, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                          max_tokens=1024, reasoning_effort=args.reasoning or None)
        record = {"task_id": row["task_id"], "condition": condition, "receiver": args.receiver,
                  "locations": parse(result["text"]), "raw": result["text"][:4000],
                  **{key: result[key] for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "error")}}
        with lock, out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    with ThreadPoolExecutor(args.workers) as pool:
        for count, record in enumerate(pool.map(run, jobs), 1):
            if count % 25 == 0 or record["error"]:
                print(f"[{count}/{len(jobs)}] {record['task_id']} {record['condition']} "
                      f"{'ERROR ' + record['error'] if record['error'] else 'ok'}", flush=True)


if __name__ == "__main__":
    main()
