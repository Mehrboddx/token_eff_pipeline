"""Handoff contexts for every task, one per condition (HSC-Bench Section 4.1,
Appendix A, "Context preparation"), plus CPC-Mem:

  state      structured state: a JSON record of relevant facts, commitments,
             prior actions and remaining work, written by an LLM from the task
             and the history; valid JSON, at most 4,000 characters, about 1,800
             targeted, at least five times shorter than its source.
  recent     the end of the history, verbatim, cut to the state's token count.
  summary    prose of the same facts, capped at the state's length in
             characters (bounded to 200-1,600), so it may come out shorter.
  cpc        CPC + Provence (W = 0.25): the history lines most relevant to the
             issue, verbatim, under the state's token count. No LLM call.
  cpc_2x     the same at twice that budget (how the excerpt scales).
  full       the whole pre-handoff history.
  cpc_ctx    CPC-Mem on the context only: every step's tool call verbatim,
             and CPC's selection from the tool outputs in the rest of the
             state's token count, each line under the step it came from.
  cpc_ctx_2x the same at twice that budget. (Added to built rows by --extend.)

Token counts use o200k_base, as in the paper. The paper's writer is GPT-4.1;
here it is --writer.

    python -m integrations.hscbench.handoff --tasks logs_hsc/tasks.jsonl --out logs_hsc/contexts.jsonl
"""

import argparse
import json
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from integrations.hscbench.llm import complete, count_tokens
from integrations.hscbench.trajectories import history_text, pair_text

STATE_MAX_CHARS = 4000
STATE_TARGET_CHARS = 1800
SUMMARY_CHARS = (200, 1600)

WRITER_SYSTEM = (
    "You prepare handoffs between software agents. Another agent will take over an unfinished task "
    "from the transcript below and must continue from what is recorded there. Use only what appears "
    "in the transcript; do not invent file contents, paths or line numbers."
)

STATE_PROMPT = """## Task
{issue}

## Transcript of the work so far
{history}

Write the structured handoff state as a single JSON object with these keys:
"task" (one sentence), "relevant_files" (list of {{"path", "lines" (ranges seen, if any), "why"}}),
"facts" (observations established so far), "prior_actions" (what was done), "commitments"
(decisions and plans already made), "remaining_work".
Target about {target} characters; never more than {max_chars}. Output the JSON object only."""

SUMMARY_PROMPT = """## Task
{issue}

## Transcript of the work so far
{history}

Summarize the work so far in prose for the agent taking over: the relevant files and code
locations observed, what has been established, what was done and decided, and what remains.
Keep it under {cap} characters. Output the summary only."""


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def write_state(task: dict, writer: str) -> dict:
    history = history_text(task["history"])
    limit = min(STATE_MAX_CHARS, len(history) // 5)
    usage, feedback, text = [], "", ""
    for _ in range(3):
        prompt = STATE_PROMPT.format(issue=task["issue"], history=history, target=min(STATE_TARGET_CHARS, limit), max_chars=limit)
        result = complete(writer, [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt + feedback}],
                          max_tokens=4096, reasoning_effort="low")
        usage.append(result)
        text = _strip_fence(result["text"])
        try:
            json.loads(text)
            valid = True
        except ValueError:
            valid = False
        if valid and len(text) <= limit:
            return {"text": text, "valid": True, "calls": usage}
        feedback = f"\n\nYour previous answer was {'not valid JSON' if not valid else f'{len(text)} characters'}; it must be valid JSON of at most {limit} characters."
    return {"text": text[:limit], "valid": False, "calls": usage}


def write_summary(task: dict, writer: str, cap: int) -> dict:
    prompt = SUMMARY_PROMPT.format(issue=task["issue"], history=history_text(task["history"]), cap=cap)
    result = complete(writer, [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt}],
                      max_tokens=4096, reasoning_effort="low")
    text = result["text"].strip()
    return {"text": text[:cap], "truncated": len(text) > cap, "calls": [result]}


def recent(task: dict, budget: int) -> str:
    """The end of the history, verbatim, in at most `budget` o200k tokens."""
    from integrations.hscbench.llm import _encoding

    tokens = _encoding().encode(history_text(task["history"]), disallowed_special=())
    return _encoding().decode(tokens[-budget:]) if budget > 0 else ""


def cpc(task: dict, budget: int, url: str) -> dict:
    """CPC-Mem excerpt in at most `budget` o200k tokens. The service counts its own
    tokens, so the budget is corrected until the rendered excerpt fits."""
    entries = [{"role": "assistant", "label": "agent action and output", "text": pair_text(pair)} for pair in task["history"]]
    asked, result = budget, {}
    for _ in range(4):
        payload = {"entries": entries, "query": task["issue"][:3000], "budget": max(asked, 1)}
        request = urllib.request.Request(url.rstrip("/") + "/compress", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read())
        size = count_tokens(result["text"])
        if size <= budget:
            break
        asked = int(asked * budget / size) - 5
    return {"text": result["text"], "units_in": result.get("units_in"), "units_kept": result.get("units_kept"),
            "asked_budget": asked, "ms": result.get("ms")}


_OUTPUT_BLOCK = re.compile(r"^\[step (\d+), output\]$", re.MULTILINE)


def _head_tokens(text: str, limit: int) -> str:
    from integrations.hscbench.llm import _encoding

    tokens = _encoding().encode(text, disallowed_special=())
    return text if len(tokens) <= limit else _encoding().decode(tokens[:limit]) + " …"


def cpc_context(task: dict, budget: int, url: str) -> dict:
    """CPC-Mem on the context, not the messages: every step's tool call is kept
    verbatim (what was run or opened: commands, paths, line ranges) and CPC
    selects, from the tool outputs only, the lines that fill the rest of the
    budget; each kept line stays under the step that produced it. The tool
    calls take at most half the budget (each cut to an equal share if more)."""
    steps = task["history"]
    commands = [(pair.get("command") or "").strip() for pair in steps]
    if count_tokens("\n".join(commands)) > budget // 2:
        commands = [_head_tokens(command, budget // 2 // len(steps)) for command in commands]
    entries = [{"role": "tool", "label": "output", "text": pair["observation"]} for pair in steps]

    def assemble(excerpt: str) -> str:
        parts = _OUTPUT_BLOCK.split(excerpt)  # [before, step, lines, step, lines, ...]
        kept = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        blocks = []
        for number, command in enumerate(commands, 1):
            block = f"[step {number}] {command}".rstrip()
            if kept.get(number):
                block += f"\n--- output (excerpt) ---\n{kept[number]}"
            blocks.append(block)
        return "\n\n".join(blocks)

    skeleton = count_tokens(assemble(""))
    room = budget - skeleton
    asked, result, text = room, {}, assemble("")
    for _ in range(4):
        if asked <= 0:
            text = assemble("")
            break
        payload = {"entries": entries, "query": task["issue"][:3000], "budget": asked}
        request = urllib.request.Request(url.rstrip("/") + "/compress", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.loads(response.read())
        text = assemble(result["text"])
        size = count_tokens(text)
        if size <= budget:
            break
        asked = int(asked * room / max(1, size - skeleton)) - 5
    return {"text": text, "units_in": result.get("units_in"), "units_kept": result.get("units_kept"),
            "asked_budget": asked, "command_tokens": skeleton, "ms": result.get("ms")}


def extend(row: dict, task: dict, cpc_url: str) -> dict:
    """Add the context-compression conditions to an already built row."""
    for name, factor in (("cpc_ctx", 1), ("cpc_ctx_2x", 2)):
        excerpt = cpc_context(task, factor * row["budget"], cpc_url)
        row["contexts"][name] = excerpt["text"]
        row["tokens"][name] = count_tokens(excerpt["text"])
        row[name] = {key: excerpt[key] for key in ("units_in", "units_kept", "asked_budget", "command_tokens", "ms")}
    return row


def build(task: dict, writer: str, cpc_url: str) -> dict:
    state = write_state(task, writer)
    budget = count_tokens(state["text"])
    summary = write_summary(task, writer, cap=max(SUMMARY_CHARS[0], min(SUMMARY_CHARS[1], len(state["text"]))))
    excerpt, excerpt_2x = cpc(task, budget, cpc_url), cpc(task, 2 * budget, cpc_url)
    contexts = {
        "state": state["text"],
        "recent": recent(task, budget),
        "summary": summary["text"],
        "cpc": excerpt["text"],
        "cpc_2x": excerpt_2x["text"],
        "full": history_text(task["history"]),
    }
    return {
        "task_id": task["task_id"],
        "budget": budget,
        "contexts": contexts,
        "tokens": {name: count_tokens(text) for name, text in contexts.items()},
        "state_valid": state["valid"],
        "summary_truncated": summary["truncated"],
        "cpc": {key: excerpt[key] for key in ("units_in", "units_kept", "asked_budget", "ms")},
        "writer_calls": [{key: call[key] for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "error")}
                         for call in state["calls"] + summary["calls"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="logs_hsc/tasks.jsonl")
    parser.add_argument("--out", default="logs_hsc/contexts.jsonl")
    parser.add_argument("--writer", default="vertex_ai/gemini-3-flash-preview")
    parser.add_argument("--cpc-url", default="http://127.0.0.1:8765")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--extend", help="contexts file whose rows get the cpc_ctx conditions added (no LLM calls)")
    args = parser.parse_args()

    out = Path(args.out)
    done = {json.loads(line)["task_id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    tasks = [task for task in map(json.loads, Path(args.tasks).open(encoding="utf-8")) if task["task_id"] not in done]
    base = {row["task_id"]: row for row in map(json.loads, Path(args.extend).open(encoding="utf-8"))} if args.extend else {}
    if args.extend:
        tasks = [task for task in tasks if task["task_id"] in base]
    lock = threading.Lock()

    def run(task):
        row = extend(base[task["task_id"]], task, args.cpc_url) if args.extend else build(task, args.writer, args.cpc_url)
        with lock, out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        for count, row in enumerate(pool.map(run, tasks), 1):
            print(f"[{count}/{len(tasks)}] {row['task_id']} budget {row['budget']} tokens {row['tokens']}", flush=True)


if __name__ == "__main__":
    main()
