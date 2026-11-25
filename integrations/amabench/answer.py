"""The reader answers each question from one context, with the reference answer
prompt (AMA-Bench src/memory_interface.py): one question per call for the
budgeted contexts; the full trajectory gets all of an episode's questions in
one call, as the reference long-context method does.

    python -m integrations.amabench.answer --episodes logs_ama/episodes.jsonl \
        --contexts logs_ama/contexts.jsonl --out logs_ama/answers.jsonl
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from integrations.amabench.contexts import step_block, truncate
from integrations.hscbench.llm import complete

FULL_LIMIT = 700_000  # cl100k tokens; only the longest trajectory (~1M) is cut, head 70% + tail 30%

HEADER = ("## Task Description\n{task}\n\n## Agent Trajectory\n"
          "The following is a step-by-step trajectory of the agent's actions and observations{note}:\n\n{context}")
PARTIAL_NOTE = " (only some steps and lines are shown)"
SINGLE = ("{memory}\n\n## Questions\nQuestion 1: {question}\n\n## Instructions\n"
          "Provide a direct and concise answer.\n\nAnswer[1]: [your answer here]")
BATCH = ("{memory}\n\n## Questions\nPlease answer the following questions based on the task description and agent "
         "trajectory above. For each question, provide a direct and concise answer.\n\n{questions}\n"
         "## Instructions\nPlease provide answers in the following format:\n\n{slots}")


def single_prompt(task: str, context: str, question: str) -> str:
    memory = HEADER.format(task=task, note=PARTIAL_NOTE, context=context)
    return SINGLE.format(memory=memory, question=question)


def batch_prompt(task: str, context: str, questions: list[str]) -> str:
    memory = HEADER.format(task=task, note="", context=context)
    listed = "\n".join(f"Question {i}: {q}\n" for i, q in enumerate(questions, 1))
    slots = "\n".join(f"Answer[{i}]: [your answer here]" for i in range(1, len(questions) + 1))
    return BATCH.format(memory=memory, questions=listed, slots=slots)


def parse_answers(text: str, count: int) -> list[str]:
    """Answer[i]: ... up to the next Answer[j]: marker; a single answer with no
    marker is the whole reply."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    marks = list(re.finditer(r"Answer\[(\d+)\]:", text))
    answers = [""] * count
    for k, mark in enumerate(marks):
        index = int(mark.group(1)) - 1
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        if 0 <= index < count and not answers[index]:
            answers[index] = text[mark.end():end].strip()
    if count == 1 and not marks:
        answers[0] = text
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default="logs_ama/episodes.jsonl")
    parser.add_argument("--contexts", default="logs_ama/contexts.jsonl")
    parser.add_argument("--out", default="logs_ama/answers.jsonl")
    parser.add_argument("--reader", default="vertex_ai/gemini-3.5-flash-lite")
    parser.add_argument("--reasoning", default="minimal")
    parser.add_argument("--conditions", default="full,trunc,bm25,actions,random,cpc,act_random,act_cpc")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    episodes = {e["episode_id"]: e for e in map(json.loads, Path(args.episodes).open(encoding="utf-8"))}
    contexts = {}
    if Path(args.contexts).exists():
        for row in map(json.loads, Path(args.contexts).open(encoding="utf-8")):
            if not row["error"]:
                contexts[(row["question_uuid"], row["condition"])] = row
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["question_uuid"], r["condition"], r["reader"]) for r in map(json.loads, out.open(encoding="utf-8"))
                if not r["error"]}
    conditions = args.conditions.split(",")
    jobs = []  # (episode, [qa], condition)
    for episode in episodes.values():
        for condition in conditions:
            todo = [qa for qa in episode["qa_pairs"] if (qa["question_uuid"], condition, args.reader) not in done]
            if condition == "full":
                if todo:
                    jobs.append((episode, todo, condition))
            else:
                jobs.extend((episode, [qa], condition) for qa in todo if (qa["question_uuid"], condition) in contexts)
    lock = threading.Lock()

    def run(job):
        episode, qas, condition = job
        if condition == "full":
            text = truncate("\n".join(step_block(step) for step in episode["trajectory"]), FULL_LIMIT)
            prompt = batch_prompt(episode["task"], text, [qa["question"] for qa in qas])
            max_tokens = 1024 * len(qas)
        else:
            prompt = single_prompt(episode["task"], contexts[(qas[0]["question_uuid"], condition)]["text"], qas[0]["question"])
            max_tokens = 1024
        result = complete(args.reader, [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                          reasoning_effort=args.reasoning or None)
        answers = parse_answers(result["text"], len(qas))
        rows = [{"episode_id": episode["episode_id"], "question_uuid": qa["question_uuid"], "condition": condition,
                 "reader": args.reader, "answer": answer, "raw": result["text"][:6000] if len(qas) == 1 else "",
                 "prompt_tokens": result["prompt_tokens"] / len(qas), "completion_tokens": result["completion_tokens"] / len(qas),
                 "error": result["error"]} for qa, answer in zip(qas, answers)]
        with lock, out.open("a", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rows[0]

    with ThreadPoolExecutor(args.workers) as pool:
        for count, row in enumerate(pool.map(run, jobs), 1):
            if count % 50 == 0 or row["error"]:
                print(f"[{count}/{len(jobs)}] {row['condition']} {'ERROR ' + row['error'] if row['error'] else 'ok'}",
                      flush=True)


if __name__ == "__main__":
    main()
