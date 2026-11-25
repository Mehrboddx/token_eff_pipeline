"""LLM-as-judge with the reference prompt (AMA-Bench utils/evaluation_metrics.py,
compute_llm_as_judge): yes/no, the last yes or no in the reply decides. The
same judge model scores every condition. An empty answer is wrong without a call.

    python -m integrations.amabench.judge --episodes logs_ama/episodes.jsonl \
        --answers logs_ama/answers.jsonl --out logs_ama/judged.jsonl
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from integrations.hscbench.llm import complete

PROMPT = """You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context}

Question: {question}

Reference Answer: {answer}

Predicted Answer: {predicted}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:<think></think>"""


def judge_prompt(episode: dict, qa: dict, predicted: str) -> str:
    context = "\n".join([f"Task Type: {episode.get('task_type', '')}", f"Episode ID: {episode['episode_id']}",
                         f"Task Context: {episode.get('task', '')}"])
    return PROMPT.format(context=context, question=qa["question"], answer=qa["answer"], predicted=predicted)


def verdict(text: str) -> float | None:
    """1.0 / 0.0 by the last standalone yes or no; None when there is neither."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).lower()
    yes = [m.start() for m in re.finditer(r"\byes\b", text)]
    no = [m.start() for m in re.finditer(r"\bno\b", text)]
    if not yes and not no:
        return None
    return 1.0 if (yes[-1] if yes else -1) > (no[-1] if no else -1) else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default="logs_ama/episodes.jsonl")
    parser.add_argument("--answers", default="logs_ama/answers.jsonl")
    parser.add_argument("--out", default="logs_ama/judged.jsonl")
    parser.add_argument("--judge", default="vertex_ai/gemini-3.6-flash")
    parser.add_argument("--reasoning", default="minimal")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    episodes = {e["episode_id"]: e for e in map(json.loads, Path(args.episodes).open(encoding="utf-8"))}
    questions = {qa["question_uuid"]: (episode, qa) for episode in episodes.values() for qa in episode["qa_pairs"]}
    out = Path(args.out)
    key = lambda r: (r["question_uuid"], r["condition"], r["reader"])
    done = set()
    if out.exists():
        done = {key(r) for r in map(json.loads, out.open(encoding="utf-8")) if not r["error"]}
    answers = [r for r in map(json.loads, Path(args.answers).open(encoding="utf-8"))
               if not r["error"] and r["question_uuid"] in questions and key(r) not in done]
    lock = threading.Lock()

    def run(row):
        episode, qa = questions[row["question_uuid"]]
        record = {k: row[k] for k in ("episode_id", "question_uuid", "condition", "reader")}
        if not row["answer"].strip():
            record.update(score=0.0, parsed=True, judge=args.judge, raw="", error="")
        else:
            result = complete(args.judge, [{"role": "user", "content": judge_prompt(episode, qa, row["answer"])}],
                              max_tokens=256, reasoning_effort=args.reasoning or None)
            score = verdict(result["text"])
            record.update(score=score or 0.0, parsed=score is not None, judge=args.judge, raw=result["text"][:300],
                          error=result["error"])
        with lock, out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    with ThreadPoolExecutor(args.workers) as pool:
        for count, record in enumerate(pool.map(run, answers), 1):
            if count % 100 == 0 or record["error"]:
                print(f"[{count}/{len(answers)}] {'ERROR ' + record['error'] if record['error'] else 'ok'}", flush=True)


if __name__ == "__main__":
    main()
