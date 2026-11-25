"""AMA-Bench open-ended QA set (HuggingFace AMA-bench/AMA-bench, MIT): download,
format trajectories as the reference code does, and draw the pilot sample.

    python -m integrations.amabench.data --out logs_ama/episodes.jsonl --per-domain 8
"""

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

URL = "https://huggingface.co/datasets/AMA-bench/AMA-bench/resolve/main/test/open_end_qa_set.jsonl"


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")  # what the compressor service budgets in


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text or "", disallowed_special=()))


def step_text(step: dict) -> str:
    return f"Action: {step.get('action') or ''}\nObservation: {step.get('observation') or ''}"


def trajectory_text(trajectory: list[dict]) -> str:
    """The reference formatting (AMA-Bench src/memory_interface.py, _trajectory_to_text)."""
    return "\n".join(f"Step {step.get('turn_idx', 0)}:\n{step_text(step)}\n" for step in trajectory)


def sample(episodes: list[dict], per_domain: int, seed: int = 0) -> list[dict]:
    """per_domain episodes from each domain, uniformly at random (long ones included)."""
    by_domain = defaultdict(list)
    for episode in episodes:
        by_domain[episode["domain"]].append(episode)
    rng = random.Random(seed)
    chosen = []
    for domain in sorted(by_domain):
        pool = sorted(by_domain[domain], key=lambda episode: episode["episode_id"])
        chosen.extend(rng.sample(pool, min(per_domain, len(pool))))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="logs_ama/episodes.jsonl")
    parser.add_argument("--raw", default="logs_ama/open_end_qa_set.jsonl")
    parser.add_argument("--per-domain", type=int, default=0, help="0: every episode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, raw)
    episodes = [json.loads(line) for line in raw.open(encoding="utf-8")]
    if args.per_domain:
        episodes = sample(episodes, args.per_domain, args.seed)
    with Path(args.out).open("w", encoding="utf-8") as sink:
        for episode in episodes:
            episode["full_tokens"] = count_tokens(trajectory_text(episode["trajectory"]))
            sink.write(json.dumps(episode, ensure_ascii=False) + "\n")
    questions = sum(len(episode["qa_pairs"]) for episode in episodes)
    print(f"{len(episodes)} episodes, {questions} questions -> {args.out}")


if __name__ == "__main__":
    main()
