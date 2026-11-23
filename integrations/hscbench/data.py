"""Build the repository-track tasks from public data.

HSC-Bench (Section 3.2, Appendix A) pairs the code-location annotations of
ContextBench's 500-task verified subset with GPT-5 trajectories from
CodeTraceBench (OpenHands, SWE-agent, mini-SWE-agent), and hands each
trajectory over after ten complete action-observation pairs. The paper's
final 300 tasks also passed a manual screen that is not published, so this
rebuild keeps every eligible task, one trajectory per task, chosen
deterministically (OpenHands, then SWE-agent, then mini-SWE-agent, as the
paper's 193/52/55 mix leans the same way).

    python -m integrations.hscbench.data --out logs_hsc/tasks.jsonl [--limit N]
"""

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from integrations.hscbench.trajectories import cap_observation, from_miniswe, from_openhands, from_sweagent

CODETRACE = "https://huggingface.co/datasets/NJU-LINK/CodeTraceBench/resolve/main/"
CONTEXTBENCH = "https://huggingface.co/datasets/Contextbench/ContextBench/resolve/main/data/contextbench_verified.parquet"
HANDOFF_PAIRS = 10
AGENT_ORDER = ["OpenHands", "SWE-agent", "mini-SWE-agent"]


def fetch(url: str, cache: Path) -> Path:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as response:
            cache.write_bytes(response.read())
    return cache


def read_artifact(path: Path) -> dict[str, bytes]:
    import zstandard

    files = {}
    with path.open("rb") as raw, tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(raw), mode="r|") as tar:
        for member in tar:
            if member.isfile():
                files[member.name] = tar.extractfile(member).read()
    return files


def pairs_of(agent: str, files: dict[str, bytes]) -> list[dict]:
    if agent == "mini-SWE-agent":
        name = next(n for n in files if n.endswith(".traj.json"))
        return from_miniswe(json.loads(files[name]))
    if agent == "SWE-agent":
        name = next(n for n in files if n.endswith(".traj"))
        return from_sweagent(json.loads(files[name]))
    logs = sorted((n for n in files if n.rsplit("/", 1)[-1].startswith("gpt-5-") and n.endswith(".json")),
                  key=lambda n: float(n.rsplit("gpt-5-", 1)[1][:-5]))
    return from_openhands(json.loads(files[logs[-1]])) if logs else []


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="logs_hsc/tasks.jsonl")
    parser.add_argument("--cache", default="third_party/hsc-cache")
    parser.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke test)")
    args = parser.parse_args()
    cache = Path(args.cache)

    contextbench = pd.read_parquet(fetch(CONTEXTBENCH, cache / "contextbench_verified.parquet"))
    manifest = pd.read_parquet(fetch(CODETRACE + "bench_manifest.full.parquet", cache / "bench_manifest.full.parquet"))
    tasks = contextbench.set_index("original_inst_id")
    runs = manifest[(manifest.model == "OpenAI/GPT-5") & manifest.task_name.isin(set(tasks.index)) & manifest.agent.isin(AGENT_ORDER)]
    runs = runs.assign(order=runs.agent.map(AGENT_ORDER.index)).sort_values(["task_name", "order", "traj_id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, skipped = 0, {"short": 0, "unreadable": 0}
    with out.open("w", encoding="utf-8") as sink:
        for task_name, candidates in runs.groupby("task_name", sort=True):
            if args.limit and kept >= args.limit:
                break
            for run in candidates.itertuples():
                try:
                    pairs = pairs_of(run.agent, read_artifact(fetch(CODETRACE + run.artifact_path, cache / run.artifact_path)))
                except Exception as error:  # a broken archive: try the next trajectory of this task
                    print(f"unreadable {run.traj_id}: {type(error).__name__}: {error}")
                    skipped["unreadable"] += 1
                    continue
                if len(pairs) <= HANDOFF_PAIRS:  # something must remain to be taken over
                    skipped["short"] += 1
                    continue
                task = tasks.loc[task_name]
                sink.write(json.dumps({
                    "task_id": task.instance_id,
                    "original_id": task_name,
                    "repo": task.repo,
                    "language": task.language,
                    "source": task.source,
                    "issue": task.problem_statement,
                    "gold": [{"file": span["file"], "start": int(span["start_line"]), "end": int(span["end_line"])}
                             for span in json.loads(task.gold_context)],
                    "agent": run.agent,
                    "traj_id": run.traj_id,
                    "pairs_total": len(pairs),
                    "history": [{**pair, "observation": cap_observation(pair["observation"])} for pair in pairs[:HANDOFF_PAIRS]],
                }, ensure_ascii=False) + "\n")
                kept += 1
                break
    print(f"{kept} tasks -> {out}; skipped trajectories {skipped}")


if __name__ == "__main__":
    main()
