# CPC-Mem on CooperBench

[CooperBench](https://github.com/cooperbench/CooperBench) ([paper](https://arxiv.org/abs/2601.13295))
pairs two coding agents on one repository. Each implements a different feature, and they coordinate only through
messages. A pair passes when both features' tests pass on the merge of the two patches. This integration places
the CPC + Provence compressor between the agents and asks two questions: does it change how often pairs succeed,
and how many LLM tokens does it save?

## Agent and arms

The harness is CooperBench's `mini_swe_agent_v2`. It is the benchmark's default and, in the authors' `core`-subset
comparison, the strongest of the four harnesses (6/10, against 5/10 for `claude_code`, 5/10 for `codex` and 4/10
for `openhands_sdk`). The model is the benchmark's default, `gemini-3-flash-preview`. The stock agent already
compresses its own history: once a prompt reaches 28k tokens, an extra LLM call summarises everything except the
last two turns.

| arm (`arms/*.yaml`) | colleague messages | own history at 28k prompt tokens |
|---|---|---|
| `none` (baseline) | verbatim | LLM summary (stock) |
| `messages` | compressed to 50% (messages of 150 tokens or less untouched) | LLM summary (stock) |
| `full` | compressed to 50% | compressor keeps the most relevant verbatim lines, 6,000-token budget, no LLM call |

What is scored against what:

- **Messages:** scored against the receiving agent's task and its latest step.
- **History:** scored against the latest step and the task.

Units are built the same way for every input: lines (split into sentences where a line holds several), indentation kept; pieces under four words
are merged into a neighbour. The scorers and the fusion weight (W = 0.25) are those of the LongMemEval/LoCoMo
runs.

For every agent, every LLM call (agent steps and the stock summary calls) and every compression is logged to
`<pair log dir>/<agent>_ledger.jsonl`. All three arms are therefore measured identically.

## Files

- `compress_server.py`: HTTP service around the compressor. It runs in the `longmemeval-eval` image on one GPU.
- `agent.py`: `CpcAgent`, a subclass of the harness's `DefaultAgent`, plus `install()`, which makes the adapter
  build it. It needs only CooperBench and the standard library.
- `launch.py`: CooperBench's own CLI, with the agent installed first.
- `report.py`: pass rates and tokens per arm. Each arm is compared with the baseline on common pairs (exact
  McNemar test; paired bootstrap CI on the change in tokens).
- `render.py`: excerpt layout. Lines stay lines, and `…` marks omitted lines.
- `../../scripts/cooperbench.sh`: setup, service, run, eval and report, all in Docker.

## Running

On the GPU machine, from the repository root:

```bash
docker build -f Dockerfile.eval -t longmemeval-eval .   # image now includes integrations/
scripts/cooperbench.sh setup                            # pinned CooperBench clone + venv + flash task images
scripts/cooperbench.sh server                           # compressor on GPU $GPU (default 1), port 8765

# the agents' LLM: an OpenAI-compatible endpoint at $LLM_BASE, key in .llm_proxy_key
scripts/cooperbench.sh run none     flash_none     -s flash
scripts/cooperbench.sh run messages flash_messages -s flash
scripts/cooperbench.sh run full     flash_full     -s flash
scripts/cooperbench.sh report flash_none flash_messages flash_full
```

`run` evaluates each pair automatically (CooperBench's auto-eval), and `eval <name>` re-runs evaluation. For a
smoke test, pass `-s flash_10` or `-r <repo> -t <task> -f <i,j>`.
