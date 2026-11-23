# Agent context management on SWE-bench: CPC recall

A coding agent's context fills up with tool outputs: file views, grep hits, test logs. "The Complexity Trap"
(Lindenbauer et al., 2025, [arXiv 2508.21433](https://arxiv.org/abs/2508.21433)) compares three ways of handling
this in SWE-agent on SWE-bench Verified:

- **raw:** the full history.
- **observation masking:** tool outputs older than the last 10 are replaced by "Old environment output: (N lines
  omitted)".
- **LLM summary:** every 21 turns are summarized by the model, and the last 10 are kept verbatim.

Masking turned out as good as summaries, at lower cost. Its weakness is that once an output is masked, the agent
cannot see it again unless it re-runs the command.

**CPC recall** is masking plus recall. At every step:

1. The history is masked exactly as in the baseline, so the conversation prefix and prompt caching are the same.
2. The compressor selects the lines most relevant to what the agent is doing now from all masked outputs. The query
   is the agent's latest thought and action.
3. The selection is appended to the newest tool output as a recall block, at most `RECALL_BUDGET` tokens (default
   2,000). Each excerpt sits under the command that produced it, which records the file, path and lines it came from.

Different steps recall different parts of the session, and nothing is deleted. The compressor is CPC + Provence
fusion (W = 0.25), the CooperBench service. If the service fails, that step falls back to plain masking.

## Setup

- **Code:** the paper's SWE-agent fork, pinned to commit `bf15b5f`, with its own configs for raw, masking and
  summary. CPC recall uses the masking config with the history processor swapped. It is registered from outside the
  fork (`launch.py`), so the fork's code is unchanged.
- **Tasks:** [SWE-bench Verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini). These
  are 50 tasks (25 django, 25 sphinx) with about the same difficulty spread as Verified, and about 5 GB of images.
- **Model:** Gemini 2.5 Flash without thinking, as in the paper. Temperature 0.8, at most 250 calls and $1.50 per
  task.
- **The paper's results on these 50 tasks** (`paper_instance_results.csv`, from its released data):

  | Policy | Solved | Cost per task |
  |---|---|---|
  | masking | 30% | $0.12 |
  | summary | 22% | $0.26 |
  | raw | 20% (49 tasks) | $0.58 |

## Run (GPU machine)

```bash
bash scripts/swebench_ctx.sh setup
bash scripts/cooperbench.sh server                      # compressor, GPU 1
VERTEX_PROJECT=... bash scripts/swebench_ctx.sh run mask mask1
VERTEX_PROJECT=... bash scripts/swebench_ctx.sh run recall recall1
bash scripts/swebench_ctx.sh eval mask1; bash scripts/swebench_ctx.sh eval recall1
bash scripts/swebench_ctx.sh report mask1 recall1
bash scripts/cooperbench.sh stop
```

At n = 50 and temperature 0.8, a difference of a few tasks is within noise. `report` gives paired per-task counts
and a sign test against our masking run and the paper's runs.
