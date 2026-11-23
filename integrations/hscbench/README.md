# HSC-Bench repository track with CPC-Mem as the handoff

[HSC-Bench](https://arxiv.org/abs/2610.36523) (Li et al., 2026) measures how well an agent takes over unfinished
work. In its repository track, a GPT-5 coding agent's trajectory is cut after ten action-observation pairs and
handed to a receiving model. The receiver predicts the files and line spans relevant to the issue, scored by
line-F1 against ContextBench's expert annotations. The paper compares four handoff representations at a matched
token count: recent transcript, structured state (GPT-4.1), narrative summary, and full history.

The paper's code and data are not released, so the track is rebuilt here from the same public sources:

- [ContextBench](https://github.com/EuniAI/ContextBench), verified subset (500 tasks): issue texts and gold code
  locations.
- [CodeTraceBench](https://huggingface.co/datasets/NJU-LINK/CodeTraceBench): GPT-5 trajectories of OpenHands,
  SWE-agent and mini-SWE-agent.

## Differences from the paper

- **Tasks:** every eligible task is kept, with one trajectory per task. Eligible means the trajectory has more than
  ten complete pairs. Agents are preferred in the order OpenHands, SWE-agent, mini-SWE-agent. The paper's final
  300 tasks went through a manual screen that is not published.
- **Writer:** the structured state and the summary are written by `--writer` (default Gemini 3 Flash) instead of
  GPT-4.1, following the paper's constraints:
  - the state is valid JSON, about 1,800 characters, at most 4,000;
  - the state is at least five times shorter than its source;
  - the summary is capped at the state's length, bounded to 200–1,600 characters.

  Their prompts are not published, so they are written here (`handoff.py`).
- **Receiver prompt:** written here (`receive.py`). The settings follow the paper: temperature 0, top-p 1, 1,024
  output tokens, minimal reasoning.

## Conditions

All compact conditions are matched to the structured state's o200k token count per task, as in the paper.

| Condition | Handoff |
|---|---|
| `recent` | the end of the history, verbatim |
| `summary` | LLM prose summary |
| `state` | LLM structured JSON state |
| `cpc` | CPC + Provence (W = 0.25): the history lines most relevant to the issue, verbatim; no LLM call |
| `cpc_2x` | the same, at twice the budget |
| `full` | the whole ten-pair history |

## Run (GPU machine)

```bash
bash scripts/hscbench.sh setup
bash scripts/cooperbench.sh server                 # the compressor service, GPU 1
VERTEX_PROJECT=... bash scripts/hscbench.sh data   # [--limit N] for a smoke test
VERTEX_PROJECT=... bash scripts/hscbench.sh contexts
VERTEX_PROJECT=... bash scripts/hscbench.sh receive --receiver vertex_ai/gemini-3.5-flash-lite
VERTEX_PROJECT=... bash scripts/hscbench.sh score
bash scripts/cooperbench.sh stop
```
