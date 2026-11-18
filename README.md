# CPC-Mem: compressed memory for long-running agents

CPC-Mem keeps a long conversation history inside a small token budget. When a
new question arrives, it scores every sentence of the history against the
question and passes only the best-scoring sentences (for example 1k or 4k
tokens) to the answering model, in their original order with session dates.
Older turns are compressed; the most recent turns are always kept word for word.

```
 history (30k-110k tokens)                                   answer model
 ───────────────────────┐                                   ┌──────────────┐
 session 1 ... session N │──► split into sentences ──► score ──► select ──►│ Gemini       │──► answer
 + new question          │      (units)          CPC (+Provence) budget    │ (any LLM)    │
 ───────────────────────┘                                   └──────────────┘
```

- **Scorer:** CPC (Liskavets et al., AAAI 2025) with a Mistral-7B or Llama encoder scores each sentence
  against the question. Optionally, the scores of Provence (Chirkova et al., ICLR 2025) are fused in
  (`--fuse-provence`), which is the best configuration we found.
- **Selection:** greedy by score under the token budget, rendered as dated sessions with speaker labels.
- **Evaluation:** [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (500 questions) and
  [LoCoMo](https://github.com/snap-research/locomo) (1,540 questions), answered by Gemini and graded by an
  LLM judge.

## Results

Gemini 2.5 Flash as reader, temperature 0, accuracy judged by an LLM. p = exact McNemar test against CPC-Mem
on the same questions.

| Benchmark, budget | CPC-Mem | CPC-Mem + Provence fusion | p |
| --- | --- | --- | --- |
| LoCoMo, 4k tokens | 78.5 | **82.1** | < .001 |
| LoCoMo, 2k tokens (vs CPC-Mem at 4k) | 78.5 | 80.5 | .031 |
| LongMemEval, 1k tokens | 80.8 | **85.0** | .019 |
| LongMemEval, 4k tokens | 82.0 / 82.6 | 84.2 | .22 |

For reference, the full uncompressed history scores 80.0 on LongMemEval (about 107k tokens per question)
and 81.6 on LoCoMo (about 29k tokens).

## Repository layout

```
core/        the library: memory, sentence units, scorers, selection, compressors, agent loop
  token_wise.py       TokenWise: score units, fuse scores, select under a budget
  selection.py        score fusion (z-normalized), greedy selection, MMR, rendering budget
  cpc_compressor.py   CPC scorer (wraps replicate/cpc_compressor.py)
  provence.py         Provence scorer
  entities.py         entity-key scorer (spaCy keys, idf overlap)
  dycp.py, gemini_compressor.py   baselines
  memory.py, units.py, runner.py, agent.py   conversation memory and the agent loop
eval/        benchmarks: the main runner, data loaders, evidence recall, score cache
  longmemeval.py      compress / answer / judge stages for both benchmarks
  compare_runs.py     paired McNemar test and per-type accuracy between two runs
  entity_keys.py      precompute entity keys for --entity-keys
  ablate.py, locomo_f1.py, evidence.py, locomo.py, score_cache.py
analysis/    studies on finished runs: budget gates, self-escalation, fusion analysis, latency, profiling
scripts/     shell helpers: GPU machine setup, Vertex AI deploy, a run suite
replicate/   standalone port of the original CPC code on LongBench (run from inside that folder)
tests/       unit tests (no models or network needed)
main.py      an interactive chat agent with compressed memory
```

## Setup

You need Python 3.11, a Google Cloud project with Vertex AI enabled (for Gemini), and a CUDA GPU for
the CPC encoder: Mistral-7B peaks at about 15 GB, or 20 GB with Provence fusion (measured on an RTX 4090).
The lexical and Gemini compressors run on CPU.

### Option A: Docker (recommended on the GPU machine)

```bash
docker build -f Dockerfile.eval -t longmemeval-eval .
docker run --rm --gpus all \
  -v "$PWD/logs:/app/logs" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e HF_TOKEN \
  longmemeval-eval --dataset locomo --limit 20 --compressor cpc --cpc-preset mistral --stage compress
```

The image's entry point is `python -m eval.longmemeval`; for other modules use
`--entrypoint python longmemeval-eval -m <module> ...`.

### Option B: local Python environment

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m spacy download en_core_web_sm                # only for entity keys
```

`scripts/setup_gpu_machine.sh` does the same on a fresh GPU machine.

### Credentials

Create a `.env` file in the project root (it is git-ignored):

```bash
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_CLOUD_LOCATION=us-central1
HF_TOKEN=hf_...            # Hugging Face token; the Llama/Mistral base models are gated
```

and log in once with `gcloud auth application-default login`. The benchmark data downloads on first use
(LongMemEval from Hugging Face, LoCoMo from its GitHub repository at a pinned commit, checksum-verified).

## Quick start: one benchmark run

A run has three stages, so the GPU part and the API part can run on different machines.

```bash
# 1. Compress (GPU). Writes logs/eval_longmemeval_run<N>_<settings>.jsonl
python -m eval.longmemeval --stage compress --dataset locomo --limit 1540 \
    --compressor cpc --cpc-preset mistral --token-budget 4000 \
    --fuse-provence 0.25 --score-cache logs/score_cache

# 2. Answer with Gemini (no GPU). Writes logs/gemini_runs/<same name>_t0.jsonl
python -m eval.longmemeval --stage answer --answer-temperature 0 \
    --from-log logs/eval_longmemeval_run0001_compress_locomo_cpc-mistral_b4000_fuse0.25.jsonl

# 3. Judge (no GPU). Writes logs/judge_runs/<same name>.jsonl
python -m eval.longmemeval --stage judge \
    --from-log logs/gemini_runs/eval_longmemeval_run0001_compress_locomo_cpc-mistral_b4000_fuse0.25_t0.jsonl

# 4. Compare two judged runs (paired McNemar test, accuracy per question type)
python -m eval.compare_runs logs/judge_runs/<baseline>.jsonl logs/judge_runs/<variant>.jsonl
```

Every stage can be stopped and continued: run the same command again, or pass `--resume <run number>`.
Questions whose API call failed are retried on the next run. The compress log already reports
**evidence recall** (the share of gold evidence turns kept), which is free to compute and useful for
choosing settings before paying for answers.

Use `--stage full` to compress and answer in one go, and `--limit 20` for a quick smoke test.

## Main options (`python -m eval.longmemeval --help` for all)

| Option | What it does |
| --- | --- |
| `--dataset longmemeval_s` / `locomo` | Benchmark |
| `--compressor cpc` / `provence` / `dycp` / `lexical` / `gemini` / `none` | Sentence scorer, or `none` for the full history |
| `--cpc-preset mistral` / `llama` | CPC base model |
| `--token-budget 4000` | Tokens of compressed history given to the reader |
| `--fuse-provence 0.25` | Blend Provence's scores into CPC's with this weight (recommended) |
| `--score-cache logs/score_cache` | Cache sentence scores on disk; reruns with other selection settings need no GPU |
| `--entity-keys FILE --entity-weight 0.3` | Blend in entity-key scores (run `eval.entity_keys` first) |
| `--answer-temperature 0` | Answer at temperature 0 (recommended: halves run-to-run noise) |
| `--answer-model` | Answer with another Gemini model |
| `--reader plain` / `memory` / `notes` | Answer prompt: plain, "this is your memory of the user", or Chain-of-Note |
| `--resume N` | Continue run N |

## Reproducing the reported numbers

```bash
for budget in 4000 2000; do
  python -m eval.longmemeval --stage compress --dataset locomo --limit 1540 --compressor cpc \
      --cpc-preset mistral --token-budget $budget --fuse-provence 0.25 --score-cache logs/score_cache
done
for budget in 4000 1000; do
  python -m eval.longmemeval --stage compress --dataset longmemeval_s --limit 500 --compressor cpc \
      --cpc-preset mistral --token-budget $budget --fuse-provence 0.25 --score-cache logs/score_cache
done
```

Then answer each log at temperature 0 and judge it (steps 2 and 3 above). The CPC-Mem baselines are the
same commands without `--fuse-provence`. With `--score-cache`, the baseline and fused runs share CPC's
scores, so the second one only adds Provence's.

### Entity keys

```bash
python -m eval.entity_keys --dataset locomo --source ner        # writes logs/entity_keys/locomo_ner.json
python -m eval.longmemeval --stage compress --dataset locomo --limit 1540 --compressor cpc \
    --cpc-preset mistral --token-budget 4000 --score-cache logs/score_cache \
    --entity-keys logs/entity_keys/locomo_ner.json --entity-weight 0.3
```

## Analyses on finished runs

These read existing logs and cached scores; none of them load a model unless noted.

| Command | What it does |
| --- | --- |
| `python -m analysis.adaptive_budget` | Per-question budget policies vs. fixed budgets |
| `python -m analysis.score_gate` | Rule and learned budget gates on CPC's score profile |
| `python -m analysis.escalate` | Self-escalation: re-answer with more memory when the answer says it's missing (calls Gemini) |
| `python -m analysis.fusion_analysis` | Which evidence CPC, Provence and their fusion each keep |
| `python -m analysis.latency` | Compression latency and GPU memory (loads the models) |
| `python -m eval.locomo_f1 <judge logs>` | LoCoMo's original token-F1 metric |

## The interactive agent

`main.py` is a chat agent whose memory is compressed the same way:

```bash
COMPRESSOR_BACKEND=cpc python main.py     # or "local" (word overlap, CPU) or "gemini"
```

## Development

```bash
python -m unittest discover -s tests -t .      # 71 tests, no GPU or network needed
ruff check . && ruff format --check .
pyright
```
