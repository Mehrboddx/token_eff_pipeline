# Experiment journal

Autonomous experiments on top of CPC-Mem (CPC Mistral-7B, 4k cl100k budget, dated session layout).
Answering: Gemini 2.5 Flash, default settings; grading: LLM judge (Gemini, temperature 0).
Significance: paired bootstrap (10k) or exact McNemar, as stated per row.

Reference numbers (LLM judge, %):

| | LongMemEval (500) | LoCoMo (1,540) |
|---|---|---|
| CPC-Mem 4k | 83.0 | 78.6 |
| Full history | 80.0 (107k tokens) | 81.6 (29k tokens) |

---

## 2026-11-16 · Adaptive budget: new rungs

GPU runs 47–50 (server, GPU 1): LongMemEval 0.5k and 1k, LoCoMo 2k and 8k, all from the score cache.
Gate features (CPC score profile per question) for all 500 + 1,540 questions.

LongMemEval fixed budgets (judge): 0.5k **74.6** · 1k **81.0** · 2k 79.2 · 4k 83.0 · full 80.0.
1k already matches full history with 1% of its tokens; 2k below 1k is answer noise.

Gates on LongMemEval, compared with the best random mix of two fixed budgets at the same mean tokens:
- rule gate (tokens of units with z ≥ τ): worse than the mix (−3.4 at τ 2.5–3.0, p ≤ 0.01);
- learned gate (logistic regression on the score profile, out-of-fold): mostly level with the mix;
  +2.0 at θ 0.75 (614 tokens, p 0.005), but θ was chosen after looking — a hint, not a result.
- "ideal gate" 88.4 at 675 tokens: 27 questions right at 0.5k but wrong at 4k — mostly
  sampling noise of a single temperature-1.0 answer, which caps what any gate can learn.

LoCoMo (old ladder 4k / full): learned gate at θ 0.85 = 80.5 at 7.2k tokens vs 79.0 for the
best mix (+1.4, p < 0.001). New ladder with 2k / 8k: answering in progress.

Side result: the paper-faithful CPC's 93 s per question is 71 s of `pysbd` sentence splitting
over the whole transcript; GPU encoding is 24 s vs our 21.5 s. Paper text updated.

## 2026-11-17 · Chain-of-Note reader

Idea (LongMemEval paper, Wu et al. 2025: up to +10 points): the answering model first lists
the relevant context with session dates, resolves relative times, reasons, then answers on a
final `Answer:` line; only that line is judged. No GPU — reuses the 4k contexts of runs 1 and 21.
Code: `--stage answer --reader notes` (branch `exp/adaptive-gate-and-notes`).

**LongMemEval: 83.8 vs 83.0** (25 better / 21 worse, McNemar p 0.66) — no real gain.
Multi-session 79.7 → 83.5, knowledge-update 87.2 → 89.7, but single-session-assistant
83.9 → 78.6. Gemini 2.5 Flash already thinks before answering, which likely covers most of what
explicit notes add. LoCoMo: running.

## 2026-11-17 · Answer noise: the same run twice ⚠

Re-answered the LongMemEval CPC-Mem 4k contexts (run 1) with the identical setup — same
contexts, same prompt, Gemini 2.5 Flash at its default temperature 1.0 — six weeks after the
original: **80.2 vs 83.0**, 52 of 500 questions flipped (19 better / 33 worse, McNemar p 0.07).
Multi-session 79.7 → 72.2, temporal 87.2 → 83.5.

So a single answer run moves by about ±3 points on its own (part of it may be model drift since
1 Oct). Most single-run differences in the paper and above are within this. Next: answer at
temperature 0 (`--answer-temperature 0`, added) twice and count flips; if they drop sharply,
re-answer every main configuration at temperature 0 for the paper. Otherwise report means
over ≥3 runs.

## 2026-11-17 · Self-escalating memory (offline)

Idea: answer from a small budget; if the answer itself signals missing memory, answer again
from the next budget up. A Gemini classifier (temperature 0) sees only question + answer,
never the reference, and says FOUND or MISSING. `eval/escalate.py`; costs counted as the tokens
of every budget tried.

LongMemEval: the signal is strong — FOUND answers are 86–89% correct, MISSING ones 34–48% —
but escalation barely pays: a question that is MISSING at 0.5k is usually MISSING at 1k and 2k
too. Best policy: 4k, and only MISSING questions (15%) go to full history → **84.2 at 19.7k
mean tokens** (+1.2 over 4k, vs best mix p 0.21, n.s.).

The underlying finding matters more: at 4k, 97.5% of LongMemEval questions have their evidence
in the context, yet accuracy is 83%. **The reader is the bottleneck, not the memory.**

## 2026-11-17 · Stronger readers on compressed memory (running)

Idea: compression makes a much stronger answer model affordable. Gemini 3.1 Pro on a 4k
context costs about $0.02 per question (measured tokens: ~7k prompt incl. system prompt,
~0.6k output+thinking), below Gemini 2.5 Flash on the full history ($0.033). Running on the
LongMemEval 4k contexts: `gemini-3.6-flash` and `gemini-3.1-pro-preview` (judge unchanged:
Gemini 2.5 Flash, temperature 0). `--answer-model`; answer rows now log token usage.

## 2026-11-17 · CPC + Provence score fusion (GPU, running)

Idea: the two scorers are complementary — CPC wins on LongMemEval (83.0 vs 79.4), Provence on
LoCoMo (81.3 vs 78.6); Provence keeps answer-like assistant text, CPC the user's own statements.
Blend their per-query z-scores: `--fuse-provence W` (Provence scores cached like CPC's).
Smoke test on 5 questions: peak GPU memory 7.2 GB (CPC from cache, only Provence loaded).
Queued on the server, GPU 1: W = 0.5 and 0.25, both benchmarks.

Server note: rebuilding `longmemeval-eval:latest` in place leaves the previous build as an
untagged ("dangling") image; not deleted — your call (`docker image prune` removes them).

## Idea queue

1. Chain-of-Note reader — LongMemEval done (n.s.), LoCoMo running.
2. Learned budget gate on LoCoMo with 2k/4k/8k/full — waiting on the 8k judge.
3. Answer noise — measured (±3 points per run). Next: temperature-0 answers, twice, on the
   LongMemEval 4k contexts (~$6; tomorrow's Gemini budget).
4. Time-aware query expansion (LongMemEval paper: +7–11% temporal recall): extract the time
   window a question refers to and boost units from sessions inside it. Selection-only, runs from
   the score cache on the server.
6. Multi-view pooling (JustMem, arXiv 2609.19877: +5.4 LongMemEval, +2.0 LoCoMo, mostly
   aggregation questions): 1–2 answer-free rewrites of the question that keep its entities and
   time constraints, scored like our decompositions (max-pooled). Combine with fusion if it helps.
   JustMem reports 83.4 on LongMemEval-S with GPT-4.1-mini (full text 77.2) — a related-work
   comparison point for our 83.0 at 4k.
7. Provenance replay for assistant-output questions (JustMem: 32% → 95% on that subset once the
   source turn is found): when the question asks what the assistant said, add the full source
   turn of the best-scoring assistant sentence. Our weakest LongMemEval category at 4k (83.9).
5. Fact-augmented units: short LLM-extracted user facts per session added as extra candidate
   units (LongMemEval: +5 QA). Costs one Gemini pass over every session — check cost first.
