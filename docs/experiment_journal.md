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

## 2026-11-17 · Chain-of-Note reader (running)

Idea (LongMemEval paper, Wu et al. 2025: up to +10 points): the answering model first lists
the relevant context with session dates, resolves relative times, reasons, then answers on a
final `Answer:` line; only that line is judged. No GPU — reuses the 4k contexts of runs 1 and 21.
Code: `--stage answer --reader notes` (branch `exp/adaptive-gate-and-notes`).

## Idea queue

1. Chain-of-Note reader — running.
2. Learned budget gate on LoCoMo with 2k/4k/8k/full — waiting on answers.
3. Answer-noise control: re-answer the 4k contexts once more (plain reader) to measure how much
   two identical runs disagree; needed to interpret every ±2-point difference above.
4. Time-aware query expansion (LongMemEval paper: +7–11% temporal recall): extract the time
   window a question refers to and boost units from sessions inside it. Selection-only, runs from
   the score cache on the server.
5. Fact-augmented units: short LLM-extracted user facts per session added as extra candidate
   units (LongMemEval: +5 QA). Costs one Gemini pass over every session — check cost first.
