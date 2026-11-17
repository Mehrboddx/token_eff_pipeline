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

## 2026-11-17 · Adaptive budget: LoCoMo with the full ladder

LoCoMo fixed budgets (judge): 2k 73.5 · 4k 78.6 · **8k 80.9** · full 81.6 (29k tokens).
**CPC-Mem at 8k is level with full history** (93 better / 103 worse, n.s.) with 28% of its tokens.

With the 8k rung in place, neither gate beats mixing fixed budgets any more: learned gate at
θ 0.95 = 81.2 at 18.1k tokens, exactly the mix; θ 0.85 = 77.2 at 4.0k (−1.3 vs the mix, p 0.11).
The earlier +1.4 on the 4k/full ladder came from the 4k→full gap that a fixed 8k now fills.

**Conclusion for the paper:** gates built on CPC's score profile do not beat fixed budgets on
either benchmark (honest negative result). What carries is the budget curve: CPC-Mem matches
full history at **1k on LongMemEval** and at **8k on LoCoMo**.

## 2026-11-17 · Chain-of-Note reader

Idea (LongMemEval paper, Wu et al. 2025: up to +10 points): the answering model first lists
the relevant context with session dates, resolves relative times, reasons, then answers on a
final `Answer:` line; only that line is judged. No GPU — reuses the 4k contexts of runs 1 and 21.
Code: `--stage answer --reader notes` (branch `exp/adaptive-gate-and-notes`).

**LongMemEval: 83.8 vs 83.0** (25 better / 21 worse, McNemar p 0.66) — no real gain.
Multi-session 79.7 → 83.5, knowledge-update 87.2 → 89.7, but single-session-assistant
83.9 → 78.6. Gemini 2.5 Flash already thinks before answering, which likely covers most of what
explicit notes add.

**LoCoMo, first 559 questions** (the run was interrupted there): **80.0 vs 78.5** (39 / 31,
p 0.40). Same pattern as LongMemEval: it helps the reasoning-heavy categories — multi-hop
55.0 → 63.1, temporal 79.2 → 86.2 — and slightly hurts simple lookups (single-hop 87.4 → 85.0).
A per-question-type reader (notes only for multi-hop/temporal/multi-session questions) is the
natural follow-up. Remaining ~1,000 LoCoMo questions: tomorrow's budget.

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

**LoCoMo: escalation works.** FOUND answers 79–84% correct, MISSING 24–35%. Policies, cost =
tokens of every budget tried, compared with the best mix of fixed budgets:
- **8k → full: 81.9 at 9.7k tokens** — level with full history (81.6 at 29.0k) for a third of its
  tokens; +1.0 vs the mix (bootstrap p < 0.001). Only 6% of questions escalate.
- **4k → full: 80.7 at 5.7k tokens** — fixed 8k's accuracy (80.9) at 70% of its tokens; +1.1 vs
  the mix (p 0.015).
- from 2k: no better than the mix.

Caveats: the classifier is an extra Gemini call; in a deployed agent the answering model would
flag missing memory itself (next test: an answer prompt that may reply NOT FOUND). Bootstrap
holds the mix weight fixed. Answer noise (±3/run) affects absolute numbers, but every policy
here is evaluated on the same answer runs.

The underlying finding matters more: at 4k, 97.5% of LongMemEval questions have their evidence
in the context, yet accuracy is 83%. **The reader is the bottleneck, not the memory.**

## 2026-11-17 · Stronger readers on compressed memory (running)

Idea: compression makes a much stronger answer model affordable. Gemini 3.1 Pro on a 4k
context costs about $0.02 per question (measured tokens: ~7k prompt incl. system prompt,
~0.6k output+thinking), below Gemini 2.5 Flash on the full history ($0.033). Running on the
LongMemEval 4k contexts: `gemini-3.6-flash` and `gemini-3.1-pro-preview` (judge unchanged:
Gemini 2.5 Flash, temperature 0). `--answer-model`; answer rows now log token usage.

**Gemini 3.6 Flash on the same 4k contexts: 87.2 vs 83.0** (38 better / 17 worse, McNemar
p 0.006) — the first change clearly outside the ±3 answer noise. Per type: multi-session
79.7 → 87.2, single-session-user 88.6 → 94.3, temporal 87.2 → 91.0, preference 53.3 → 60.0;
only single-session-assistant drops (83.9 → 80.4). Measured per question: 6.4k prompt tokens,
0.73k thinking, 0.11k output. Gemini 2.5 Flash on the full history reaches 80.0 with 107k
prompt tokens per question; CPC-Mem at 4k + a stronger reader is +7 points with 6% of the tokens.

**Gemini 3.1 Pro on the same 4k contexts: 88.4 vs 83.0** (41 better / 14 worse, McNemar
p < 0.001); vs 3.6 Flash 88.4 vs 87.2 (21 / 15, p 0.41 — level). Preference 53.3 → **76.7**
(Pro tailors advice to the memory without being told), multi-session 79.7 → 86.5, knowledge-update
87.2 → 91.0, single-session-user 88.6 → 94.3, temporal 87.2 → 91.0; assistant-recall still flat
(83.9 → 82.1: the truncated lists, see below). Per question: 6.4k prompt, 0.61k thinking, 0.10k output.

| Reader on CPC-Mem 4k | LongMemEval | vs 2.5 Flash |
|---|---|---|
| Gemini 2.5 Flash | 83.0 | — |
| Gemini 3.6 Flash | 87.2 | +4.2 (p 0.006) |
| Gemini 3.1 Pro | **88.4** | +5.4 (p < 0.001) |
| (2.5 Flash, full history, 107k tokens) | 80.0 | |

## 2026-11-17 · CPC + Provence score fusion (GPU, running)

Idea: the two scorers are complementary — CPC wins on LongMemEval (83.0 vs 79.4), Provence on
LoCoMo (81.3 vs 78.6); Provence keeps answer-like assistant text, CPC the user's own statements.
Blend their per-query z-scores: `--fuse-provence W` (Provence scores cached like CPC's).
Smoke test on 5 questions: peak GPU memory 7.2 GB (CPC from cache, only Provence loaded).
Queued on the server, GPU 1: W = 0.5 and 0.25, both benchmarks.

Done (runs 51–54, all exit 0). Before spending on answers, evidence recall (share of gold
evidence turns kept in the 4k context; free, from the compress logs):

| | CPC | Provence | fuse 0.25 | fuse 0.5 |
|---|---|---|---|---|
| LongMemEval | 97.5 | 82.1 | **98.5** | 95.9 |
| LoCoMo | 91.7 | 95.5 | **95.5** | 95.9 |

W = 0.25 keeps (slightly beats) CPC on LongMemEval and reaches Provence on LoCoMo — the best of
both on this proxy. Answers for W = 0.25 next (Gemini budget).

Sweep (runs 55–62, all exit 0), evidence turns kept, %:

| W | 0 | 0.1 | 0.15 | 0.25 | 0.35 | 0.5 | Provence |
|---|---|---|---|---|---|---|---|
| LongMemEval 4k | 97.5 | **98.7** | 98.5 | 98.5 | 98.3 | 95.9 | 82.1 |
| LoCoMo 4k | 91.7 | 94.3 | 94.8 | 95.5 | **96.0** | 95.9 | 95.5 |

Broad plateau over 0.15–0.35; W = 0.25 chosen (not tuned on accuracy). It also helps at the
other budgets: LongMemEval 1k 90.7 → **93.2**, LoCoMo 8k 96.8 → **98.1**.

Latency (eval/latency.py, 50 questions per benchmark, RTX 4090, same questions for both):

| | LME cold | LME incremental | LoCoMo cold | LoCoMo incremental | peak GPU |
|---|---|---|---|---|---|
| CPC | 23.2 s | 1.38 s | 5.0 s | 0.49 s | 14.6 GiB |
| CPC + Provence | 28.8 s | **6.49 s** | 6.4 s | **1.79 s** | 19.5 GiB |

Cold cost +24–28%; the per-call (incremental) cost is 4–5× higher, because Provence is a
cross-encoder: it reads question and history together, so nothing carries over between calls,
while CPC re-encodes only the sessions that changed. A real trade-off to state in the paper;
a cheaper variant would run Provence only on CPC's top few hundred candidates.

Server note: rebuilding `longmemeval-eval:latest` in place leaves the previous build as an
untagged ("dangling") image; not deleted — your call (`docker image prune` removes them).

## 2026-11-17 · Evidence that scores low

Questions answered wrong at 4k but right with full history, where a gold evidence turn was dropped
(no sentence of it in the context): LoCoMo 41 of 143 such questions (22 multi-hop), LongMemEval 6.
The dropped evidence falls into five patterns (examples from the logs):
1. **Aside inside an off-topic sentence** — "I've used Trello in my previous role *as a marketing
   specialist*…" (Q: previous occupation); "…BBQ sauce recipes. *By the way, I just got a smoker
   today*" (Q: kitchen appliance bought 10 days ago); "…playing my violin" (Q: instruments).
2. **Bridge sentence that shares no words with the question** — "my old area was hit by a flood"
   kept, "improving my old area, *West County*" dropped (Q: what area was hit by a flood).
3. **Answer only by inference** — "tough as a *single parent*" (Q: relationship status);
   "their *brother* would be OK" (Q: how many children).
4. **Every instance needed** — count/list questions: the budget fills with the clearest mention;
   "my daughter *Sara's* birthday" (Q: names of John's children) dropped.
5. **Date-pinned** — "investment for a competition four weeks ago": the session exactly four weeks
   earlier says "I got my own set of sculpting tools… today". Session dates are not scored.
Some LoCoMo evidence labels point at filler turns ("Oh? Are you going to invite your tournament
friends?"), so 41 overstates it.

Fixes, cheapest first: (A) date-aware focus `--time-focus`, (D) session coverage for count/list
questions `--session-cover` — both selection-only, sweep queued (runs 69–77); (B) second CPC
pass with question + top sentences as query (bridges); (C) split asides into their own units;
(E) LLM-extracted facts at storage time.

(A) check without scores: LoCoMo has 91 questions with an explicit date ("on 3 June, 2023"); the
window holds ~3% of sessions and all gold sessions for 76/89. They score 76.9 at 4k, 82.4 at 8k,
86.8 full. On LongMemEval, "last month" / "past two weeks" windows cover nearly all sessions
(haystacks span a few weeks), so (A) does little there.

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
8. Largest-gap budget (LooComp, arXiv 2603.09222): keep the units above the largest gap in the
   sorted scores. A close relative of our rule gate, which lost to fixed budgets — low priority.
9. Full history + Gemini 3.6 Flash on a ~100-question stratified subset (~$4): needed to say
   whether CPC-Mem + strong reader beats, or only undercuts in cost, a strong reader on full text.

## 2026-11-17 · Why the reader fails with the evidence in context

Failure reading, Gemini 3.6 Flash on 4k contexts (no model calls):
- **Assistant-recall (11 wrong / 56):** 10 ask for one item of a long list the assistant once gave
  ("the 7th job", "the 27th parameter", "the last venue"). Sentence selection keeps fragments of
  that turn; the reader even says the list "was truncated". Turn-level evidence recall (1/1) hides
  this. → **Provenance replay** (`--replay-turn`): for questions that ask what the assistant said,
  keep the turn of the best-scoring assistant sentence whole. Cue regex hits 54/56 assistant
  questions, 0 other LongMemEval questions, 0 LoCoMo questions (caveat: wording written after
  looking at these questions). Queued on the server (4k, 4k + fuse 0.25, 1k).
- **Preference (12 wrong / 30):** evidence kept every time, but the answer is generic advice that
  ignores what the memory says about the user. → **Memory-aware reader** (`--reader memory`):
  the system prompt says the earlier context is the agent's memory of this user and asks to build
  advice on it. Gemini-only test, after the temperature-0 runs.

**Memory-aware reader, result** (Gemini 2.5 Flash, temperature 0, run 1's 4k contexts):
**84.2** vs 82.0 / 82.6 for the two plain temperature-0 runs (overall n.s., p 0.14 / 0.30).
All of it is preference: **40.0 / 46.7 → 76.7** (12 fixed / 1 broken and 9 / 0, exact
p 0.003 / 0.004), level with Gemini 3.1 Pro's 76.7 without the prompt. The other 470 questions are
untouched (84.7 → 84.7, 17 / 17 and 18 / 19 flips). So Gemini 2.5 Flash reads the compressed
excerpt as background unless told it is the user's memory; one paragraph of system prompt closes
the preference gap to a much stronger model. Caveat: written after reading 12 preference failures
of the 3.6 Flash run (same 30 questions); the prompt is generic, but a held-out check (LoCoMo has no
preference category) is not possible here. Make it the default reader for everything below.

**Provenance replay, context check** (no model calls; share of the 56 assistant-recall questions
whose reference words all appear in the context — crude, references are often paraphrased):

| | plain | replay v1 (run 63/65) | replay v2 (run 66/68) |
|---|---|---|---|
| 4k | 39 / 56 (78.6% words) | 41 (84.2%) | **44 (88.6%)** |
| 1k | 34 / 56 (74.1%) | 36 (79.3%) | **40 (85.7%)** |

v1 (turn of the single best assistant sentence) often replayed the wrong turn: a generic list
("transit apps" for "jobs for seniors"), or the general five-bottle list instead of its gin-specific
follow-up. v2 ranks each assistant turn by its best sentence *or the user request just before it*,
and replays the top two. v2 + fusion 0.25 (run 67): 44 (89.1%). All other 444 questions get
exactly the same context as without replay. Both versions designed from these same failures —
to disclose. Answering: next Gemini budget (temperature 0, memory reader).

## 2026-11-17 · Temperature-0 answers

Run 1's 4k contexts answered twice at temperature 0 (tags a, b): **82.0 vs 82.6, 27 flips**
(15 / 12, p 0.70), against 52 flips for two runs at the default temperature 1.0. Only 266 of 500
answer texts are word-for-word identical — Gemini 2.5 Flash's thinking is not deterministic even at
temperature 0. Temperature 0 halves the noise but does not remove it: the run-to-run difference
has a standard deviation of about 1 point (√27 / 500), vs about 1.4 at temperature 1.0.
Temperature-0 accuracy (82.3 mean) is level with the default (83.0 / 80.2).

**Decision:** all further answer runs at temperature 0, compared against the mean of runs a and b;
an effect counts only if it beats both (or McNemar p < 0.05 against each). For the paper, the
main configurations should be re-answered at temperature 0 (≥ 2 runs each). The stronger-reader
gains above (+4.2, +5.4) are several times this noise.
