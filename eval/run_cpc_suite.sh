#!/usr/bin/env bash
#
# The whole CPC evaluation on one local GPU machine, in order:
#   1. bidirectional-attention sanity check (+ small causal vs bidirectional A/B)
#   2. selection ablation grid, bidirectional -- fills the score cache (GPU, once)
#   3. the same pipeline with causal attention, base config (the "before" point)
#   4. compress -> answer -> judge with the chosen selection flags (Gemini)
#   5. optional: re-run the no-compression and Gemini-compressor baselines,
#      which the calendar-day date fix changes too
#
# Each step's output is in logs/suite_<preset>_<budget>/. Steps can be
# skipped with SKIP="1 3" etc. Re-running is cheap: step 2 reuses the score cache.
#
# Usage:
#   export GOOGLE_CLOUD_PROJECT=<project>        # steps 4-5 call Gemini
#   bash eval/run_cpc_suite.sh
#   PRESET=llama BUDGET=2000 bash eval/run_cpc_suite.sh
#   SELECTION="--window 1 --pair-turns --mmr 1 --mmr-similarity semantic" SKIP="1 2 3" bash eval/run_cpc_suite.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PRESET="${PRESET:-mistral}"
BUDGET="${BUDGET:-500}"
LIMIT="${LIMIT:-500}"
SEQ="${SEQ:-6144}"
CACHE="${CACHE:-logs/score_cache}"
DECOMP="${DECOMP:-eval/data/longmemeval_s_decompositions.json}"
SELECTION="${SELECTION:-}"          # CLI selection flags for step 4, chosen from step 2's table
RUN_BASELINES="${RUN_BASELINES:-0}"
SKIP="${SKIP:-}"
OUT="logs/suite_${PRESET}_${BUDGET}"
mkdir -p "$OUT"

CPC=(--compressor cpc --cpc-preset "$PRESET" --cpc-max-seq-length "$SEQ" --token-budget "$BUDGET" --limit "$LIMIT")

skipped() { [[ " $SKIP " == *" $1 "* ]]; }
saved_log() { grep -oE "saved to .*\.jsonl" "$1" | tail -1 | sed 's/^saved to //'; }
need_project() {
  if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
    echo "Step $1 calls Gemini: export GOOGLE_CLOUD_PROJECT first (or SKIP it)." >&2
    exit 1
  fi
}

if ! skipped 1; then
  echo "== 1. Bidirectional sanity check ($PRESET)"
  python -m eval.check_bidirectional --cpc-preset "$PRESET" --cpc-max-seq-length "$SEQ" \
    --token-budget "$BUDGET" | tee "$OUT/1_check_bidirectional.txt"
fi

if ! skipped 2; then
  echo "== 2. Selection ablation grid (bidirectional)"
  python -m eval.ablate "${CPC[@]}" --score-cache "$CACHE" --decompositions "$DECOMP" \
    --out "$OUT/2_ablate_bidirectional.json" | tee "$OUT/2_ablate_bidirectional.txt"
fi

if ! skipped 3; then
  echo "== 3. Causal attention, base config (before the fix)"
  python -m eval.ablate "${CPC[@]}" --cpc-attention causal --score-cache "$CACHE" --configs "base=" \
    --out "$OUT/3_ablate_causal.json" | tee "$OUT/3_ablate_causal.txt"
fi

if ! skipped 4; then
  need_project 4
  echo "== 4. Compress -> answer -> judge with: ${SELECTION:-<base config>}"
  # shellcheck disable=SC2086
  python -m eval.longmemeval --stage compress "${CPC[@]}" --score-cache "$CACHE" --decompositions "$DECOMP" \
    $SELECTION | tee "$OUT/4a_compress.txt"
  COMPRESS_LOG="$(saved_log "$OUT/4a_compress.txt")"
  python -m eval.longmemeval --stage answer --from-log "$COMPRESS_LOG" --project "$GOOGLE_CLOUD_PROJECT" \
    | tee "$OUT/4b_answer.txt"
  ANSWER_LOG="$(saved_log "$OUT/4b_answer.txt")"
  python -m eval.longmemeval --stage judge --from-log "$ANSWER_LOG" --project "$GOOGLE_CLOUD_PROJECT" \
    | tee "$OUT/4c_judge.txt"
fi

if [[ "$RUN_BASELINES" == "1" ]] && ! skipped 5; then
  need_project 5
  for compressor in none gemini; do
    echo "== 5. Baseline re-run: $compressor"
    budget_flag=(--token-budget 2000)
    [[ "$compressor" == "none" ]] && budget_flag=()
    python -m eval.longmemeval --stage full --compressor "$compressor" "${budget_flag[@]}" --limit "$LIMIT" \
      --project "$GOOGLE_CLOUD_PROJECT" | tee "$OUT/5_${compressor}_full.txt"
    python -m eval.longmemeval --stage judge --from-log "$(saved_log "$OUT/5_${compressor}_full.txt")" \
      --project "$GOOGLE_CLOUD_PROJECT" | tee "$OUT/5_${compressor}_judge.txt"
  done
fi

echo
echo "Done. Outputs in $OUT/"
