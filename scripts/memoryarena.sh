#!/usr/bin/env bash
# MemoryArena formal reasoning (arXiv 2602.16313; math + physics) with CPC-Mem as the
# agent's memory between sessions (integrations/memoryarena). Runs on the GPU machine:
#   - the compressor service: the CooperBench one (GPU=1 scripts/cooperbench.sh server),
#     needed for the cpc and random conditions;
#   - everything else in python:3.12-slim, as the calling user (the HSC-Bench venv).
#
#   scripts/memoryarena.sh setup          # clone MemoryArena (pinned) and fetch the data
#   scripts/memoryarena.sh run [args]     # -> $RUN_DIR/results.jsonl, usage.jsonl
#   scripts/memoryarena.sh score [args]
#
# LLM access as in scripts/hscbench.sh (Google credentials at $GCP_CREDENTIALS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/third_party/hsc-venv"
MA_COMMIT="${MA_COMMIT:-6cd9de14b71915e39ac742a20dc33785e14b6aab}"
MA_DIR="$ROOT/third_party/MemoryArena"
RUN_DIR="${RUN_DIR:-$ROOT/logs_marena}"
GCP_CREDENTIALS="${GCP_CREDENTIALS:-$ROOT/.gcp_credentials.json}"
VERTEX_PROJECT="${VERTEX_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
HF="https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main"

runner() {
  mkdir -p "$RUN_DIR" "$ROOT/third_party/hsc-cache"
  local llm=()
  if [[ -f "$GCP_CREDENTIALS" ]]; then
    [[ -n "$VERTEX_PROJECT" ]] || { echo "set VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT)" >&2; exit 1; }
    llm=(-v "$GCP_CREDENTIALS:/run/secrets/gcp_credentials.json:ro"
         -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp_credentials.json
         -e VERTEXAI_PROJECT="$VERTEX_PROJECT" -e VERTEXAI_LOCATION="$VERTEX_LOCATION")
  fi
  docker run --rm --network host --label cpc-cb=1 --user "$(id -u):$(id -g)" \
    -v "$ROOT:$ROOT" -w "$ROOT" -e HOME="$ROOT/third_party/home" -e PYTHONPATH="$ROOT" \
    -e TIKTOKEN_CACHE_DIR="$ROOT/third_party/hsc-cache/tiktoken" -e PYTHONUNBUFFERED=1 \
    -e MEMORYARENA_REPO="$MA_DIR" "${llm[@]}" python:3.12-slim "$@"
}

R="$RUN_DIR"
case "${1:-}" in
  setup)
    if [[ ! -d "$MA_DIR/.git" ]]; then
      git clone -q https://github.com/ZexueHe/MemoryArena.git "$MA_DIR"
    fi
    git -C "$MA_DIR" fetch -q origin && git -C "$MA_DIR" checkout -q "$MA_COMMIT"
    mkdir -p "$R/data"
    for config in formal_reasoning_math formal_reasoning_phys; do
      curl -sfL "$HF/$config/data.jsonl" -o "$R/data/$config.jsonl"
    done
    runner "$VENV/bin/pip" install -q rank-bm25 openai
    wc -l "$R"/data/*.jsonl
    ;;
  run)
    shift; runner "$VENV/bin/python" -m integrations.memoryarena.run_formal --data-dir "$R/data" \
      --out "$R/results.jsonl" --usage "$R/usage.jsonl" "$@"
    ;;
  score)
    shift; runner "$VENV/bin/python" -m integrations.memoryarena.score --results "$R/results.jsonl" \
      --usage "$R/usage.jsonl" "$@"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
