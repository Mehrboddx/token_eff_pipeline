#!/usr/bin/env bash
# HSC-Bench repository track (rebuilt from ContextBench + CodeTraceBench) with
# CPC-Mem as a handoff representation (integrations/hscbench). Runs on the GPU
# machine, in Docker:
#   - the compressor service: the CooperBench one (scripts/cooperbench.sh server), GPU $GPU;
#   - everything else in python:3.12-slim, as the calling user.
#
#   scripts/hscbench.sh setup                 # venv (third_party/hsc-venv)
#   scripts/hscbench.sh check <model>         # one tiny call, to check a model is reachable
#   scripts/hscbench.sh data [--limit N]      # -> $RUN_DIR/tasks.jsonl
#   scripts/hscbench.sh contexts [args]       # needs the compressor service; -> contexts.jsonl
#   scripts/hscbench.sh receive [args]        # -> predictions.jsonl
#   scripts/hscbench.sh score [args]
#
# LLM access as in scripts/cooperbench.sh: Google credentials at $GCP_CREDENTIALS
# (git-ignored, mode 600), mounted read-only into the container only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/third_party/hsc-venv"
RUN_DIR="${RUN_DIR:-$ROOT/logs_hsc}"
GCP_CREDENTIALS="${GCP_CREDENTIALS:-$ROOT/.gcp_credentials.json}"
VERTEX_PROJECT="${VERTEX_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
VERTEX_LOCATION="${VERTEX_LOCATION:-global}"

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
    "${llm[@]}" python:3.12-slim "$@"
}

case "${1:-}" in
  setup)
    mkdir -p "$ROOT/third_party/home"
    runner bash -c "python -m venv '$VENV' && '$VENV/bin/pip' install -q --upgrade pip && \
      '$VENV/bin/pip' install -q litellm google-auth tiktoken pandas pyarrow zstandard"
    ;;
  check)
    runner "$VENV/bin/python" -c "
from integrations.hscbench.llm import complete, count_tokens
r = complete('$2', [{'role': 'user', 'content': 'Reply with the word ok.'}], max_tokens=64, reasoning_effort='${3:-minimal}' or None, retries=1)
print(r); print('o200k test', count_tokens('hello world'))"
    ;;
  data)
    shift; runner "$VENV/bin/python" -m integrations.hscbench.data --out "$RUN_DIR/tasks.jsonl" "$@"
    ;;
  contexts)
    shift; runner "$VENV/bin/python" -m integrations.hscbench.handoff --tasks "$RUN_DIR/tasks.jsonl" --out "$RUN_DIR/contexts.jsonl" "$@"
    ;;
  receive)
    shift; runner "$VENV/bin/python" -m integrations.hscbench.receive --tasks "$RUN_DIR/tasks.jsonl" \
      --contexts "$RUN_DIR/contexts.jsonl" --out "$RUN_DIR/predictions.jsonl" "$@"
    ;;
  score)
    shift; runner "$VENV/bin/python" -m integrations.hscbench.score --tasks "$RUN_DIR/tasks.jsonl" \
      --contexts "$RUN_DIR/contexts.jsonl" --predictions "$RUN_DIR/predictions.jsonl" "$@"
    ;;
  *)
    sed -n '2,16p' "$0"; exit 1
    ;;
esac
