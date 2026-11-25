#!/usr/bin/env bash
# AMA-Bench open-ended QA (arXiv 2602.22769) with CPC-Mem selecting what the
# reader sees from a long agent trajectory (integrations/amabench). Runs on the
# GPU machine, in Docker:
#   - the compressor service: the CooperBench one (GPU=1 scripts/cooperbench.sh server);
#   - everything else in python:3.12-slim, as the calling user (the HSC-Bench venv).
#
#   scripts/amabench.sh setup                  # adds rank-bm25 to third_party/hsc-venv
#   scripts/amabench.sh data [--per-domain N]  # -> $RUN_DIR/episodes.jsonl
#   scripts/amabench.sh contexts [args]        # needs the compressor service; -> contexts.jsonl
#   scripts/amabench.sh answer [args]          # -> answers.jsonl
#   scripts/amabench.sh judge [args]           # -> judged.jsonl
#   scripts/amabench.sh score [args]
#
# LLM access as in scripts/hscbench.sh (Google credentials at $GCP_CREDENTIALS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/third_party/hsc-venv"
RUN_DIR="${RUN_DIR:-$ROOT/logs_ama}"
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

R="$RUN_DIR"
case "${1:-}" in
  setup)
    runner "$VENV/bin/pip" install -q rank-bm25
    ;;
  data)
    shift; runner "$VENV/bin/python" -m integrations.amabench.data --raw "$R/open_end_qa_set.jsonl" --out "$R/episodes.jsonl" "$@"
    ;;
  contexts)
    shift; runner "$VENV/bin/python" -m integrations.amabench.contexts --episodes "$R/episodes.jsonl" --out "$R/contexts.jsonl" "$@"
    ;;
  answer)
    shift; runner "$VENV/bin/python" -m integrations.amabench.answer --episodes "$R/episodes.jsonl" \
      --contexts "$R/contexts.jsonl" --out "$R/answers.jsonl" "$@"
    ;;
  judge)
    shift; runner "$VENV/bin/python" -m integrations.amabench.judge --episodes "$R/episodes.jsonl" \
      --answers "$R/answers.jsonl" --out "$R/judged.jsonl" "$@"
    ;;
  score)
    shift; runner "$VENV/bin/python" -m integrations.amabench.score --episodes "$R/episodes.jsonl" \
      --contexts "$R/contexts.jsonl" --judged "$R/judged.jsonl" "$@"
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1
    ;;
esac
