#!/usr/bin/env bash
# CooperBench with the CPC-Mem compressor between the agents (integrations/cooperbench).
# Everything runs in Docker on the GPU machine:
#   - the compressor service in the existing longmemeval-eval image, on one GPU;
#   - CooperBench's runner in python:3.12-slim, driving the host's Docker
#     (task containers, its own Redis) through the mounted socket.
#
#   scripts/cooperbench.sh setup                  # clone CooperBench (pinned), venv, pull flash task images
#   scripts/cooperbench.sh server                 # start the compressor service (GPU $GPU, port 8765)
#   scripts/cooperbench.sh run <arm> <name> [cooperbench run args...]
#   scripts/cooperbench.sh eval <name> [cooperbench eval args...]
#   scripts/cooperbench.sh report <baseline> <run>...
#   scripts/cooperbench.sh stop                   # remove this experiment's service and Redis containers
#
# LLM access, kept local to this experiment: Google credentials (a gcloud
# application-default credentials file or a service-account key) at $GCP_CREDENTIALS
# (default .gcp_credentials.json in the repo root, git-ignored, mode 600) are
# mounted read-only into the runner container only -- nothing global (gcloud
# config, shell profile, other users' env or files) is read or changed. Without
# a key, the agents use an OpenAI-compatible endpoint at $LLM_BASE instead.
# Model prices come from integrations/cooperbench/pricing.json, so costs are
# tracked and CooperBench's per-agent cost cap ($3, coop.yaml) holds.
#
# Shared machine: this experiment uses its own Redis (127.0.0.1:$REDIS_PORT, not
# the default 6379), GPU $GPU only, few concurrent pairs, and task containers
# with CPU/memory limits and a low CPU share ($TASK_LIMITS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CB_COMMIT="${CB_COMMIT:-63b9d44}"
CB_DIR="$ROOT/third_party/CooperBench"
VENV="$ROOT/third_party/cb-venv"
GPU="${GPU:-1}"
GCP_CREDENTIALS="${GCP_CREDENTIALS:-$ROOT/.gcp_credentials.json}"
VERTEX_PROJECT="${VERTEX_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
LLM_BASE="${LLM_BASE:-http://127.0.0.1:14873}"
LLM_KEY_FILE="${LLM_KEY_FILE:-$ROOT/.llm_proxy_key}"
CONCURRENCY="${CONCURRENCY:-3}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-3}"
REDIS_PORT="${REDIS_PORT:-16379}"
# logs/ is written by the root-run GPU containers, so CooperBench logs live apart, owned by the caller.
LOG_ROOT="${LOG_ROOT:-$ROOT/logs_cooperbench}"
# Every container this experiment starts carries the label cpc-cb=1, so it can be found
# and stopped without touching anyone else's containers.
export CB_TASK_LIMITS="${TASK_LIMITS:---cpus=2 --memory=8g --cpu-shares=256 --label=cpc-cb=1}"
DOCKER_BIN="$(command -v docker)"
DOCKER_GID="$(stat -c %g /var/run/docker.sock)"

if [[ -f "$GCP_CREDENTIALS" ]]; then
  MODEL="${MODEL:-vertex_ai/gemini-3-flash-preview}"
else
  MODEL="${MODEL:-openai/gemini-3-flash-preview}"
fi

runner() {  # run a command in the CooperBench runner container, as the calling user
  mkdir -p "$LOG_ROOT"
  local llm=()
  if [[ -f "$GCP_CREDENTIALS" ]]; then
    [[ -n "$VERTEX_PROJECT" ]] || { echo "set VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT)" >&2; exit 1; }
    llm=(-v "$GCP_CREDENTIALS:/run/secrets/gcp_credentials.json:ro"
         -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp_credentials.json
         -e VERTEXAI_PROJECT="$VERTEX_PROJECT" -e VERTEXAI_LOCATION="$VERTEX_LOCATION")
  elif [[ -f "$LLM_KEY_FILE" ]]; then
    llm=(-e OPENAI_API_BASE="$LLM_BASE" -e OPENAI_API_KEY="$(cat "$LLM_KEY_FILE")")
  fi
  docker run --rm --network host --label cpc-cb=1 \
    --user "$(id -u):$(id -g)" --group-add "$DOCKER_GID" \
    -v /var/run/docker.sock:/var/run/docker.sock -v "$DOCKER_BIN:/usr/bin/docker:ro" \
    -v "$ROOT:$ROOT" -w "$CB_DIR" \
    -e HOME="$ROOT/third_party/home" -e PYTHONPATH="$ROOT" -e CB_TASK_LIMITS \
    -e LITELLM_MODEL_REGISTRY_PATH="$ROOT/integrations/cooperbench/pricing.json" \
    "${llm[@]}" python:3.12-slim "$@"
}

ensure_redis() {  # this experiment's own Redis, never the default port another user may rely on
  if [[ -z "$(docker ps -q -f name=^cpc-cb-redis$)" ]]; then
    docker rm -f cpc-cb-redis >/dev/null 2>&1 || true
    docker run -d --name cpc-cb-redis --label cpc-cb=1 -p "127.0.0.1:$REDIS_PORT:6379" redis:alpine >/dev/null
    sleep 2
  fi
}

case "${1:-}" in
  setup)
    mkdir -p "$ROOT/third_party/home"
    if [[ ! -d "$CB_DIR/.git" ]]; then
      git clone https://github.com/cooperbench/CooperBench "$CB_DIR"
    fi
    git -C "$CB_DIR" fetch -q origin && git -C "$CB_DIR" checkout -q "$CB_COMMIT"
    # pip does not read [tool.uv.sources], so the vendored OpenHands packages are installed by path first.
    oh="$CB_DIR/src/cooperbench/agents/openhands_agent_sdk"
    runner bash -c "python -m venv '$VENV' && '$VENV/bin/pip' install -q --upgrade pip && \
      '$VENV/bin/pip' install -q -e '$oh/openhands-sdk' -e '$oh/openhands-tools' -e '$oh/openhands-workspace' && \
      '$VENV/bin/pip' install -q -e '$CB_DIR'"
    # Task images of the flash subset (pulled once; docker run would pull them lazily otherwise).
    runner "$VENV/bin/python" -c "
import json
from cooperbench.utils import get_image_name
for task in json.load(open('dataset/subsets/flash.json'))['tasks']:
    print(get_image_name(task['repo'], task['task_id']))" | sort -u | while read -r image; do
      docker pull -q "$image"
    done
    docker pull -q redis:alpine
    ;;
  server)
    docker rm -f cpc-compress >/dev/null 2>&1 || true
    docker run -d --name cpc-compress --label cpc-cb=1 --gpus "device=$GPU" -p 127.0.0.1:8765:8765 \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e HF_TOKEN \
      --entrypoint python longmemeval-eval -m integrations.cooperbench.compress_server --host 0.0.0.0 --port 8765
    for _ in $(seq 1 120); do
      curl -sf http://127.0.0.1:8765/health >/dev/null && { echo "compressor ready"; exit 0; }
      sleep 5
    done
    echo "compressor did not come up:"; docker logs --tail 30 cpc-compress; exit 1
    ;;
  run)
    arm="$2"; name="$3"; shift 3
    ensure_redis
    runner "$VENV/bin/python" -m integrations.cooperbench.launch run -n "$name" -m "$MODEL" \
      --setting coop --backend docker -c "$CONCURRENCY" --eval-concurrency "$EVAL_CONCURRENCY" \
      --redis "redis://localhost:$REDIS_PORT" \
      --agent-config "$ROOT/integrations/cooperbench/arms/$arm.yaml" --log-dir "$LOG_ROOT" "$@"
    ;;
  eval)
    name="$2"; shift 2
    runner "$VENV/bin/python" -m integrations.cooperbench.launch eval -n "$name" \
      --backend docker --concurrency "$EVAL_CONCURRENCY" --log-dir "$LOG_ROOT" "$@"
    ;;
  report)
    shift
    runner "$VENV/bin/python" -m integrations.cooperbench.report --logs "$LOG_ROOT" "$@"
    ;;
  stop)
    docker rm -f cpc-compress cpc-cb-redis 2>/dev/null || true
    ;;
  *)
    sed -n '2,25p' "$0"; exit 1
    ;;
esac
