#!/usr/bin/env bash
# CooperBench with the CPC-Mem compressor between the agents (integrations/cooperbench).
# Everything runs in Docker on the GPU machine:
#   - the compressor service in the existing longmemeval-eval image, on one GPU;
#   - CooperBench's runner in python:3.12-slim, driving the host's Docker
#     (task containers, Redis) through the mounted socket.
#
#   scripts/cooperbench.sh setup                  # clone CooperBench (pinned), venv, pull flash task images
#   scripts/cooperbench.sh server                 # start the compressor service (GPU $GPU, port 8765)
#   scripts/cooperbench.sh run <arm> <name> [cooperbench run args...]
#   scripts/cooperbench.sh eval <name> [cooperbench eval args...]
#   scripts/cooperbench.sh report <baseline> <run>...
#
# The agents' LLM is reached at $LLM_BASE (OpenAI-compatible: a LiteLLM proxy
# in front of Vertex AI), with the key read from $LLM_KEY_FILE. The proxied model
# name is priced from integrations/cooperbench/pricing.json (Vertex list prices),
# so costs are tracked and CooperBench's per-agent cost cap ($3, coop.yaml) holds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CB_COMMIT="${CB_COMMIT:-63b9d44}"
CB_DIR="$ROOT/third_party/CooperBench"
VENV="$ROOT/third_party/cb-venv"
GPU="${GPU:-1}"
MODEL="${MODEL:-openai/gemini-3-flash-preview}"
LLM_BASE="${LLM_BASE:-http://127.0.0.1:14873}"
LLM_KEY_FILE="${LLM_KEY_FILE:-$ROOT/.llm_proxy_key}"
CONCURRENCY="${CONCURRENCY:-4}"
DOCKER_BIN="$(command -v docker)"
DOCKER_GID="$(stat -c %g /var/run/docker.sock)"

runner() {  # run a command in the CooperBench runner container, as the calling user
  local key=""
  [[ -f "$LLM_KEY_FILE" ]] && key="$(cat "$LLM_KEY_FILE")"
  docker run --rm --network host \
    --user "$(id -u):$(id -g)" --group-add "$DOCKER_GID" \
    -v /var/run/docker.sock:/var/run/docker.sock -v "$DOCKER_BIN:/usr/bin/docker:ro" \
    -v "$ROOT:$ROOT" -w "$CB_DIR" \
    -e HOME="$ROOT/third_party/home" -e PYTHONPATH="$ROOT" \
    -e OPENAI_API_BASE="$LLM_BASE" -e OPENAI_API_KEY="$key" \
    -e LITELLM_MODEL_REGISTRY_PATH="$ROOT/integrations/cooperbench/pricing.json" \
    python:3.12-slim "$@"
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
    docker run -d --name cpc-compress --gpus "device=$GPU" -p 127.0.0.1:8765:8765 \
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
    runner "$VENV/bin/python" -m integrations.cooperbench.launch run -n "$name" -m "$MODEL" \
      --setting coop --backend docker -c "$CONCURRENCY" \
      --agent-config "$ROOT/integrations/cooperbench/arms/$arm.yaml" --log-dir "$ROOT/logs/cooperbench" "$@"
    ;;
  eval)
    name="$2"; shift 2
    runner "$VENV/bin/python" -m integrations.cooperbench.launch eval -n "$name" \
      --backend docker --log-dir "$ROOT/logs/cooperbench" "$@"
    ;;
  report)
    shift
    runner "$VENV/bin/python" -m integrations.cooperbench.report --logs "$ROOT/logs/cooperbench" "$@"
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1
    ;;
esac
