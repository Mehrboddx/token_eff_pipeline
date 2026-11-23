#!/usr/bin/env bash
# Agent context management on SWE-bench Verified-mini (integrations/swebench_ctx):
# the policies of "The Complexity Trap" (Lindenbauer et al., 2025) in their own
# SWE-agent fork, and CPC recall (masking + query-conditioned recall of masked
# outputs). Runs on the GPU machine, in Docker:
#   - the compressor service: the CooperBench one (scripts/cooperbench.sh server), GPU 1;
#   - SWE-agent and the SWE-bench harness in python:3.12-slim, driving the host's
#     Docker (task containers) through the mounted socket.
#
#   scripts/swebench_ctx.sh setup                    # fork (pinned), venv, the 50 task images
#   scripts/swebench_ctx.sh run <policy> <name> [sweagent args]   # policy: raw | mask | summary | recall
#   scripts/swebench_ctx.sh eval <name>              # SWE-bench harness on <name>/preds.json
#   scripts/swebench_ctx.sh report <name>...         # solve rate, cost, tokens; vs the paper
#   scripts/swebench_ctx.sh stop                     # remove this experiment's task containers
#
# Settings follow the paper's Gemini 2.5 Flash runs: temperature 0.8, no thinking,
# 250 calls and $1.50 per instance at most. LLM access as in scripts/cooperbench.sh
# (credentials at $GCP_CREDENTIALS, mounted read-only into the runner only).
# Shared machine: task containers get CPU/memory limits and the label cpc-cb=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORK_COMMIT="${FORK_COMMIT:-bf15b5fb7d279679035a007ac9a81084d6b9a89a}"
FORK="$ROOT/third_party/the-complexity-trap"
VENV="$ROOT/third_party/swe-venv"
OUT="${OUT:-$ROOT/logs_swebench}"
IDS="$ROOT/integrations/swebench_ctx/verified_mini.txt"
GCP_CREDENTIALS="${GCP_CREDENTIALS:-$ROOT/.gcp_credentials.json}"
VERTEX_PROJECT="${VERTEX_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
MODEL="${MODEL:-vertex_ai/gemini-2.5-flash}"
WORKERS="${WORKERS:-4}"
RECALL_BUDGET="${RECALL_BUDGET:-2000}"
TASK_LIMITS='["--cpus=2", "--memory=8g", "--cpu-shares=256", "--label=cpc-cb=1"]'
DOCKER_BIN="$(command -v docker)"
DOCKER_GID="$(stat -c %g /var/run/docker.sock)"

runner() {  # a command in python:3.12-slim, as the calling user, driving the host's Docker
  mkdir -p "$OUT" "$ROOT/third_party/home"
  local llm=()
  if [[ -f "$GCP_CREDENTIALS" ]]; then
    [[ -n "$VERTEX_PROJECT" ]] || { echo "set VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT)" >&2; exit 1; }
    llm=(-v "$GCP_CREDENTIALS:/run/secrets/gcp_credentials.json:ro"
         -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp_credentials.json
         -e VERTEXAI_PROJECT="$VERTEX_PROJECT" -e VERTEXAI_LOCATION="$VERTEX_LOCATION")
  fi
  docker run --rm --network host --label cpc-cb=1 \
    --user "$(id -u):$(id -g)" --group-add "$DOCKER_GID" \
    -v /var/run/docker.sock:/var/run/docker.sock -v "$DOCKER_BIN:/usr/bin/docker:ro" \
    -v "$ROOT:$ROOT" -w "$ROOT" -e HOME="$ROOT/third_party/home" -e PYTHONPATH="$ROOT" -e PYTHONUNBUFFERED=1 \
    "${CPC_ENV[@]}" "${llm[@]}" python:3.12-slim "$@"
}
CPC_ENV=()

config_for() {  # policy -> the fork's config file (recall: the masking config with the processor swapped)
  case "$1" in
    raw) echo "$FORK/config/default_no_demo_raw.yaml" ;;
    mask) echo "$FORK/config/default_no_demo_N=1_M=10.yaml" ;;
    summary) echo "$FORK/config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml" ;;
    recall)
      local out="$OUT/$2/recall.yaml"
      mkdir -p "$OUT/$2"
      runner "$VENV/bin/python" -c "
import yaml
config = yaml.safe_load(open('$FORK/config/default_no_demo_N=1_M=10.yaml'))
config['agent']['history_processors'] = [{'type': 'cpc_recall', 'n': 10, 'budget': $RECALL_BUDGET}]
yaml.safe_dump(config, open('$out', 'w'), sort_keys=False, width=1000)"
      echo "$out" ;;
    *) echo "unknown policy $1" >&2; exit 1 ;;
  esac
}

case "${1:-}" in
  setup)
    if [[ ! -d "$FORK/.git" ]]; then
      git clone -q https://github.com/JetBrains-Research/the-complexity-trap "$FORK"
    fi
    git -C "$FORK" fetch -q origin && git -C "$FORK" checkout -q "$FORK_COMMIT"
    # grazie_api_gateway_client is JetBrains' internal LLM gateway; if it does not install,
    # the fork goes in without it (the runs here call Vertex through LiteLLM).
    runner bash -c "python -m venv '$VENV' && '$VENV/bin/pip' install -q --upgrade pip && \
      ( '$VENV/bin/pip' install -q -e '$FORK' || ( sed -i '/grazie_api_gateway_client/d' '$FORK/pyproject.toml' && \
        '$VENV/bin/pip' install -q -e '$FORK' ) ) && '$VENV/bin/pip' install -q swebench google-auth && \
      '$VENV/bin/pip' freeze | grep -i -E '^(litellm|swe-rex|swebench|sweagent)'"
    while read -r id; do
      docker pull -q "swebench/sweb.eval.x86_64.$(echo "${id//__/_1776_}" | tr 'A-Z' 'a-z'):latest" >/dev/null
    done < "$IDS"
    echo "task images: $(docker images -q 'swebench/sweb.eval.x86_64.*' | wc -l)"
    ;;
  run)
    policy="$2"; name="$3"; shift 3
    config="$(config_for "$policy" "$name")"
    filter="^($(paste -sd '|' "$IDS"))\$"
    mkdir -p "$OUT/$name"
    CPC_ENV=(-e CPC_RECALL_LOG="$OUT/$name/recall_calls.jsonl")
    runner "$VENV/bin/python" -m integrations.swebench_ctx.launch --config "$config" \
      --agent.model.name "$MODEL" --agent.model.temperature 0.8 \
      --agent.model.per_instance_cost_limit 1.50 --agent.model.per_instance_call_limit 250 \
      --agent.model.completion_kwargs '{"reasoning_effort": "disable"}' \
      --instances.type swe_bench --instances.subset verified --instances.split test \
      --instances.filter "$filter" --instances.deployment.docker_args "$TASK_LIMITS" \
      --num_workers "$WORKERS" --output_dir "$OUT/$name" "$@"
    ;;
  eval)
    name="$2"
    runner bash -c "cd '$OUT/$name' && '$VENV/bin/python' -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path preds.json --run_id '$name' \
      --max_workers $WORKERS --cache_level instance --report_dir ."
    ;;
  report)
    shift
    runner "$VENV/bin/python" -m integrations.swebench_ctx.report --out "$OUT" "$@"
    ;;
  stop)
    # only task containers of this experiment: labelled cpc-cb=1 and started from a SWE-bench image
    docker ps --filter label=cpc-cb=1 --format '{{.ID}} {{.Image}}' | awk '$2 ~ /sweb\.eval/ {print $1}' | xargs -r docker rm -f
    docker ps --filter label=cpc-cb=1 --format '{{.Names}} {{.Image}}'
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1
    ;;
esac
