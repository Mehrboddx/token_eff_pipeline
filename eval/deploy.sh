#!/usr/bin/env bash
#
# End-to-end setup + build + push + submit for eval/longmemeval.py on
# Vertex AI. Only --compressor cpc needs a GPU worker (a full 7B Mistral
# needs far more VRAM than a typical laptop has); lexical/gemini run on
# plain CPU, and are a good way to validate the whole pipeline (build,
# push, job submission, GCS log mirroring, --resume) before spending GPU
# time on Mistral. Mirrors replicate/deploy.sh's structure; the main
# difference is the build context has to be the project root, not this
# directory, since eval/longmemeval.py imports core/ and prompts/ too.
#
# Usage:
#   1. Edit the CONFIG block below (or export the same vars before running).
#   2. ./eval/deploy.sh          (run from anywhere -- it cd's to the repo root itself)
#
set -euo pipefail

# cd to the project root regardless of where this is invoked from, since
# the Docker build context has to be the root (Dockerfile.eval lives there).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ----------------------------- CONFIG ---------------------------------
PROJECT_ID="${PROJECT_ID:-cs-poc-pjtbxc0qllmcraowi6w1wmi}"
REGION="${REGION:-us-central1}"           # needs GPU quota for the chosen accelerator
REPO="${REPO:-cpc-longmemeval}"           # Artifact Registry repo name
BUCKET="${BUCKET:-gs://replicate}"        # created below if it doesn't exist
IMAGE_NAME="${IMAGE_NAME:-longmemeval-eval}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Only --compressor cpc needs a GPU at all -- lexical/gemini run on plain
# CPU. Defaults below pick a GPU worker only when COMPRESSOR=cpc; override
# any of these directly if that heuristic guesses wrong for your case.
# Mistral-7B in bf16 is ~14GB of weights alone -- default to an L4 (24GB)
# rather than a T4 (16GB) for headroom; a T4 is plenty for --cpc-preset llama (1B).
COMPRESSOR="${COMPRESSOR:-cpc}"
if [[ "$COMPRESSOR" == "cpc" ]]; then
  MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
  ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_L4}"
  ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
else
  MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
  ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-}"
  ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-}"
fi

# ---- eval/longmemeval.py args ----
CPC_PRESET="${CPC_PRESET:-mistral}"       # the whole reason to run this on Vertex instead of locally
CPC_MAX_SEQ_LENGTH="${CPC_MAX_SEQ_LENGTH:-}"  # blank = script default (1536)
LIMIT="${LIMIT:-500}"
SEED="${SEED:-42}"
QUESTION_TYPES="${QUESTION_TYPES:-}"      # e.g. "multi-session,knowledge-update", blank = no filter
TOKEN_BUDGET="${TOKEN_BUDGET:-2000}"
SENTENCE_THRESHOLD="${SENTENCE_THRESHOLD:-20}"
RECENT_TURNS="${RECENT_TURNS:-4}"
RESUME_RUN="${RESUME_RUN:-}"              # e.g. 12, to continue a prior (interrupted) job's run

JOB_DISPLAY_NAME="${JOB_DISPLAY_NAME:-longmemeval-eval}"
LOG_GCS_PREFIX="${LOG_GCS_PREFIX:-longmemeval_eval_logs}"

BUILD_METHOD="${BUILD_METHOD:-cloud}"     # "cloud" (gcloud builds submit) or "local" (docker)
# ------------------------------------------------------------------------

if [[ "$PROJECT_ID" == "your-gcp-project" ]]; then
  echo "Edit the CONFIG block in this script (or export PROJECT_ID/BUCKET/etc.) before running." >&2
  exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
LOG_GCS_URI="${BUCKET}/${LOG_GCS_PREFIX}"

echo "== Project:  $PROJECT_ID"
echo "== Region:   $REGION"
echo "== Image:    $IMAGE"
echo "== Log URI:  $LOG_GCS_URI"
echo "== Compressor: $COMPRESSOR (preset: $CPC_PRESET)"
echo

gcloud config set project "$PROJECT_ID" >/dev/null

echo "-- Enabling required APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com >/dev/null

echo "-- Ensuring Artifact Registry repo exists..."
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION"
else
  echo "   repo '$REPO' already exists, skipping."
fi

echo "-- Ensuring log bucket exists..."
BUCKET_NAME="${BUCKET#gs://}"
if ! gsutil ls -b "$BUCKET" >/dev/null 2>&1; then
  gsutil mb -l "$REGION" "$BUCKET"
else
  echo "   bucket '$BUCKET' already exists, skipping."
fi

echo "-- Building and pushing container image ($BUILD_METHOD)..."
if [[ "$BUILD_METHOD" == "cloud" ]]; then
  # gcloud builds submit's --tag shortcut only works with a file literally
  # named "Dockerfile" at the context root -- ours is Dockerfile.eval (to
  # not collide with replicate/dockerfile, a different image entirely), so
  # submit a minimal inline config instead of relying on the shortcut.
  gcloud builds submit . --config=- <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', 'Dockerfile.eval', '-t', '${IMAGE}', '.']
images: ['${IMAGE}']
EOF
else
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
  docker build -f Dockerfile.eval -t "$IMAGE" .
  docker push "$IMAGE"
fi

echo "-- Submitting Vertex AI Custom Job..."
ARGS="--compressor=${COMPRESSOR},--limit=${LIMIT},--seed=${SEED},--token-budget=${TOKEN_BUDGET}"
ARGS="${ARGS},--sentence-threshold=${SENTENCE_THRESHOLD},--recent-turns=${RECENT_TURNS}"
ARGS="${ARGS},--project=${PROJECT_ID},--location=${REGION},--log-gcs-uri=${LOG_GCS_URI}"
if [[ "$COMPRESSOR" == "cpc" ]]; then
  ARGS="${ARGS},--cpc-preset=${CPC_PRESET}"
  if [[ -n "$CPC_MAX_SEQ_LENGTH" ]]; then
    ARGS="${ARGS},--cpc-max-seq-length=${CPC_MAX_SEQ_LENGTH}"
  fi
fi
if [[ -n "$QUESTION_TYPES" ]]; then
  ARGS="${ARGS},--question-types=${QUESTION_TYPES}"
fi
if [[ -n "$RESUME_RUN" ]]; then
  ARGS="${ARGS},--resume=${RESUME_RUN}"
fi

WORKER_POOL_SPEC="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE}"
if [[ -n "$ACCELERATOR_TYPE" ]]; then
  WORKER_POOL_SPEC="${WORKER_POOL_SPEC},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT}"
fi

gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="$JOB_DISPLAY_NAME" \
  --worker-pool-spec="$WORKER_POOL_SPEC" \
  --args="$ARGS"

echo
echo "Job submitted. Useful follow-ups:"
echo "  gcloud ai custom-jobs list --region=$REGION"
echo "  gcloud ai custom-jobs stream-logs JOB_ID --region=$REGION"
echo "  gsutil ls ${LOG_GCS_URI}/"
echo
echo "If the job gets preempted or you Ctrl+C it, find the run number it printed"
echo "(or check ${LOG_GCS_URI}/), then re-run this script with RESUME_RUN=<n> to continue it."
