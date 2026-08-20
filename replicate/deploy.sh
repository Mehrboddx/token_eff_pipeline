#!/usr/bin/env bash
#
# End-to-end setup + build + push + submit for the CPC LongBench job on
# Vertex AI. Run from the same directory as Dockerfile / cpc_compressor.py /
# run_longbench.py / requirements.txt.
#
# Usage:
#   1. Edit the CONFIG block below (or export the same vars before running).
#   2. ./deploy.sh
#
set -euo pipefail

# ----------------------------- CONFIG ---------------------------------
PROJECT_ID="${PROJECT_ID:-cs-poc-pjtbxc0qllmcraowi6w1wmi}"
REGION="${REGION:-us-central1}"          # needs GPU quota for the chosen accelerator
REPO="${REPO:-cpc-longbench}"            # Artifact Registry repo name
BUCKET="${BUCKET:-gs://replicate}" # created below if it doesn't exist
IMAGE_NAME="${IMAGE_NAME:-cpc-longbench}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_TESLA_T4}"   # or NVIDIA_L4 / NVIDIA_TESLA_A100
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"

TARGET_TOKENS="${TARGET_TOKENS:-2000}"
LIMIT="${LIMIT:-}"                        # e.g. 20, blank = all examples per subset
SUBSETS="${SUBSETS:-}"                    # e.g. "narrativeqa,qasper", blank = all 21

JOB_DISPLAY_NAME="${JOB_DISPLAY_NAME:-cpc-longbench}"
OUTPUT_GCS_PREFIX="${OUTPUT_GCS_PREFIX:-cpc_longbench_output}"

BUILD_METHOD="${BUILD_METHOD:-cloud}"     # "cloud" (gcloud builds submit) or "local" (docker)
# ------------------------------------------------------------------------

if [[ "$PROJECT_ID" == "your-gcp-project" ]]; then
  echo "Edit the CONFIG block in this script (or export PROJECT_ID/BUCKET/etc.) before running." >&2
  exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "== Project:  $PROJECT_ID"
echo "== Region:   $REGION"
echo "== Image:    $IMAGE"
echo "== Bucket:   $BUCKET"
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

echo "-- Ensuring output bucket exists..."
BUCKET_NAME="${BUCKET#gs://}"
if ! gsutil ls -b "$BUCKET" >/dev/null 2>&1; then
  gsutil mb -l "$REGION" "$BUCKET"
else
  echo "   bucket '$BUCKET' already exists, skipping."
fi

echo "-- Building and pushing container image ($BUILD_METHOD)..."
if [[ "$BUILD_METHOD" == "cloud" ]]; then
  gcloud builds submit --tag "$IMAGE" .
else
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
  docker build -t "$IMAGE" .
  docker push "$IMAGE"
fi

echo "-- Submitting Vertex AI Custom Job..."
ARGS="--output_gcs_uri=${BUCKET}/${OUTPUT_GCS_PREFIX},--target_tokens=${TARGET_TOKENS}"
if [[ -n "$LIMIT" ]]; then
  ARGS="${ARGS},--limit=${LIMIT}"
fi
if [[ -n "$SUBSETS" ]]; then
  ARGS="${ARGS},--subsets=${SUBSETS}"
fi

gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="$JOB_DISPLAY_NAME" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},replica-count=1,accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE}" \
  --args="$ARGS"

echo
echo "Job submitted. Useful follow-ups:"
echo "  gcloud ai custom-jobs list --region=$REGION"
echo "  gcloud ai custom-jobs stream-logs JOB_ID --region=$REGION"
echo "  gsutil ls ${BUCKET}/${OUTPUT_GCS_PREFIX}/"