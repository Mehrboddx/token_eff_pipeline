# CPC on LongBench — Vertex AI Custom Job

Ported from the Colab notebook. Same `CPCCompressor` pipeline; the only
changes are Colab-specific bits (Drive mount, `files.download`) swapped for
GCS output, and config moved to CLI flags so it runs headless in a
container.

Files:
- `cpc_compressor.py` — the compression pipeline (unchanged from the notebook).
- `run_longbench.py` — the batch loop over all 21 LongBench subsets, writing
  `.txt` outputs to a GCS bucket (resumable: re-running skips files already
  uploaded).
- `Dockerfile` / `requirements.txt` — container definition.

## 1. One-time setup

```bash
PROJECT_ID=your-gcp-project
REGION=us-central1                       # pick a region with GPU quota
REPO=cpc-longbench                       # Artifact Registry repo name
BUCKET=gs://your-bucket-name             # must already exist (or: gsutil mb $BUCKET)

gcloud config set project $PROJECT_ID

gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com

gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION
```

## 2. Build and push the container

```bash
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/cpc-longbench:latest

gcloud auth configure-docker $REGION-docker.pkg.dev

docker build -t $IMAGE .
docker push $IMAGE
```

(No local GPU/Docker environment? Swap the two commands above for
`gcloud builds submit --tag $IMAGE .`, which builds in the cloud instead.)

## 3. Submit the job

A single T4 is enough (same as the notebook's minimum recommendation);
use `NVIDIA_TESLA_A100` or `NVIDIA_L4` for a faster run.

```bash
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=cpc-longbench \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=$IMAGE \
  --args="--output_gcs_uri=$BUCKET/cpc_longbench_output,--target_tokens=2000"
```

Useful flags to append to `--args` (comma-separated, matching the notebook's
config cell):
- `--limit=20` — only the first 20 examples per subset, for a quick test run.
- `--subsets=narrativeqa,qasper,hotpotqa` — restrict to specific subsets
  instead of all 21.

## 4. Monitor

```bash
gcloud ai custom-jobs list --region=$REGION
gcloud ai custom-jobs stream-logs JOB_ID --region=$REGION
```

Outputs land at `gs://your-bucket-name/cpc_longbench_output/{subset}_e_{i}.txt`
as they're produced — you can start checking the bucket before the job
finishes.

## 5. Sanity-check locally first (recommended)

Before spending GPU time on the full run, test the pipeline on one machine
with a GPU (a Workbench notebook, or any machine with `docker` + GPU
support):

```bash
docker run --gpus all $IMAGE --output_dir=/tmp/out --limit=2 --subsets=narrativeqa
```

This skips `--output_gcs_uri`, so it just writes 2 local `.txt` files under
`/tmp/out` inside the container — confirms the model loads and the
pipeline runs before you point it at GCS and all 21 subsets.

## Notes on what changed vs. the Colab notebook

- **Google Drive → GCS.** The notebook's `drive.mount()` cell and the
  `OUTPUT_DIR` local-path option are replaced by `--output_gcs_uri`.
  Resumability now checks GCS object names instead of local files, since a
  Custom Job's local disk doesn't persist across restarts (Drive did).
- **`files.download()` removed.** Not applicable outside Colab — output is
  already in GCS, so there's nothing to zip/download.
- **Config cell → CLI flags.** `TARGET_TOKENS`, `LIMIT`, `SUBSETS` are now
  `--target_tokens`, `--limit`, `--subsets` (see `--args` above).
- **`!pip install` cells → Dockerfile.** Same packages/versions, just baked
  into the image instead of run per-session.
- Everything else — `CPCCompressor`, the chunking/embedding/scoring logic,
  the output `.txt` format — is unchanged from the notebook.