"""
Run CPC compression over LongBench v1 and write one .txt per example to
Google Cloud Storage (or a local directory).

This is the Vertex AI Custom Job port of the original Colab notebook's
"Run over LongBench v1" cell. Differences from the Colab version:
  - No `google.colab.drive` mount / `google.colab.files.download` — output
    goes straight to GCS (or stays local if --output_gcs_uri is omitted).
  - Resumability is checked against GCS objects already present under the
    prefix (not local disk), so a preempted/restarted job picks up where
    it left off instead of re-doing finished examples.
  - Config is via CLI flags / env vars instead of hardcoded notebook cells.

Example (see README.md for the full gcloud submission command):

    python run_longbench.py \
        --output_gcs_uri gs://my-bucket/cpc_longbench_output \
        --target_tokens 2000 \
        --subsets narrativeqa,qasper,hotpotqa
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ALL_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
    "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news",
    "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count",
    "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
]


def format_output(example: dict, compressed_context: str) -> str:
    answers = example.get("answers") or []
    answers_str = "; ".join(str(a) for a in answers) if answers else "(none provided)"
    return (
        f"ID: {example['_id']}\n"
        f"Dataset: {example['dataset']}\n"
        f"Language: {example['language']}\n"
        f"Length: {example['length']}\n"
        f"Answers: {answers_str}\n"
        f"{'-' * 60}\n\n"
        f"CONTEXT:\n{compressed_context}\n\n"
        f"INPUT:\n{example['input']}\n"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_gcs_uri",
        default=os.environ.get("OUTPUT_GCS_URI", ""),
        help="gs://bucket/prefix to upload each .txt to. If omitted, output "
             "only goes to --output_dir (useful for local testing).",
    )
    p.add_argument(
        "--output_dir",
        default="/tmp/compressed_longbench",
        help="Local staging directory (also the final location if "
             "--output_gcs_uri is not set).",
    )
    p.add_argument(
        "--target_tokens",
        type=int,
        default=int(os.environ.get("TARGET_TOKENS", 2000)),
        help="compression_target_tokens per example.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N examples PER SUBSET (omit for all).",
    )
    p.add_argument(
        "--subsets",
        default=",".join(ALL_SUBSETS),
        help="Comma-separated list of LongBench v1 subsets to run.",
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=6144,
        help="Per-chunk token budget for the compressor's forward pass. Lower "
             "this on GPUs with limited VRAM (more, smaller chunks; slower "
             "but still correct). Only applies to --compressor=cpc.",
    )
    p.add_argument(
        "--compressor",
        choices=["cpc", "gemini"],
        default=os.environ.get("COMPRESSOR", "cpc"),
        help="'cpc' runs the local CPC embedding model (default); 'gemini' "
             "calls a hosted Gemini model over Vertex AI to do the "
             "compression instead.",
    )
    p.add_argument(
        "--gemini_model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Model id to use when --compressor=gemini.",
    )
    p.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP project for Vertex AI (only needed for --compressor=gemini).",
    )
    p.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    return p.parse_args()


def build_compressor(args):
    if args.compressor == "gemini":
        from gemini_compressor import GeminiCompressor

        return GeminiCompressor(project=args.project, location=args.location, model=args.gemini_model)

    from cpc_compressor import CPCCompressor

    return CPCCompressor(max_seq_length=args.max_seq_length)


def main():
    args = parse_args()
    print(f"Compressor: {args.compressor}"
          + (f" ({args.gemini_model})" if args.compressor == "gemini" else ""))
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gcs = bool(args.output_gcs_uri)
    bucket = None
    blob_prefix = ""
    existing_blobs = set()
    if use_gcs:
        from google.cloud import storage

        assert args.output_gcs_uri.startswith("gs://"), "--output_gcs_uri must start with gs://"
        bucket_name, _, blob_prefix = args.output_gcs_uri[len("gs://"):].partition("/")
        blob_prefix = blob_prefix.rstrip("/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        # List what's already there once, up front, so resuming a preempted
        # job doesn't re-run (and re-pay GPU time for) finished examples.
        print(f"Checking existing objects under gs://{bucket_name}/{blob_prefix} ...")
        existing_blobs = {
            b.name.rsplit("/", 1)[-1] for b in client.list_blobs(bucket_name, prefix=blob_prefix)
        }
        print(f"  found {len(existing_blobs)} existing output files.")

    compressor = build_compressor(args)

    for subset in subsets:
        dataset = load_dataset("THUDM/LongBench", subset, split="test", trust_remote_code=True)
        if args.limit is not None:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

        for i, example in enumerate(dataset):
            filename = f"{subset}_e_{i:04d}.txt"
            out_path = output_dir / filename

            if use_gcs and filename in existing_blobs:
                continue  # resumable: skip examples already uploaded to GCS
            if not use_gcs and out_path.exists():
                continue  # resumable: skip examples already on local disk

            compressed_context = compressor.compress(
                context=example["context"],
                question=example["input"],
                compression_target_tokens=args.target_tokens,
            )

            out_path.write_text(format_output(example, compressed_context), encoding="utf-8")

            if use_gcs:
                blob_name = f"{blob_prefix}/{filename}" if blob_prefix else filename
                bucket.blob(blob_name).upload_from_filename(str(out_path))

            print(f"[{subset}_e {i + 1}/{len(dataset)}] wrote {filename}"
                  + (" -> GCS" if use_gcs else ""))

    print("Done.")


if __name__ == "__main__":
    main()