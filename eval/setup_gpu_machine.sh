#!/usr/bin/env bash
#
# One-time setup for running eval/longmemeval.py's CPC compressor (Llama or
# Mistral presets) on a machine with its own GPU(s) -- no Docker, no cloud
# deployment, just a venv. Run this directly on that machine (e.g. in a
# terminal inside a NoMachine session), not from your laptop.
#
# Usage:
#   git clone https://github.com/Mehrboddx/token_eff_pipeline.git
#   cd token_eff_pipeline
#   bash eval/setup_gpu_machine.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "-- Checking for GPU(s)..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found -- is an NVIDIA driver installed on this machine?" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo
echo "-- Setting up Python venv (.venv)..."
if [[ ! -d .venv ]]; then
  if ! python3 -m venv .venv 2>/tmp/venv_err.$$; then
    # Common on shared/no-sudo machines: the stdlib venv module needs the
    # system's python3-venv package (Debian/Ubuntu) or a working ensurepip,
    # neither of which a non-root user can install. The virtualenv PyPI
    # package bundles its own bootstrapping instead, so it works without
    # sudo -- installed to the user site-packages, not system-wide.
    cat /tmp/venv_err.$$ >&2
    rm -f /tmp/venv_err.$$
    echo "python3 -m venv failed (likely missing python3-venv, and you may not have sudo)." >&2
    echo "Falling back to the 'virtualenv' package instead (pip install --user)..." >&2
    pip install --user virtualenv || python3 -m pip install --user virtualenv
    python3 -m virtualenv .venv
  fi
  rm -f /tmp/venv_err.$$
fi
source .venv/bin/activate
pip install --upgrade pip

echo "-- Installing torch (cu126 wheels, same as Dockerfile.eval)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "-- Installing eval/requirements.txt..."
pip install -r eval/requirements.txt

echo
echo "== Setup done. =="
echo
echo "Before running the eval, you still need Vertex AI credentials for the"
echo "Gemini answering calls (separate from anything to do with the CPC"
echo "compressor, which runs entirely locally on this machine's GPU(s) --"
echo "no cloud needed for that part at all). Pick one:"
echo
echo "  A) Interactive login (needs a browser reachable from this session):"
echo "       gcloud auth application-default login"
echo
echo "  B) Service account key file, if you have one:"
echo "       export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
echo
echo "Either way, set which project to bill the Gemini calls to:"
echo "  export GOOGLE_CLOUD_PROJECT=<your-project-id>"
echo
echo "Then run the eval, e.g. (1B Llama preset, fits on a single 4090 easily):"
echo "  source .venv/bin/activate"
echo "  python -m eval.longmemeval --compressor cpc --cpc-preset llama --limit 25"
echo
echo "Once that looks right, scale up to Mistral and the full corpus:"
echo "  python -m eval.longmemeval --compressor cpc --cpc-preset mistral --limit 500"
echo
echo "Safe to Ctrl+C anytime -- resume with:"
echo "  python -m eval.longmemeval --resume <run_number>   (printed at the start of the run)"
echo
echo "With two GPUs, you can run two instances in parallel, one pinned to each:"
echo "  CUDA_VISIBLE_DEVICES=0 python -m eval.longmemeval --compressor cpc --cpc-preset mistral --question-types multi-session,temporal-reasoning,knowledge-update &"
echo "  CUDA_VISIBLE_DEVICES=1 python -m eval.longmemeval --compressor cpc --cpc-preset mistral --question-types single-session-user,single-session-assistant,single-session-preference &"
echo "(splits the 6 question types roughly evenly across both GPUs -- each writes its own run file, so results merge cleanly afterward.)"
