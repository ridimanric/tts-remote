#!/bin/bash
# Set up an isolated venv for the vLLM Phase 3.D spike.
#
# Why isolated: vLLM has its own torch version constraints that may not
# match the main tts-remote venv (torch 2.6.0+cu124). Running it in a
# separate .venv-vllm-spike keeps the main inference env untouched until
# we know vLLM is viable for our use case.
#
# Usage (from /workspace/tts-remote on the pod):
#     bash scripts/setup_vllm_spike.sh
#
# Runs FOREGROUND, fails loudly. ~5-10 min for the vLLM install (large
# dependencies including its own CUDA-paired torch).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/.venv-vllm-spike"

echo "=== Step 1: locate uv ==="
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not on PATH. Install via:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
echo "uv: $(command -v uv)"
uv --version

echo
echo "=== Step 2: create $VENV (Python 3.12) ==="
if [ ! -d "$VENV" ]; then
    uv venv "$VENV" --python 3.12
else
    echo "$VENV already exists — reusing."
fi

echo
echo "=== Step 3: install vllm ==="
# vLLM pulls its own torch + dependencies. Latest stable.
# This intentionally does NOT install qwen-tts in this venv — we're
# just introspecting vllm's model registry, not running models here yet.
uv pip install --python "$VENV/bin/python" vllm

echo
echo "=== Step 4: verify vllm import ==="
"$VENV/bin/python" -c "
import vllm
print('vllm version:', vllm.__version__)
import torch
print('torch in spike venv:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
"

echo
echo "Setup complete. Run the inspector with:"
echo "    uv run --project . --python $VENV/bin/python python scripts/inspect_vllm_qwen3_tts.py"
echo "or directly:"
echo "    $VENV/bin/python scripts/inspect_vllm_qwen3_tts.py"
