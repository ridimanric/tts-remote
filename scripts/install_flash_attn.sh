#!/bin/bash
# Build flash-attn from source against the local torch install.
#
# Why source build: torch 2.6.0+cu124 from pytorch.org uses the OLD C++ ABI,
# but every prebuilt flash-attn wheel on Dao-AILab's release page (regardless
# of cxx11abiTRUE/FALSE filename label) requires the NEW ABI. Source build
# picks up torch's actual ABI at compile time and is the only path that
# produces a working binary.
#
# Reference: anti-voice/docs/debug-reports/2026-04-20-qwen3-tts-flash-attn-latency.md
#
# Why pip -v instead of uv: uv pip install buffers compile output. For a
# 30-60min build we need to see every nvcc/g++ line live so any stall or
# error is visible immediately. pip -v streams stdout/stderr in real time.
#
# Usage (from /workspace/tts-remote on the pod):
#     bash scripts/install_flash_attn.sh
#
# Runs in the FOREGROUND. Build takes ~30-60 minutes at MAX_JOBS=4 depending
# on which kernels are heaviest.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: $VENV/bin/python not found. Run 'uv sync' first." >&2
    exit 1
fi

echo "=== locate nvcc ==="
NVCC_PATH="$(command -v nvcc || true)"
if [ -z "$NVCC_PATH" ]; then
    echo "ERROR: nvcc not found in PATH. Install nvidia-cuda-toolkit:" >&2
    echo "    apt-get install -y nvidia-cuda-toolkit" >&2
    exit 1
fi
echo "nvcc: $NVCC_PATH"
nvcc --version

NVCC_BIN_DIR="$(dirname "$NVCC_PATH")"
CUDA_HOME_VAL="$(dirname "$NVCC_BIN_DIR")"
echo "CUDA_HOME -> $CUDA_HOME_VAL"

echo
echo "=== ensure pip is available in the venv ==="
# uv venvs don't include pip by default. Bootstrap it.
"$VENV/bin/python" -m ensurepip --upgrade 2>/dev/null || true
"$VENV/bin/python" -m pip --version

echo
echo "=== build flash-attn (foreground, pip -v streamed, ~30-60 min) ==="
echo "Each nvcc/g++ command will print as it runs."
echo "If it hangs, you'll see exactly which file last started compiling."
echo

cd "$REPO_DIR"
MAX_JOBS=4 \
CUDA_HOME="$CUDA_HOME_VAL" \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
"$VENV/bin/python" -m pip install -v \
    "git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3" \
    --no-build-isolation --no-cache-dir

echo
echo "=== verify import ==="
"$VENV/bin/python" -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
