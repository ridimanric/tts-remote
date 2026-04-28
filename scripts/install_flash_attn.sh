#!/bin/bash
# Install flash-attn from source against the local torch install.
#
# Why source build: torch 2.6.0+cu124 from pytorch.org uses the OLD C++ ABI,
# but every prebuilt flash-attn wheel on Dao-AILab's release page (regardless
# of cxx11abiTRUE/FALSE filename label) requires the NEW ABI. The mismatch
# manifests as ImportError "_ZN3c105ErrorC2...__cxx1112basic_string..." at
# import time. Source build picks up torch's actual ABI at compile time and
# is the only path that produces a working binary.
#
# Reference: anti-voice/docs/debug-reports/2026-04-20-qwen3-tts-flash-attn-latency.md
#
# Usage (from /workspace/tts-remote on the pod):
#     bash scripts/install_flash_attn.sh
#
# The build runs in background via nohup and survives SSH disconnects.
# Tail /tmp/flash_build.log to monitor progress.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/.venv"
LOG="/tmp/flash_build.log"

if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: $VENV/bin/python not found. Run 'uv sync' first." >&2
    exit 1
fi

echo "=== Step 1: install nvidia-cuda-toolkit (provides nvcc) ==="
if ! command -v nvcc >/dev/null 2>&1; then
    apt-get update
    apt-get install -y nvidia-cuda-toolkit
else
    echo "nvcc already present at $(command -v nvcc), skipping apt install"
fi

echo
echo "=== Step 2: symlink nvcc to /usr/lib/cuda/bin/nvcc ==="
# flash-attn's setup.py looks for nvcc at $CUDA_HOME/bin/nvcc.
# Ubuntu's nvidia-cuda-toolkit puts nvcc at /usr/bin/nvcc, so symlink it.
mkdir -p /usr/lib/cuda/bin
ln -sf /usr/bin/nvcc /usr/lib/cuda/bin/nvcc
echo "Symlink: $(ls -l /usr/lib/cuda/bin/nvcc)"
nvcc --version

echo
echo "=== Step 3: kick off flash-attn build in background ==="
# MAX_JOBS controls parallelism. Each cicc process peaks ~6GB RAM. With
# 755GB pod RAM, MAX_JOBS=4 is safe and cuts build time roughly in quarter
# vs MAX_JOBS=1.
#
# FLASH_ATTENTION_FORCE_BUILD=TRUE disables setup.py's hidden path that
# downloads prebuilt wheels from GitHub releases (which have the wrong ABI).
#
# We install from git tag v2.8.3 specifically (matches anti-voice prod).
cd "$REPO_DIR"
nohup env \
    MAX_JOBS=4 \
    CUDA_HOME=/usr/lib/cuda \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    uv pip install --python "$VENV/bin/python" \
    "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3" \
    --no-build-isolation --no-cache-dir \
    > "$LOG" 2>&1 &

BUILD_PID=$!
echo "flash-attn build started: PID $BUILD_PID"
echo "Log: $LOG"
echo
echo "=== Monitor progress ==="
echo "  tail -f $LOG"
echo "  ps aux | grep -E 'nvcc|cicc|ptxas' | grep -v grep | wc -l   # active compile procs"
echo
echo "Expected build time at MAX_JOBS=4: ~25 minutes."
echo
echo "When complete, verify with:"
echo "  $VENV/bin/python -c \"import flash_attn; print('flash-attn', flash_attn.__version__)\""
