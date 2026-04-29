#!/bin/bash
# Set up an isolated venv to evaluate the andimarafioti/faster-qwen3-tts repo.
#
# Why isolated: we don't yet know FasterQwenTTS's dependency footprint
# vs our main tts-remote venv. Keep it separate until we know whether
# to adopt as a library, port internals, or skip.
#
# Usage (from /workspace/tts-remote on the pod):
#     bash scripts/setup_faster_qwen3_spike.sh
#
# Foreground, fails loud. Clones into /workspace/faster-qwen3-tts (sibling
# of tts-remote on the pod), creates its own venv there per repo conventions.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"
TARGET_DIR="$PARENT_DIR/faster-qwen3-tts"
VENV="$TARGET_DIR/.venv"

echo "=== Step 1: clone repo ==="
if [ -d "$TARGET_DIR/.git" ]; then
    echo "$TARGET_DIR already exists — pulling latest."
    cd "$TARGET_DIR"
    git pull --ff-only
else
    git clone https://github.com/andimarafioti/faster-qwen3-tts.git "$TARGET_DIR"
fi
cd "$TARGET_DIR"
echo "HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current)"

echo
echo "=== Step 2: read license ==="
if [ -f LICENSE ]; then
    echo "--- LICENSE first 30 lines ---"
    head -30 LICENSE
    echo "--- ---"
elif [ -f LICENSE.md ]; then
    echo "--- LICENSE.md first 30 lines ---"
    head -30 LICENSE.md
    echo "--- ---"
else
    echo "WARNING: no LICENSE or LICENSE.md found in repo root."
    ls -la | head -20
fi

echo
echo "=== Step 3: enumerate top-level files ==="
ls -la
echo
echo "--- pyproject.toml (if present) ---"
[ -f pyproject.toml ] && cat pyproject.toml || echo "(none)"

echo
echo "=== Step 4: create venv and install ==="
if [ ! -d "$VENV" ]; then
    uv venv "$VENV" --python 3.12
fi

# Try installing as an editable package per their packaging
if [ -f pyproject.toml ]; then
    uv pip install --python "$VENV/bin/python" -e "$TARGET_DIR"
elif [ -f setup.py ]; then
    uv pip install --python "$VENV/bin/python" -e "$TARGET_DIR"
elif [ -f requirements.txt ]; then
    uv pip install --python "$VENV/bin/python" -r "$TARGET_DIR/requirements.txt"
else
    echo "ERROR: no pyproject.toml / setup.py / requirements.txt found."
    echo "Read the README and install manually:"
    [ -f README.md ] && head -80 README.md
    exit 1
fi

echo
echo "=== Step 5: report installed package metadata ==="
"$VENV/bin/python" - << 'EOF'
import importlib.metadata as md
candidates = [
    "faster-qwen3-tts",
    "faster_qwen3_tts",
    "fasterqwen3tts",
    "qwen-tts",
]
print("installed packages matching candidates:")
for name in candidates:
    try:
        dist = md.distribution(name)
        print(f"  {name}: {dist.version} (from {dist._path})")
    except md.PackageNotFoundError:
        print(f"  {name}: not found")

import torch
print()
print(f"torch in faster-qwen3-tts venv: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
EOF

echo
echo "Setup complete."
echo "Repo path:  $TARGET_DIR"
echo "Venv path:  $VENV"
echo
echo "Next: run the benchmark inspector:"
echo "    cd $REPO_DIR"
echo "    $VENV/bin/python scripts/inspect_faster_qwen3_tts.py"
