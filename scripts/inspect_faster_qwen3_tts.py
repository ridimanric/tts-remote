"""
Inspect the andimarafioti/faster-qwen3-tts repo: API surface, integration
points, and on-L4 latency / correctness.

Per Phase 3.D pivot (vLLM was a misdirection — FasterQwenTTS uses CUDA
graphs, same approach as our parked Exp 3). This inspector decides
whether to adopt the repo as a dependency, port internals, or skip.

Run from /workspace/tts-remote (after setup_faster_qwen3_spike.sh):
    /workspace/faster-qwen3-tts/.venv/bin/python scripts/inspect_faster_qwen3_tts.py

Captures:
  - Public Python API (entry points, classes, generation methods)
  - Integration footprint (does it import qwen-tts? Replace it?)
  - Voice-cloning support (essential for our product)
  - On-L4 latency on the same 48-char benchmark sentence
  - Output WAV saved for ear-check vs vanilla qwen-tts
"""
import inspect
import os
import sys
import time
import traceback
from typing import Final

REPO_PATH: Final[str] = "/workspace/faster-qwen3-tts"
TEST_SENTENCE: Final[str] = "Hello, this is a test of Qwen three TTS latency."
WARMUP_RUNS: Final[int] = 1
MEASURED_RUNS: Final[int] = 3
OUTPUT_DIR: Final[str] = "/workspace/tts-remote/traces/faster_qwen3"


def header(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> None:
    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: {REPO_PATH} does not exist.")
        print(f"Run scripts/setup_faster_qwen3_spike.sh first.")
        return

    sys.path.insert(0, REPO_PATH)

    header("Step 1: Python API discovery")

    # The package may be importable as 'faster_qwen3_tts' or similar.
    candidate_modules = (
        "faster_qwen3_tts",
        "fasterqwen3tts",
        "faster_qwen_tts",
    )
    pkg_module = None
    pkg_name: str = ""
    for name in candidate_modules:
        try:
            pkg_module = __import__(name)
            pkg_name = name
            break
        except ImportError:
            continue

    if pkg_module is None:
        print("Could not import a faster-qwen3-tts module by any expected name.")
        print(f"Inspect repo manually: {REPO_PATH}")
        for entry in sorted(os.listdir(REPO_PATH)):
            print(f"  {entry}")
        return

    print(f"Imported module: {pkg_name}")
    print(f"Module file: {pkg_module.__file__}")

    print()
    print(f"Public attributes of {pkg_name}:")
    for attr in sorted(dir(pkg_module)):
        if attr.startswith("_"):
            continue
        value = getattr(pkg_module, attr)
        kind = type(value).__name__
        print(f"  {attr:30s}  {kind}")

    header("Step 2: integration footprint — does it depend on qwen-tts?")
    try:
        import importlib.metadata as md
        dist = md.distribution(pkg_name.replace("_", "-"))
        print(f"Package: {dist.name} {dist.version}")
        print()
        print("Declared requirements:")
        for req in dist.requires or []:
            print(f"  {req}")
    except Exception as e:
        print(f"Could not read metadata: {e}")

    # Static check: does the package import qwen_tts?
    print()
    print("Source-grep for qwen_tts imports inside the package directory:")
    pkg_dir = os.path.dirname(pkg_module.__file__) if pkg_module.__file__ else None
    if pkg_dir:
        import subprocess
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "qwen_tts\\|qwen-tts\\|from transformers", pkg_dir],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout:
                for line in result.stdout.splitlines()[:30]:
                    print(f"  {line}")
            else:
                print("  no matches found.")
        except Exception as e:
            print(f"  grep failed: {e}")

    header("Step 3: identify the synthesis entry point")
    # Look for the most likely callable that runs synthesis.
    candidates: list[tuple[str, object]] = []
    for attr in dir(pkg_module):
        if attr.startswith("_"):
            continue
        value = getattr(pkg_module, attr)
        attr_lower = attr.lower()
        if any(k in attr_lower for k in ("model", "tts", "synthesize", "generate", "engine")):
            candidates.append((attr, value))

    print(f"  candidates ({len(candidates)}):")
    for name, value in candidates:
        try:
            sig = inspect.signature(value) if callable(value) else None
            print(f"    {name:40s} {type(value).__name__:30s} {sig}")
        except (ValueError, TypeError):
            print(f"    {name:40s} {type(value).__name__:30s} (sig unknown)")

    header("Step 4: voice-cloning API check")
    voice_clone_hits = []
    for attr in dir(pkg_module):
        if "voice" in attr.lower() or "clone" in attr.lower() or "speaker" in attr.lower():
            voice_clone_hits.append(attr)
    if voice_clone_hits:
        print("  voice/clone/speaker-related attributes:")
        for h in voice_clone_hits:
            print(f"    {h}")
    else:
        print("  no voice/clone/speaker attributes at top-level.")
        print("  (cloning may live on a class method — see Step 3 candidates.)")

    header("Step 5: skipping in-line synthesis benchmark")
    print("  Benchmarking is heavy and depends on the exact API the repo uses.")
    print("  Once we identify the entry-point class above, the next inspector")
    print("  call should run a real synthesis. This script stops at static")
    print("  discovery so we can decide adoption strategy first.")

    header("Done")
    print("Decision inputs collected:")
    print("  1. License — see setup_faster_qwen3_spike.sh output")
    print("  2. Module name + version — Step 1")
    print("  3. Dependency footprint — Step 2 (does it replace or wrap qwen-tts?)")
    print("  4. Synthesis entry-point candidates — Step 3")
    print("  5. Voice-cloning surface — Step 4")
    print()
    print("Once the user/engineer reviews these, the next step is either:")
    print("  - Write a benchmark script using the identified entry point")
    print("  - Decide adoption strategy (dependency vs port vs skip)")


if __name__ == "__main__":
    main()
