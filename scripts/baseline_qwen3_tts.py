"""
Baseline benchmark: Qwen3-TTS synthesis latency on L4 GPU.

Mirrors anti-voice prod's _synthesize_qwen3 path:
  - Qwen3TTSModel.from_pretrained with attn_implementation="flash_attention_2"
  - bfloat16, cuda:0
  - generate_voice_clone with default Aliyun reference WAV

Goal: confirm we land in the ~10s/call range observed on AWS (relaxed exit
gate from the investigation plan: anywhere 6-15s confirms we have the same
problem to fix on this pod).

Run from /workspace/tts-remote (with .venv active or via .venv/bin/python):
    .venv/bin/python scripts/baseline_qwen3_tts.py
"""
import time
from typing import Final

# 48 chars exactly — matches the sentence used on AWS.
TEST_SENTENCE: Final[str] = "Hello, this is a test of Qwen three TTS latency."
WARMUP_RUNS: Final[int] = 1
MEASURED_RUNS: Final[int] = 3
REF_AUDIO: Final[str] = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)


def main() -> None:
    print(f"Test sentence ({len(TEST_SENTENCE)} chars): {TEST_SENTENCE!r}")
    print()

    print("Loading Qwen3-TTS (flash_attention_2, bf16, cuda:0)...")
    import torch
    from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f} ms")
    print()

    print(f"Warmup runs: {WARMUP_RUNS}")
    for i in range(WARMUP_RUNS):
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=TEST_SENTENCE,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i + 1}: {dt_ms:.0f} ms (sr={sr}, samples={len(wavs[0])})")
    print()

    print(f"Measured runs: {MEASURED_RUNS}")
    measurements: list[float] = []
    for i in range(MEASURED_RUNS):
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=TEST_SENTENCE,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        measurements.append(dt_ms)
        print(f"  call {i + 1}: {dt_ms:.0f} ms")

    print()
    print("=== Summary ===")
    print(f"  test sentence: {len(TEST_SENTENCE)} chars")
    print(f"  per-call (ms): {[f'{m:.0f}' for m in measurements]}")
    print(f"  min: {min(measurements):.0f} ms")
    print(f"  max: {max(measurements):.0f} ms")
    print(f"  mean: {sum(measurements) / len(measurements):.0f} ms")
    print()
    aws_baseline_ms = 10000.0
    delta_pct = (
        (sum(measurements) / len(measurements) - aws_baseline_ms) / aws_baseline_ms * 100
    )
    print(f"  AWS baseline reference: ~{aws_baseline_ms:.0f} ms")
    print(f"  delta: {delta_pct:+.0f}%")
    if 6000 <= sum(measurements) / len(measurements) <= 15000:
        print(f"  EXIT GATE PASSED (mean is in 6-15s range; same problem reproduced)")
    else:
        print(f"  EXIT GATE FAILED (mean outside 6-15s range; investigate env drift)")


if __name__ == "__main__":
    main()
