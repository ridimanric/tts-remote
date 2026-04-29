"""
Find the input-text-length break point for FasterQwen3TTS using its
canonical config + seeded RNG (matching tests/test_e2e_parity.py).

Background: previous diagnosis showed:
  - Stochastic sampling (do_sample=True) is unreliable — not validated
    by the library's parity tests.
  - Greedy decoding (do_sample=False) is 100% reliable on 48-char input.
  - Greedy decoding hits max_new_tokens on 250-char input regardless of
    cap (also seen in upstream issue #73 with max_new_tokens=3600).

We don't yet know:
  1. Where between 48 and 250 chars the model stops behaving correctly.
  2. Whether the parity test's seeding (`_seed_all(0)` per call) affects
    behaviour vs no seeding (default in our previous tests).

This script sweeps input length 10..250 chars with their canonical config
plus per-call seeding. Each length runs 3 times — if the model is truly
deterministic with seed=0, all 3 audio outputs should be identical.

Output tells us:
  - The maximum input char count for which generation is reliable
  - Whether seeding makes a difference (vs unseeded)
  - Where to chunk text in prod (text > N chars must be split before TTS)

Run from /workspace/tts-remote (after setup_faster_qwen3_spike.sh):
    /workspace/faster-qwen3-tts/.venv/bin/python scripts/diagnose_length_breakpoint.py
"""
import csv
import os
import random
import sys
import time
from typing import Final


REPO_PATH: Final[str] = "/workspace/faster-qwen3-tts"
REF_AUDIO: Final[str] = f"{REPO_PATH}/ref_audio.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)

# Sentence length sweep — manually crafted at target char counts so prosody
# is consistent across the sweep (no truncation artefacts mid-word).
SENTENCES: Final[list[tuple[int, str]]] = [
    (10, "Hi, I see."),
    (40, "Welcome to the system. Today is testing."),
    (70, "Welcome to the system. Today is testing. We will run a quick check."),
    (100, "Welcome to the system. Today is testing. We will run a quick check on audio synthesis pipeline."),
    (130, "Welcome to the system. Today is testing. We will run a quick check on the audio synthesis pipeline before we begin."),
    (160, "Welcome to the system. Today is testing. We will run a quick check on the audio synthesis pipeline before we begin the benchmark process."),
    (190, "Welcome to the system. Today is testing. We will run a quick check on the audio synthesis pipeline before we begin the benchmark process for latency."),
    (220, "Welcome to the system. Today is testing. We will run a quick check on the audio synthesis pipeline before we begin the benchmark process for latency and stability."),
    (
        250,
        "Welcome to the system. Today is testing. We will run a quick check on the audio synthesis pipeline before we begin the benchmark process for latency and stability across the test sentences.",
    ),
]

# FasterQwen3TTS canonical config from tests/test_e2e_parity.py — only
# combination validated by upstream parity tests.
CANONICAL_KWARGS: Final[dict] = {
    "do_sample": False,
    "top_k": 0,
    "top_p": 1.0,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "min_new_tokens": 0,
    "max_new_tokens": 2048,
}

RUNS_PER_LENGTH: Final[int] = 3
WARMUP_RUNS: Final[int] = 1
CHARS_PER_SEC_EXPECTED: Final[float] = 8.0
MAX_AUDIO_SECONDS: Final[float] = CANONICAL_KWARGS["max_new_tokens"] / 12.0

CSV_OUT: Final[str] = "/workspace/tts-remote/traces/faster_qwen3_length_breakpoint.csv"


def header(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def seed_all(seed: int = 0) -> None:
    """Mirror FasterQwen3TTS's tests/test_e2e_parity.py::_seed_all."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classify(char_count: int, audio_seconds: float) -> str:
    if audio_seconds <= 0:
        return "empty"
    expected = char_count / CHARS_PER_SEC_EXPECTED
    near_cap = audio_seconds >= 0.9 * MAX_AUDIO_SECONDS
    too_long = audio_seconds > 2.0 * expected
    if near_cap or too_long:
        return "runaway"
    if audio_seconds < 0.5 * expected:
        return "truncated"
    return "expected"


def main() -> None:
    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: {REPO_PATH} not found. Run setup_faster_qwen3_spike.sh first.")
        return
    sys.path.insert(0, REPO_PATH)

    print("Sentence length sweep:")
    for label, text in SENTENCES:
        print(f"  bucket={label:>3}  actual={len(text):>3}  text: {text[:70]}{'...' if len(text) > 70 else ''}")
    print(f"\nCanonical config (matching tests/test_e2e_parity.py):")
    for k, v in CANONICAL_KWARGS.items():
        print(f"  {k}={v}")
    print(f"\nSeeding `_seed_all(0)` before EACH call.")
    print(f"Runs per length: {RUNS_PER_LENGTH} (greedy with seed=0 should be deterministic)")
    print(f"max_new_tokens cap = {CANONICAL_KWARGS['max_new_tokens']} → audio cap ~{MAX_AUDIO_SECONDS:.1f}s")

    header("Step 1: load FasterQwen3TTS")
    from faster_qwen3_tts import FasterQwen3TTS  # pyright: ignore[reportMissingImports]

    t0 = time.perf_counter()
    engine = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print(f"  engine constructed in {(time.perf_counter() - t0) * 1000:.0f} ms")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    csv_file = open(CSV_OUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["bucket", "actual_chars", "run_idx", "total_ms", "audio_seconds", "classification"])

    # results[bucket] = list of (audio_secs, classification)
    results: dict[int, list[tuple[float, str]]] = {}

    # Warmup once total (any length is fine — it primes the CUDA graphs)
    print()
    print("Warmup run (primes CUDA graphs)...")
    seed_all(0)
    _ = engine.generate_voice_clone(
        text="Warmup.", language="English",
        ref_audio=REF_AUDIO, ref_text=REF_TEXT,
        **CANONICAL_KWARGS,
    )
    print("  warmup done.")

    for bucket, text in SENTENCES:
        header(f"bucket {bucket} chars (actual {len(text)})")
        results[bucket] = []
        for run_idx in range(RUNS_PER_LENGTH):
            seed_all(0)  # match parity test pattern: seed BEFORE each call
            t0 = time.perf_counter()
            try:
                wav_out = engine.generate_voice_clone(
                    text=text,
                    language="English",
                    ref_audio=REF_AUDIO,
                    ref_text=REF_TEXT,
                    **CANONICAL_KWARGS,
                )
            except Exception as e:
                print(f"  run {run_idx + 1} FAILED: {type(e).__name__}: {e}")
                results[bucket].append((0.0, "error"))
                continue
            total_ms = (time.perf_counter() - t0) * 1000

            # Extract audio length
            audio_secs = 0.0
            try:
                if isinstance(wav_out, tuple) and len(wav_out) >= 2:
                    wav_list, sr = wav_out[0], wav_out[1]
                    if isinstance(wav_list, list) and len(wav_list) > 0:
                        wav_arr = wav_list[0]
                        import numpy as np
                        if hasattr(wav_arr, "cpu"):
                            wav_arr = wav_arr.cpu().numpy()
                        wav_arr = np.asarray(wav_arr).squeeze()
                        audio_secs = len(wav_arr) / sr if sr > 0 else 0.0
            except Exception:
                pass

            cls = classify(bucket, audio_secs)
            results[bucket].append((audio_secs, cls))
            writer.writerow([bucket, len(text), run_idx + 1, f"{total_ms:.0f}",
                            f"{audio_secs:.3f}", cls])
            csv_file.flush()
            print(f"  run {run_idx + 1}: total={total_ms:.0f} ms, audio={audio_secs:.3f}s [{cls}]")

    csv_file.close()

    # ------------------------------------------------------------------
    header("Per-bucket summary")
    # ------------------------------------------------------------------
    print(f"  {'bucket':>6} {'min_s':>8} {'max_s':>8} {'spread':>8} {'expected':>9} {'classes'}")
    last_safe_bucket: int | None = None
    for bucket, _ in SENTENCES:
        runs = results[bucket]
        if not runs:
            continue
        lengths = [s for s, _ in runs]
        classes = [c for _, c in runs]
        spread = max(lengths) - min(lengths)
        expected_label = "expected" if all(c == "expected" for c in classes) else ("runaway" if all(c == "runaway" for c in classes) else "mixed")
        cls_summary = "/".join(f"{c}={classes.count(c)}" for c in {*classes})
        print(
            f"  {bucket:>6} "
            f"{min(lengths):>8.3f} {max(lengths):>8.3f} {spread:>8.3f} "
            f"{expected_label:>9} {cls_summary}"
        )
        if expected_label == "expected" and spread < 0.01:
            last_safe_bucket = bucket

    # ------------------------------------------------------------------
    header("Determinism check (greedy + seed=0 should produce identical runs)")
    # ------------------------------------------------------------------
    deterministic_buckets = []
    nondeterministic_buckets = []
    for bucket, _ in SENTENCES:
        runs = results[bucket]
        if not runs:
            continue
        lengths = [s for s, _ in runs]
        spread = max(lengths) - min(lengths)
        if spread < 0.01:
            deterministic_buckets.append(bucket)
        else:
            nondeterministic_buckets.append((bucket, spread))

    print(f"  deterministic (spread <10ms): {deterministic_buckets}")
    print(f"  non-deterministic (spread): {nondeterministic_buckets}")

    # ------------------------------------------------------------------
    header("VERDICT")
    # ------------------------------------------------------------------
    if last_safe_bucket is not None:
        print(f"  Last sentence-length bucket producing reliable audio: {last_safe_bucket} chars")
        # find the smallest bucket where it broke
        broke_at = None
        for b, _ in SENTENCES:
            if b > last_safe_bucket:
                runs = results[b]
                classes = [c for _, c in runs]
                if any(c != "expected" for c in classes):
                    broke_at = b
                    break
        if broke_at:
            print(f"  Break point: between {last_safe_bucket} and {broke_at} chars")
            print(f"  Action: text-chunk LLM responses to <= {last_safe_bucket} chars before TTS.")
    else:
        print(f"  Even the shortest bucket failed. Something fundamental is wrong.")

    print(f"\n  Per-call CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()
