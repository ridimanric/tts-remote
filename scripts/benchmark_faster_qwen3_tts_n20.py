"""
Phase 4.A — stability/p95 benchmark for FasterQwen3TTS across sentence lengths.

Establishes whether the 4762 ms outlier in the n=3 benchmark was sampler
noise or a real tail-latency concern. Hard pass criterion (per the
investigation plan):

    Streaming TTFA p95 <= 700 ms for EVERY sentence-length bucket.

Sentence buckets: 10, 48, 120, 250 chars. n=20 measured runs per bucket
(plus 1 warmup discarded). Measures both:
  - streaming TTFA (the user-perceived latency on a phone call)
  - non-streaming total synthesis time (total compute time)

Run from /workspace/tts-remote (after setup_faster_qwen3_spike.sh):
    /workspace/faster-qwen3-tts/.venv/bin/python scripts/benchmark_faster_qwen3_tts_n20.py

Outputs:
  - Per-bucket table to stdout (p50/p95/min/max for both metrics)
  - Pass/fail verdict against 700 ms TTFA gate
  - Per-call CSV at traces/faster_qwen3_n20.csv for later analysis
"""
import csv
import os
import statistics
import sys
import time
from typing import Final

REPO_PATH: Final[str] = "/workspace/faster-qwen3-tts"
REF_AUDIO: Final[str] = f"{REPO_PATH}/ref_audio.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)

# Sentence buckets — char counts approximate the test bucket label.
SENTENCES: Final[list[tuple[int, str]]] = [
    (10, "Hi, I see."),
    (48, "Hello, this is a test of Qwen three TTS latency."),
    (
        120,
        "Welcome back to the show. Today we will explore how voice cloning "
        "has evolved over the past several years now.",
    ),
    (
        250,
        "Welcome to the system. Today I would like to walk you through the "
        "entire procedure step by step. First we will check the configuration, "
        "then verify your account details. After that, we will run the "
        "diagnostic and confirm everything is in working order today.",
    ),
]

WARMUP_RUNS: Final[int] = 1
MEASURED_RUNS: Final[int] = 20
TTFA_P95_TARGET_MS: Final[float] = 700.0
CSV_OUT: Final[str] = "/workspace/tts-remote/traces/faster_qwen3_n20.csv"


def header(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main() -> None:
    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: {REPO_PATH} not found. Run setup_faster_qwen3_spike.sh first.")
        return
    sys.path.insert(0, REPO_PATH)

    print("Sentence buckets:")
    for label, text in SENTENCES:
        print(f"  bucket={label:>3} actual_chars={len(text):>3}  text: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"\nWarmup runs per bucket: {WARMUP_RUNS}")
    print(f"Measured runs per bucket: {MEASURED_RUNS}")
    print(f"TTFA p95 target: ≤ {TTFA_P95_TARGET_MS:.0f} ms (per bucket)")

    header("Step 1: load model")
    import torch
    from faster_qwen3_tts import FasterQwen3TTS  # pyright: ignore[reportMissingImports]

    t0 = time.perf_counter()
    engine = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"engine constructed in {load_ms:.0f} ms (one-off)")

    # CSV writer setup
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    csv_file = open(CSV_OUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["bucket", "run_idx", "kind", "ttfa_ms", "total_ms", "audio_seconds"])

    # Per-bucket aggregations
    summary: dict[int, dict[str, list[float]]] = {}

    for bucket_label, text in SENTENCES:
        header(f"Bucket {bucket_label} chars — {len(text)} actual")
        ttfa_list: list[float] = []
        total_list: list[float] = []
        audio_secs_list: list[float] = []

        # Warmup
        print(f"  warmup ({WARMUP_RUNS}):")
        for i in range(WARMUP_RUNS):
            try:
                gen = engine.generate_voice_clone_streaming(
                    text=text,
                    language="English",
                    ref_audio=REF_AUDIO,
                    ref_text=REF_TEXT,
                )
                t0 = time.perf_counter()
                first_chunk = next(gen)
                ttfa_ms = (time.perf_counter() - t0) * 1000
                # Drain
                for _ in gen:
                    pass
                total_ms = (time.perf_counter() - t0) * 1000
                print(f"    warmup {i + 1}: TTFA={ttfa_ms:.0f} ms, total={total_ms:.0f} ms")
            except Exception as e:
                print(f"    warmup {i + 1} FAILED: {type(e).__name__}: {e}")
                return

        # Measured
        print(f"  measured ({MEASURED_RUNS}):")
        for i in range(MEASURED_RUNS):
            try:
                gen = engine.generate_voice_clone_streaming(
                    text=text,
                    language="English",
                    ref_audio=REF_AUDIO,
                    ref_text=REF_TEXT,
                )
                t0 = time.perf_counter()
                first_chunk = next(gen)
                ttfa_ms = (time.perf_counter() - t0) * 1000
                # Drain and count audio length
                audio_chunks = [first_chunk]
                for ch in gen:
                    audio_chunks.append(ch)
                total_ms = (time.perf_counter() - t0) * 1000

                # Compute audio duration in seconds (chunk format: (np_array, sr, meta))
                audio_secs = 0.0
                try:
                    import numpy as np
                    sr = audio_chunks[0][1] if isinstance(audio_chunks[0], tuple) else 24000
                    total_samples = sum(
                        len(ch[0]) if isinstance(ch, tuple) else len(ch)
                        for ch in audio_chunks
                    )
                    audio_secs = total_samples / sr if sr > 0 else 0.0
                except Exception:
                    pass

                ttfa_list.append(ttfa_ms)
                total_list.append(total_ms)
                audio_secs_list.append(audio_secs)
                writer.writerow([bucket_label, i + 1, "streaming", f"{ttfa_ms:.1f}", f"{total_ms:.1f}", f"{audio_secs:.2f}"])
                csv_file.flush()
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"    run {i + 1:>2}: TTFA={ttfa_ms:.0f} ms, total={total_ms:.0f} ms, audio={audio_secs:.2f}s")
            except Exception as e:
                print(f"    run {i + 1} FAILED: {type(e).__name__}: {e}")
                return

        summary[bucket_label] = {
            "ttfa": ttfa_list,
            "total": total_list,
            "audio_secs": audio_secs_list,
        }

    csv_file.close()

    # Print summary table
    header("Summary — TTFA (streaming time-to-first-audio)")
    print(f"  {'bucket':>6} {'p50':>8} {'p95':>8} {'min':>8} {'max':>8} {'pass?':>8}")
    all_pass = True
    for bucket_label in [s[0] for s in SENTENCES]:
        ttfa = summary[bucket_label]["ttfa"]
        p50 = percentile(ttfa, 50)
        p95 = percentile(ttfa, 95)
        passed = p95 <= TTFA_P95_TARGET_MS
        if not passed:
            all_pass = False
        marker = "PASS" if passed else "FAIL"
        print(f"  {bucket_label:>6} {p50:>8.0f} {p95:>8.0f} {min(ttfa):>8.0f} {max(ttfa):>8.0f} {marker:>8}")

    header("Summary — total synthesis time (full audio ready)")
    print(f"  {'bucket':>6} {'p50':>8} {'p95':>8} {'min':>8} {'max':>8} {'audio_s':>8} {'rtf':>8}")
    for bucket_label in [s[0] for s in SENTENCES]:
        total = summary[bucket_label]["total"]
        audio_s = summary[bucket_label]["audio_secs"]
        avg_audio_s = statistics.mean(audio_s) if audio_s else 0.0
        avg_total_s = statistics.mean(total) / 1000.0 if total else 0.0
        rtf = avg_total_s / avg_audio_s if avg_audio_s > 0 else float("nan")
        p50 = percentile(total, 50)
        p95 = percentile(total, 95)
        print(f"  {bucket_label:>6} {p50:>8.0f} {p95:>8.0f} {min(total):>8.0f} {max(total):>8.0f} {avg_audio_s:>8.2f} {rtf:>8.2f}")
    print()
    print("  rtf < 1.0 means generation outpaces playback (streaming never starves).")

    header("VERDICT")
    if all_pass:
        print("  PASS — TTFA p95 ≤ 700 ms across every sentence-length bucket.")
        print("  Phase 4.A criterion met. Proceed to Phase 4.B (integration).")
    else:
        print("  FAIL — at least one bucket exceeds the 700 ms TTFA p95 target.")
        print("  Investigate the failing bucket(s) before integration.")
    print(f"\n  Per-call CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()
