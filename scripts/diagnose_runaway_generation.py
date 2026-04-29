"""
Diagnose runaway/truncated generation in FasterQwen3TTS.

Phase 4.A revealed that while TTFA is rock-stable (640 ms), total synthesis
time and audio length are wildly variable:
  - 250-char sentence hits max_new_tokens (149 s of audio) on EVERY run
  - 48-char sentence: one run = 0.16 s audio (truncated), one = 149 s (runaway)
  - 10-char "Hi, I see." produced 4.00 s to 15.44 s of audio

This script tests four hypotheses about the cause:

  H1 - default params:   do_sample=True, temperature=0.9   (current config)
  H2 - greedy decoding:  do_sample=False                   (eliminates sampler)
  H3 - upstream tuned:   do_sample=True, temperature=0.7   (qwen-tts docs recco)
  H4 - low temperature:  do_sample=True, temperature=0.5   (more conservative)
  H5 - rep penalty:      do_sample=True, temperature=0.9, repetition_penalty=1.2

Each run is classified:
  - expected:   audio length in [0.5 * char_count / 8, 3 * char_count / 8] seconds
                (rough estimate: speech is ~8 chars/second)
  - runaway:    audio length > 3x expected
  - truncated:  audio length < 0.5x expected

Tests two sentence buckets known to fail at default params:
  - 48 chars  (intermittent failures)
  - 250 chars (always runaway at default)

Also reads FasterQwen3TTS's tests/test_e2e_parity.py to see what THEIR
canonical config is (in case it differs from from_pretrained defaults).

Run from /workspace/tts-remote (after setup_faster_qwen3_spike.sh):
    /workspace/faster-qwen3-tts/.venv/bin/python scripts/diagnose_runaway_generation.py

Outputs per-config table + recommendation.
"""
import csv
import os
import sys
import time
from typing import Final

REPO_PATH: Final[str] = "/workspace/faster-qwen3-tts"
REF_AUDIO: Final[str] = f"{REPO_PATH}/ref_audio.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)

SENTENCES: Final[list[tuple[int, str]]] = [
    (48, "Hello, this is a test of Qwen three TTS latency."),
    (
        250,
        "Welcome to the system. Today I would like to walk you through the "
        "entire procedure step by step. First we will check the configuration, "
        "then verify your account details. After that, we will run the "
        "diagnostic and confirm everything is in working order today.",
    ),
]

CONFIGS: Final[list[tuple[str, dict]]] = [
    ("H1_default", {"do_sample": True, "temperature": 0.9}),
    ("H2_greedy", {"do_sample": False, "temperature": 1.0}),  # temp ignored when greedy
    ("H3_upstream_tuned", {"do_sample": True, "temperature": 0.7}),
    ("H4_low_temp", {"do_sample": True, "temperature": 0.5}),
    ("H5_rep_penalty", {"do_sample": True, "temperature": 0.9, "repetition_penalty": 1.2}),
]

RUNS_PER_BUCKET: Final[int] = 10
WARMUP_RUNS: Final[int] = 1

# Speech rate heuristic: ~8 characters per second for natural English speech.
# Wide tolerance because reference voice's pace varies.
CHARS_PER_SEC_EXPECTED: Final[float] = 8.0

CSV_OUT: Final[str] = "/workspace/tts-remote/traces/faster_qwen3_runaway_diagnosis.csv"


def header(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def classify_audio_length(char_count: int, audio_seconds: float) -> str:
    """expected / runaway / truncated based on char-vs-audio ratio."""
    if audio_seconds <= 0:
        return "empty"
    expected_seconds = char_count / CHARS_PER_SEC_EXPECTED
    if audio_seconds > 3.0 * expected_seconds:
        return "runaway"
    if audio_seconds < 0.5 * expected_seconds:
        return "truncated"
    return "expected"


def main() -> None:
    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: {REPO_PATH} not found. Run setup_faster_qwen3_spike.sh first.")
        return
    sys.path.insert(0, REPO_PATH)

    # ------------------------------------------------------------------
    header("Step 0: read FasterQwen3TTS's parity test for canonical config")
    # ------------------------------------------------------------------
    parity_test_path = os.path.join(REPO_PATH, "tests", "test_e2e_parity.py")
    if os.path.exists(parity_test_path):
        with open(parity_test_path) as f:
            content = f.read()
        # Look for generation kwargs in the test
        print(f"  reading {parity_test_path}")
        print(f"  ({len(content)} bytes)")
        # Print lines with relevant kwargs
        relevant_terms = (
            "do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
            "max_new_tokens", "min_new_tokens", "seed",
        )
        for i, line in enumerate(content.splitlines(), 1):
            if any(t in line for t in relevant_terms):
                print(f"    L{i}: {line.strip()}")
    else:
        print(f"  WARNING: {parity_test_path} not found. Skipping.")

    # ------------------------------------------------------------------
    header("Step 1: load FasterQwen3TTS once")
    # ------------------------------------------------------------------
    import torch
    from faster_qwen3_tts import FasterQwen3TTS  # pyright: ignore[reportMissingImports]

    t0 = time.perf_counter()
    engine = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  engine constructed in {load_ms:.0f} ms (one-off)")

    # ------------------------------------------------------------------
    # Run all configs × buckets × N runs
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    csv_file = open(CSV_OUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "config", "bucket", "run_idx", "ttfa_ms", "total_ms",
        "audio_seconds", "classification",
    ])

    # results[config][bucket] = list of (audio_secs, classification)
    results: dict[str, dict[int, list[tuple[float, str]]]] = {}

    for config_name, gen_kwargs in CONFIGS:
        results[config_name] = {}
        for bucket, text in SENTENCES:
            header(f"Config {config_name}  |  bucket {bucket} chars")
            print(f"  kwargs: {gen_kwargs}")
            results[config_name][bucket] = []

            # Warmup
            for _ in range(WARMUP_RUNS):
                try:
                    _ = engine.generate_voice_clone(
                        text=text,
                        language="English",
                        ref_audio=REF_AUDIO,
                        ref_text=REF_TEXT,
                        **gen_kwargs,
                    )
                except Exception as e:
                    print(f"    warmup FAILED: {type(e).__name__}: {e}")
                    return

            # Measured
            for run_idx in range(RUNS_PER_BUCKET):
                try:
                    t0 = time.perf_counter()
                    wav_out = engine.generate_voice_clone(
                        text=text,
                        language="English",
                        ref_audio=REF_AUDIO,
                        ref_text=REF_TEXT,
                        **gen_kwargs,
                    )
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

                    cls = classify_audio_length(bucket, audio_secs)
                    results[config_name][bucket].append((audio_secs, cls))
                    writer.writerow([
                        config_name, bucket, run_idx + 1,
                        f"{0.0:.1f}",  # not measuring TTFA here (non-streaming)
                        f"{total_ms:.1f}",
                        f"{audio_secs:.2f}",
                        cls,
                    ])
                    csv_file.flush()
                    if (run_idx + 1) % 5 == 0 or run_idx == 0:
                        print(f"    run {run_idx + 1:>2}: total={total_ms:.0f} ms, audio={audio_secs:.2f}s [{cls}]")
                except Exception as e:
                    print(f"    run {run_idx + 1} FAILED: {type(e).__name__}: {e}")

    csv_file.close()

    # ------------------------------------------------------------------
    header("Summary table — runaway/truncated/expected counts per config × bucket")
    # ------------------------------------------------------------------
    print(f"  {'config':<20} {'bucket':>6} {'expected':>9} {'runaway':>9} {'truncated':>10} {'success_pct':>12}")
    best_config: str | None = None
    best_success: float = -1.0
    for config_name, gen_kwargs in CONFIGS:
        for bucket, _ in SENTENCES:
            runs = results[config_name][bucket]
            counts = {"expected": 0, "runaway": 0, "truncated": 0, "empty": 0}
            for _, cls in runs:
                counts[cls] = counts.get(cls, 0) + 1
            success_pct = counts["expected"] / max(len(runs), 1) * 100.0
            print(
                f"  {config_name:<20} {bucket:>6} "
                f"{counts['expected']:>9} {counts['runaway']:>9} {counts['truncated']:>10} "
                f"{success_pct:>11.0f}%"
            )
            # Track best by 250-char success (the hardest case)
            if bucket == 250 and success_pct > best_success:
                best_success = success_pct
                best_config = config_name

    header("Audio-length distribution per config (250-char bucket)")
    for config_name, _ in CONFIGS:
        runs = results[config_name][250]
        lengths = [s for s, _ in runs]
        if lengths:
            print(
                f"  {config_name:<20}  "
                f"min={min(lengths):>6.2f}s  "
                f"max={max(lengths):>6.2f}s  "
                f"mean={sum(lengths)/len(lengths):>6.2f}s  "
                f"(expected ~31s)"
            )

    header("VERDICT")
    if best_config and best_success >= 80.0:
        print(f"  {best_config} has {best_success:.0f}% success on 250-char (worst case).")
        print(f"  Tentative recommendation: adopt {best_config}'s gen_kwargs.")
    elif best_config and best_success > 0:
        print(f"  Best config is {best_config} at {best_success:.0f}% success on 250-char.")
        print(f"  None of the tested configs reached 80% success on the worst case.")
        print(f"  Investigation required: open issue on faster-qwen3-tts repo,")
        print(f"  or look at FasterQwen3TTS's parity test config in Step 0 above.")
    else:
        print(f"  All configs failed to produce expected-length audio reliably.")
        print(f"  Stop here and re-evaluate.")
    print()
    print(f"  Per-call CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()
