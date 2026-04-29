"""
Compare vanilla qwen-tts vs FasterQwen3TTS on the same length sweep.

If vanilla qwen-tts produces correct-length audio for ALL buckets but
FasterQwen3TTS doesn't, the bug is in FasterQwen3TTS. If vanilla
qwen-tts shows the same failure pattern, the model itself has issues.

This is the decisive test before deciding whether to abandon
FasterQwen3TTS or pursue an upstream fix / accept a workaround.

Also:
  - Saves each generation's WAV to traces/vanilla_length_breakpoint/
    so we can ear-check what the failure cases actually sound like.
  - Reads FasterQwen3TTS's tests/test_e2e_parity.py and prints any
    test phrases found, telling us what lengths they actually validate.

Run from /workspace/tts-remote (uses the FasterQwen3TTS venv since it
has both qwen-tts and FasterQwen3TTS — clean apples-to-apples since
torch versions match):
    /workspace/faster-qwen3-tts/.venv/bin/python scripts/compare_vanilla_qwen_tts.py
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

# Identical sentences to diagnose_length_breakpoint.py for direct comparison.
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

# Same canonical config used by FasterQwen3TTS parity tests, applied to
# vanilla qwen-tts via its own generate_voice_clone parameters.
CANONICAL_KWARGS: Final[dict] = {
    "do_sample": False,
    "top_k": 0,
    "top_p": 1.0,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 2048,
}

RUNS_PER_LENGTH: Final[int] = 1  # greedy + seed = deterministic, 1 is enough
CHARS_PER_SEC_EXPECTED: Final[float] = 8.0
MAX_AUDIO_SECONDS: Final[float] = CANONICAL_KWARGS["max_new_tokens"] / 12.0
WAV_OUT_DIR: Final[str] = "/workspace/tts-remote/traces/vanilla_length_breakpoint"
CSV_OUT: Final[str] = "/workspace/tts-remote/traces/vanilla_qwen_tts_breakpoint.csv"


def header(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def seed_all(seed: int = 0) -> None:
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
    # ------------------------------------------------------------------
    header("Step 0: read FasterQwen3TTS parity-test phrases")
    # ------------------------------------------------------------------
    parity_path = os.path.join(REPO_PATH, "tests", "test_e2e_parity.py")
    if os.path.exists(parity_path):
        with open(parity_path) as f:
            content = f.read()
        # Extract any 'text="..."' or text=variable assignments + their definitions
        import re
        # Look for string literals that look like sentences (>20 chars)
        sentence_matches = re.findall(r'"([A-Z][^"]{20,300}\.)"', content)
        # Filter to plausible test phrases (start with a capital letter, end with period)
        seen = set()
        unique_phrases = []
        for s in sentence_matches:
            if s not in seen:
                seen.add(s)
                unique_phrases.append(s)
        if unique_phrases:
            print(f"  found {len(unique_phrases)} unique sentence-like strings:")
            for s in unique_phrases[:20]:
                print(f"    [{len(s):>3} chars] {s[:100]}{'...' if len(s) > 100 else ''}")
            print(f"\n  Length distribution of validated phrases:")
            lengths = sorted(len(s) for s in unique_phrases)
            print(f"    min={min(lengths)}  max={max(lengths)}  median={lengths[len(lengths)//2]}")
        else:
            print("  no sentence-like strings found — they may be in a fixture file.")
    else:
        print(f"  WARNING: {parity_path} not found.")

    # ------------------------------------------------------------------
    header("Step 1: load vanilla qwen-tts (NOT FasterQwen3TTS)")
    # ------------------------------------------------------------------
    import torch
    from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"  loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")

    os.makedirs(WAV_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    csv_file = open(CSV_OUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["bucket", "actual_chars", "run_idx", "total_ms", "audio_seconds",
                     "classification", "wav_path"])

    # ------------------------------------------------------------------
    header("Step 2: pre-extract voice_clone_prompt")
    # ------------------------------------------------------------------
    seed_all(0)
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
    )
    print(f"  voice_clone_prompt extracted in {(time.perf_counter() - t0) * 1000:.0f} ms")

    # ------------------------------------------------------------------
    # Run the same length sweep
    # ------------------------------------------------------------------
    import numpy as np
    import soundfile as sf

    results: dict[int, list[tuple[float, str, str]]] = {}

    for bucket, text in SENTENCES:
        header(f"vanilla qwen-tts | bucket {bucket} chars (actual {len(text)})")
        results[bucket] = []
        for run_idx in range(RUNS_PER_LENGTH):
            seed_all(0)
            t0 = time.perf_counter()
            try:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="English",
                    voice_clone_prompt=prompt,
                    **CANONICAL_KWARGS,
                )
            except Exception as e:
                print(f"  run {run_idx + 1} FAILED: {type(e).__name__}: {e}")
                results[bucket].append((0.0, "error", ""))
                continue
            total_ms = (time.perf_counter() - t0) * 1000

            # extract audio length + save WAV
            audio_secs = 0.0
            wav_path = ""
            try:
                wav_arr = wavs[0] if isinstance(wavs, list) else wavs
                if hasattr(wav_arr, "cpu"):
                    wav_arr = wav_arr.cpu().numpy()
                wav_arr = np.asarray(wav_arr).squeeze()
                audio_secs = len(wav_arr) / sr if sr > 0 else 0.0
                wav_path = os.path.join(WAV_OUT_DIR, f"vanilla_bucket{bucket:03d}_run{run_idx + 1}.wav")
                sf.write(wav_path, wav_arr, int(sr))
            except Exception as e:
                print(f"  WAV save failed: {e}")

            cls = classify(bucket, audio_secs)
            results[bucket].append((audio_secs, cls, wav_path))
            writer.writerow([bucket, len(text), run_idx + 1, f"{total_ms:.0f}",
                            f"{audio_secs:.3f}", cls, wav_path])
            csv_file.flush()
            print(f"  run {run_idx + 1}: total={total_ms:.0f} ms, audio={audio_secs:.3f}s [{cls}] -> {os.path.basename(wav_path)}")

    csv_file.close()

    # ------------------------------------------------------------------
    header("Comparison summary — vanilla qwen-tts vs FasterQwen3TTS (prior run)")
    # ------------------------------------------------------------------
    # FasterQwen3TTS results from previous run, hardcoded for comparison.
    fast_results: Final[dict[int, tuple[float, str]]] = {
        10: (10.480, "runaway"),
        40: (5.280, "expected"),
        70: (6.480, "expected"),
        100: (9.520, "expected"),
        130: (1.120, "truncated"),
        160: (6.880, "truncated"),
        190: (149.360, "runaway"),
        220: (149.360, "runaway"),
        250: (149.360, "runaway"),
    }

    print(f"  {'bucket':>6} {'vanilla_s':>10} {'vanilla':>11} {'fast_s':>10} {'fast':>11} {'agreement':>11}")
    same_bug_count = 0
    diff_count = 0
    for bucket, _ in SENTENCES:
        runs = results[bucket]
        if not runs:
            continue
        vanilla_s, vanilla_cls, _ = runs[0]
        fast_s, fast_cls = fast_results.get(bucket, (0.0, "?"))
        agreement = "SAME-BUG" if vanilla_cls == fast_cls and vanilla_cls != "expected" else (
            "BOTH-OK" if vanilla_cls == "expected" and fast_cls == "expected" else "DIFFER"
        )
        if agreement == "SAME-BUG":
            same_bug_count += 1
        if agreement == "DIFFER":
            diff_count += 1
        print(f"  {bucket:>6} {vanilla_s:>10.3f} {vanilla_cls:>11} {fast_s:>10.3f} {fast_cls:>11} {agreement:>11}")

    header("VERDICT")
    if diff_count == 0 and same_bug_count > 0:
        print(f"  Vanilla qwen-tts has the SAME failure pattern as FasterQwen3TTS.")
        print(f"  → The bug is at the MODEL level (not introduced by FasterQwen3TTS).")
        print(f"  → No upstream fix to FasterQwen3TTS will solve this.")
        print(f"  → Path forward: chunk text in the safe 40-100 char range,")
        print(f"    or accept Qwen3-TTS's broken edge cases, or pivot.")
    elif diff_count > 0 and same_bug_count == 0:
        print(f"  Vanilla qwen-tts is RELIABLE on lengths where FasterQwen3TTS fails.")
        print(f"  → The bug is in FasterQwen3TTS (its CUDA-graph wrapping).")
        print(f"  → Path forward: file upstream issue, OR fall back to vanilla")
        print(f"    on inputs outside the safe range.")
    else:
        print(f"  Mixed results: {same_bug_count} same-bug, {diff_count} differ.")
        print(f"  → Some failures are model-level, some are FasterQwen3TTS-specific.")
        print(f"  → Per-bucket analysis required.")

    print(f"\n  WAVs: {WAV_OUT_DIR}")
    print(f"  CSV:  {CSV_OUT}")


if __name__ == "__main__":
    main()
