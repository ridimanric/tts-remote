"""
Combined inspector + benchmark for andimarafioti/faster-qwen3-tts.

Phase 3.D — pivoted from vLLM port to FasterQwen3TTS adoption evaluation.

This script:
  1. Introspects FasterQwen3TTS class methods (factory, synthesis, cloning)
  2. Constructs an instance via the discovered API
  3. Runs the same 48-char benchmark sentence we used for the 10s baseline
     (1 warmup + 3 measured runs)
  4. Saves output WAVs for ear-check vs vanilla qwen-tts output
  5. If voice cloning is supported, tests cloning with the same reference
     audio used in our profile_qwen3_tts.py

Run from /workspace/tts-remote (after setup_faster_qwen3_spike.sh):
    /workspace/faster-qwen3-tts/.venv/bin/python scripts/benchmark_faster_qwen3_tts.py

Outputs:
  - Per-run synthesis_ms (for direct comparison to 10s baseline)
  - WAVs at /workspace/tts-remote/traces/faster_qwen3_outputs/
  - Voice-cloning test WAV (if cloning works)
"""
import inspect
import os
import sys
import time
import traceback
from typing import Any, Final

REPO_PATH: Final[str] = "/workspace/faster-qwen3-tts"
TEST_SENTENCE: Final[str] = "Hello, this is a test of Qwen three TTS latency."
REF_AUDIO_LOCAL: Final[str] = "/workspace/faster-qwen3-tts/ref_audio.wav"
REF_AUDIO_REMOTE: Final[str] = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)
WARMUP_RUNS: Final[int] = 1
MEASURED_RUNS: Final[int] = 3
OUTPUT_DIR: Final[str] = "/workspace/tts-remote/traces/faster_qwen3_outputs"


def header(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> None:
    if not os.path.isdir(REPO_PATH):
        print(f"ERROR: {REPO_PATH} not found. Run setup_faster_qwen3_spike.sh first.")
        return

    sys.path.insert(0, REPO_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    header("Step 1: introspect FasterQwen3TTS class")
    # ------------------------------------------------------------------
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"FasterQwen3TTS source: {inspect.getsourcefile(FasterQwen3TTS)}")
    print()
    print("Public methods:")
    for name in sorted(dir(FasterQwen3TTS)):
        if name.startswith("_") and name != "__init__":
            continue
        attr = getattr(FasterQwen3TTS, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                print(f"  {name}{sig}")
            except (ValueError, TypeError):
                print(f"  {name}(?)")

    # ------------------------------------------------------------------
    header("Step 2: locate factory / construction pattern")
    # ------------------------------------------------------------------
    factory_methods = [
        name for name in dir(FasterQwen3TTS)
        if name.startswith("from_") and callable(getattr(FasterQwen3TTS, name))
    ]
    print(f"  factory methods: {factory_methods}")

    # ------------------------------------------------------------------
    header("Step 3: construct an instance")
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    if "from_pretrained" in factory_methods:
        print("  using FasterQwen3TTS.from_pretrained(...)")
        try:
            engine = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        except Exception as e:
            print(f"  from_pretrained failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            return
    else:
        print("  no from_pretrained — falling back to manual construction")
        print("  (loading qwen-tts base model + building graphs)")
        try:
            from qwen_tts import Qwen3TTSModel
            from faster_qwen3_tts.talker_graph import TalkerGraph  # type: ignore[import-not-found]
            from faster_qwen3_tts.predictor_graph import PredictorGraph  # type: ignore[import-not-found]

            base_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )
            talker_graph = TalkerGraph(base_model.model.talker)
            predictor_graph = PredictorGraph(base_model.model.talker.code_predictor)
            engine = FasterQwen3TTS(
                base_model=base_model,
                predictor_graph=predictor_graph,
                talker_graph=talker_graph,
            )
        except Exception as e:
            print(f"  manual construction failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            return

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  engine constructed in {load_ms:.0f} ms")
    print(f"  type: {type(engine).__name__}")

    # ------------------------------------------------------------------
    header("Step 4: identify the synthesis call")
    # ------------------------------------------------------------------
    synth_candidates = []
    for name in dir(engine):
        if name.startswith("_"):
            continue
        attr = getattr(engine, name)
        if not callable(attr):
            continue
        lower = name.lower()
        if any(k in lower for k in ("generate", "synthesize", "tts", "infer", "speak")):
            try:
                sig = inspect.signature(attr)
            except (ValueError, TypeError):
                sig = None
            synth_candidates.append((name, sig))

    print(f"  synthesis-call candidates ({len(synth_candidates)}):")
    for name, sig in synth_candidates:
        print(f"    {name}{sig}")

    # ------------------------------------------------------------------
    header("Step 5: synthesis benchmark — voice cloning (our prod use case)")
    # ------------------------------------------------------------------
    # FasterQwen3TTS.generate() raises NotImplementedError for default voice.
    # Our actual use case in anti-voice prod IS voice cloning (every session
    # has a primed voice profile), so generate_voice_clone is what we benchmark.
    # This also exercises the CUDA-graph fast path the same way prod would.
    import soundfile as sf
    import numpy as np

    if "generate_voice_clone" not in dir(engine):
        print("  ERROR: generate_voice_clone not available on engine.")
        return

    ref_path = REF_AUDIO_LOCAL if os.path.exists(REF_AUDIO_LOCAL) else REF_AUDIO_REMOTE
    print(f"  ref_audio: {ref_path}")
    print(f"  ref_text:  {REF_TEXT[:60]}...")

    synth_fn = engine.generate_voice_clone

    def _save_wav(out_obj: Any, path: str) -> None:
        """Save a synthesizer output (tuple/list/dict/array) as a WAV."""
        if isinstance(out_obj, tuple) and len(out_obj) >= 2:
            wav_array, sr = out_obj[0], out_obj[1]
        elif isinstance(out_obj, dict):
            wav_array = out_obj.get("wav") or out_obj.get("audio") or out_obj.get("waveform")
            sr = out_obj.get("sample_rate") or out_obj.get("sr") or 24000
        else:
            wav_array = out_obj
            sr = 24000
        # generate_voice_clone returns Tuple[list, int] per the signature —
        # the list contains numpy arrays. Take the first.
        if isinstance(wav_array, list) and len(wav_array) > 0:
            wav_array = wav_array[0]
        if hasattr(wav_array, "cpu"):
            wav_array = wav_array.cpu().numpy()
        wav_array = np.asarray(wav_array).squeeze()
        sf.write(path, wav_array, int(sr))
        return wav_array, sr

    print(f"  warmup runs: {WARMUP_RUNS}")
    for i in range(WARMUP_RUNS):
        try:
            t0 = time.perf_counter()
            wav_out = synth_fn(
                text=TEST_SENTENCE,
                language="English",
                ref_audio=ref_path,
                ref_text=REF_TEXT,
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"    warmup {i + 1}: {dt_ms:.0f} ms")
        except Exception as e:
            print(f"    warmup {i + 1} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return

    print(f"  measured runs: {MEASURED_RUNS}")
    measurements: list[float] = []
    last_wav_out = None
    for i in range(MEASURED_RUNS):
        try:
            t0 = time.perf_counter()
            wav_out = synth_fn(
                text=TEST_SENTENCE,
                language="English",
                ref_audio=ref_path,
                ref_text=REF_TEXT,
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            measurements.append(dt_ms)
            last_wav_out = wav_out
            print(f"    call {i + 1}: {dt_ms:.0f} ms")
        except Exception as e:
            print(f"    call {i + 1} FAILED: {type(e).__name__}: {e}")
            return

    # Save the last cloned output for ear-check
    try:
        out_path = os.path.join(OUTPUT_DIR, "cloned_voice.wav")
        wav_array, sr = _save_wav(last_wav_out, out_path)
        print(f"  saved cloned output: {out_path} ({len(wav_array)/sr:.2f}s @ {sr}Hz)")
    except Exception as e:
        print(f"  WAV save failed: {e}")

    # ------------------------------------------------------------------
    header("Step 6: streaming variant — measure TTFA")
    # ------------------------------------------------------------------
    # The streaming generator yields chunks. Measure time-to-first-audio
    # (first chunk yielded) — that's the user-perceived latency for a
    # voice agent (audio starts playing as soon as first chunk arrives).
    if "generate_voice_clone_streaming" not in dir(engine):
        print("  generate_voice_clone_streaming not available; skipping.")
    else:
        print("  measuring TTFA via generate_voice_clone_streaming...")
        stream_fn = engine.generate_voice_clone_streaming
        try:
            t0 = time.perf_counter()
            gen = stream_fn(
                text=TEST_SENTENCE,
                language="English",
                ref_audio=ref_path,
                ref_text=REF_TEXT,
            )
            first_chunk = next(gen)
            ttfa_ms = (time.perf_counter() - t0) * 1000
            print(f"    TTFA (time to first audio chunk): {ttfa_ms:.0f} ms")
            # Drain the rest to get total
            chunk_count = 1
            for _ in gen:
                chunk_count += 1
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"    total streaming time: {total_ms:.0f} ms ({chunk_count} chunks)")
        except Exception as e:
            print(f"    streaming benchmark failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    header("Step 7: summary")
    # ------------------------------------------------------------------
    if measurements:
        sorted_m = sorted(measurements)
        p50 = sorted_m[len(sorted_m) // 2]
        print(f"  voice-cloning synthesis on L4 (FasterQwen3TTS):")
        print(f"    per-call ms: {[f'{m:.0f}' for m in measurements]}")
        print(f"    p50: {p50:.0f} ms")
        print(f"    min: {min(measurements):.0f} ms | max: {max(measurements):.0f} ms")
        print()
        print(f"  vanilla qwen-tts baseline (10s) → p50 {p50:.0f} ms = {(10000 - p50) / 10000 * 100:.0f}% reduction")
        print(f"  target (700 ms p95): {'PASS' if max(measurements) <= 700 else 'NOT YET'}")
        print()
        print(f"  Output WAVs at {OUTPUT_DIR} for ear-check.")
        print(f"  Compare against vanilla qwen-tts output to verify quality parity.")


if __name__ == "__main__":
    main()
