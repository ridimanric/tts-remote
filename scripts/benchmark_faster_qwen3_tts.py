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
    header("Step 5: synthesis benchmark — default voice (no cloning)")
    # ------------------------------------------------------------------
    # Pick a sensible synthesis function. Most likely 'generate' or 'synthesize'.
    synth_fn_name = None
    for preferred in ("generate", "synthesize", "tts", "speak"):
        if preferred in {n for n, _ in synth_candidates}:
            synth_fn_name = preferred
            break

    if synth_fn_name is None and synth_candidates:
        synth_fn_name = synth_candidates[0][0]

    if synth_fn_name is None:
        print("  ERROR: could not auto-pick a synthesis function. Inspect class manually.")
        return

    print(f"  using engine.{synth_fn_name}(...)")
    synth_fn = getattr(engine, synth_fn_name)

    print(f"  warmup runs: {WARMUP_RUNS}")
    for i in range(WARMUP_RUNS):
        try:
            t0 = time.perf_counter()
            wav_out = synth_fn(TEST_SENTENCE)
            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"    warmup {i + 1}: {dt_ms:.0f} ms")
        except TypeError as e:
            print(f"    warmup {i + 1}: signature mismatch — {e}")
            print(f"    Trying with text= keyword:")
            try:
                t0 = time.perf_counter()
                wav_out = synth_fn(text=TEST_SENTENCE)
                dt_ms = (time.perf_counter() - t0) * 1000
                print(f"      kwarg form: {dt_ms:.0f} ms")
            except Exception as e2:
                print(f"      kwarg form also failed: {e2}")
                traceback.print_exc()
                return
        except Exception as e:
            print(f"    warmup {i + 1} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            return

    print(f"  measured runs: {MEASURED_RUNS}")
    measurements: list[float] = []
    for i in range(MEASURED_RUNS):
        try:
            t0 = time.perf_counter()
            wav_out = synth_fn(TEST_SENTENCE)
            dt_ms = (time.perf_counter() - t0) * 1000
            measurements.append(dt_ms)
            print(f"    call {i + 1}: {dt_ms:.0f} ms")
        except Exception as e:
            print(f"    call {i + 1} failed: {type(e).__name__}: {e}")
            return

    # Save the last output for ear-check
    try:
        import soundfile as sf
        import numpy as np
        if isinstance(wav_out, tuple) and len(wav_out) >= 2:
            wav_array, sr = wav_out[0], wav_out[1]
        elif isinstance(wav_out, dict):
            wav_array = wav_out.get("wav") or wav_out.get("audio") or wav_out.get("waveform")
            sr = wav_out.get("sample_rate") or wav_out.get("sr") or 24000
        else:
            wav_array = wav_out
            sr = 24000

        if hasattr(wav_array, "cpu"):
            wav_array = wav_array.cpu().numpy()
        wav_array = np.asarray(wav_array).squeeze()
        out_path = os.path.join(OUTPUT_DIR, "default_voice.wav")
        sf.write(out_path, wav_array, int(sr))
        print(f"  saved output: {out_path} ({len(wav_array)/sr:.2f}s @ {sr}Hz)")
    except Exception as e:
        print(f"  WAV save failed: {e}")

    # ------------------------------------------------------------------
    header("Step 6: voice cloning test (if supported)")
    # ------------------------------------------------------------------
    # Look for a cloning-capable method or a kwarg pattern.
    cloning_attempted = False
    for name, sig in synth_candidates:
        if not sig:
            continue
        params = list(sig.parameters.keys())
        if any(p in params for p in ("ref_audio", "voice", "speaker_audio", "audio_prompt")):
            cloning_attempted = True
            fn = getattr(engine, name)
            print(f"  trying {name} with ref_audio kwarg...")
            try:
                ref_path = REF_AUDIO_LOCAL if os.path.exists(REF_AUDIO_LOCAL) else REF_AUDIO_REMOTE
                t0 = time.perf_counter()
                # Try the most likely kwarg names
                for ref_kwarg in ("ref_audio", "voice", "speaker_audio", "audio_prompt"):
                    if ref_kwarg in params:
                        kwargs = {ref_kwarg: ref_path}
                        if "ref_text" in params:
                            kwargs["ref_text"] = REF_TEXT
                        wav_out = fn(TEST_SENTENCE, **kwargs)
                        break
                dt_ms = (time.perf_counter() - t0) * 1000
                print(f"    cloning call: {dt_ms:.0f} ms")
                # Save the cloned-voice output
                if isinstance(wav_out, tuple) and len(wav_out) >= 2:
                    wav_array, sr = wav_out[0], wav_out[1]
                elif isinstance(wav_out, dict):
                    wav_array = wav_out.get("wav") or wav_out.get("audio")
                    sr = wav_out.get("sample_rate") or wav_out.get("sr") or 24000
                else:
                    wav_array = wav_out
                    sr = 24000
                if hasattr(wav_array, "cpu"):
                    wav_array = wav_array.cpu().numpy()
                wav_array = np.asarray(wav_array).squeeze()
                out_path = os.path.join(OUTPUT_DIR, "cloned_voice.wav")
                sf.write(out_path, wav_array, int(sr))
                print(f"    saved cloned output: {out_path}")
            except Exception as e:
                print(f"    cloning call failed: {type(e).__name__}: {e}")
                traceback.print_exc()
            break

    if not cloning_attempted:
        print("  no cloning kwarg found in any candidate. Possibilities:")
        print("    - cloning is via a separate prime/load-voice method")
        print("    - cloning uses a different mechanism we missed")
        print("  Inspect manually:")
        for name, sig in synth_candidates:
            print(f"    {name}{sig}")

    # ------------------------------------------------------------------
    header("Step 7: summary")
    # ------------------------------------------------------------------
    if measurements:
        sorted_m = sorted(measurements)
        p50 = sorted_m[len(sorted_m) // 2]
        print(f"  default-voice synthesis on L4:")
        print(f"    per-call ms: {[f'{m:.0f}' for m in measurements]}")
        print(f"    p50: {p50:.0f} ms")
        print(f"    min: {min(measurements):.0f} ms | max: {max(measurements):.0f} ms")
        print()
        print(f"  vanilla qwen-tts baseline (10s) → {p50:.0f} ms = {(10000 - p50) / 10000 * 100:.0f}% reduction")
        print(f"  target (700 ms p95): {'PASS' if max(measurements) <= 700 else 'NOT YET'}")
        print()
        print(f"  Output WAVs at {OUTPUT_DIR} for ear-check.")
        print(f"  Compare against vanilla qwen-tts output to verify quality parity.")


if __name__ == "__main__":
    main()
