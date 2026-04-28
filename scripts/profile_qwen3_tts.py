"""
Profile Qwen3-TTS synthesis to find where the ~10s/call goes.

Per the investigation plan
(anti-voice/docs/qwen-tts/2026-04-21-performance-investigation-plan.md
Phase 1) and the internals reference
(anti-voice/docs/qwen-tts/2026-04-28-qwen-tts-internals-reference.md), we
expect the dominant phases to be:
  F. self.model.generate  — autoregressive talker decode
  H. self.model.speech_tokenizer.decode  — speech codes -> waveform
Everything else should be plumbing.

We measure TWO modes:
  - prod_equivalent: pass ref_audio every call (re-extracts speaker embedding
    each time). This mirrors anti-voice prod.
  - prompt_cached: pre-extract voice_clone_prompt once, reuse it. Tells us
    the upper bound of what a "cache the prompt" optimisation can save.

For each mode: 1 warmup + 3 measured calls. Phase timings are accumulated
across the 3 measured calls and reported as totals + per-call averages.

Also runs torch.profiler over ONE call (in prod_equivalent mode, since that's
what prod does) and exports a Chrome trace to traces/.

Run from /workspace/tts-remote:
    .venv/bin/python scripts/profile_qwen3_tts.py
"""
import functools
import os
import time
from collections import defaultdict
from typing import Any, Callable, Final


TEST_SENTENCE: Final[str] = "Hello, this is a test of Qwen three TTS latency."
WARMUP_RUNS: Final[int] = 1
MEASURED_RUNS: Final[int] = 3
REF_AUDIO: Final[str] = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)
TRACE_DIR: Final[str] = "traces"


# accumulated phase timings, keyed by mode then phase
_timings: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
_current_mode: str = "uninitialised"


def _gpu_sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _wrap_method(obj: Any, attr: str, phase_name: str) -> Callable[[], None]:
    """Replace obj.attr with a timed wrapper. Returns an unwrap callable."""
    original = getattr(obj, attr)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _gpu_sync()
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _gpu_sync()
            dt_ms = (time.perf_counter() - t0) * 1000
            _timings[_current_mode][phase_name].append(dt_ms)

    setattr(obj, attr, wrapper)
    return lambda: setattr(obj, attr, original)


def _print_breakdown(mode: str) -> None:
    print()
    print(f"--- {mode} ---")
    rows: list[tuple[str, float, float, int]] = []
    for phase, samples in _timings[mode].items():
        total = sum(samples)
        n = len(samples)
        avg = total / n if n else 0.0
        rows.append((phase, total, avg, n))
    rows.sort(key=lambda r: -r[1])

    width_phase = max(len(r[0]) for r in rows) if rows else 20
    print(f"  {'phase'.ljust(width_phase)}  {'total ms':>10}  {'avg ms':>10}  {'calls':>6}")
    for phase, total, avg, n in rows:
        print(f"  {phase.ljust(width_phase)}  {total:>10.0f}  {avg:>10.0f}  {n:>6d}")


def _run_warmup_and_measure(
    model: Any,
    use_cached_prompt: bool,
    cached_prompt: Any,
    mode_name: str,
) -> list[float]:
    global _current_mode
    _current_mode = mode_name

    print(f"\n=== Mode: {mode_name} ===")
    print(f"  warmup runs: {WARMUP_RUNS}")
    for i in range(WARMUP_RUNS):
        t0 = time.perf_counter()
        if use_cached_prompt:
            wavs, sr = model.generate_voice_clone(
                text=TEST_SENTENCE,
                language="English",
                voice_clone_prompt=cached_prompt,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=TEST_SENTENCE,
                language="English",
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
            )
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"    warmup {i + 1}: {dt_ms:.0f} ms")

    # Discard warmup timings — clear the lists.
    for phase in list(_timings[mode_name].keys()):
        _timings[mode_name][phase].clear()

    print(f"  measured runs: {MEASURED_RUNS}")
    measurements: list[float] = []
    for i in range(MEASURED_RUNS):
        t0 = time.perf_counter()
        if use_cached_prompt:
            wavs, sr = model.generate_voice_clone(
                text=TEST_SENTENCE,
                language="English",
                voice_clone_prompt=cached_prompt,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=TEST_SENTENCE,
                language="English",
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
            )
        dt_ms = (time.perf_counter() - t0) * 1000
        measurements.append(dt_ms)
        print(f"    call {i + 1}: {dt_ms:.0f} ms")
    return measurements


def _profile_one_call(model: Any) -> str:
    """Run a single call under torch.profiler. Returns trace path."""
    import torch
    from torch.profiler import profile, ProfilerActivity

    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = os.path.join(TRACE_DIR, "qwen3_prod_equivalent_call.json")
    print(f"\n=== torch.profiler trace (one prod_equivalent call) ===")
    print(f"  saving to: {trace_path}")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        model.generate_voice_clone(
            text=TEST_SENTENCE,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
        )

    prof.export_chrome_trace(trace_path)

    print()
    print("  Top 20 ops by self CUDA time:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
        row_limit=20,
    ))
    return trace_path


def main() -> None:
    print(f"Test sentence ({len(TEST_SENTENCE)} chars): {TEST_SENTENCE!r}")
    import torch
    from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]

    print("\nLoading Qwen3-TTS (flash_attention_2, bf16, cuda:0)...")
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f} ms")

    # Hook the four phases we care about.
    # Per the internals doc: generate_voice_clone is the outer wall, create_voice_clone_prompt
    # is the per-call extraction (when no cached prompt), self.model.generate is the
    # autoregressive talker decode, self.model.speech_tokenizer.decode is the wav decoder.
    print("\nInstalling phase hooks (generate_voice_clone, create_voice_clone_prompt, "
          "model.generate, speech_tokenizer.decode)...")
    _wrap_method(type(model), "generate_voice_clone", "outer:generate_voice_clone")
    _wrap_method(type(model), "create_voice_clone_prompt", "B:create_voice_clone_prompt")
    _wrap_method(model.model, "generate", "F:talker_generate")
    _wrap_method(model.model.speech_tokenizer, "decode", "H:speech_tokenizer_decode")

    # Pre-extract a voice_clone_prompt for the cached-mode test.
    print("\nPre-extracting voice_clone_prompt for cached-mode comparison...")
    t0 = time.perf_counter()
    cached_prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
    )
    cache_ms = (time.perf_counter() - t0) * 1000
    print(f"voice_clone_prompt extraction: {cache_ms:.0f} ms (one-off cost)")
    # Drop the timings captured during cache-prep so the per-mode reports are clean.
    _timings.clear()

    # Mode 1: prod-equivalent (re-extract every call).
    prod_measurements = _run_warmup_and_measure(
        model, use_cached_prompt=False, cached_prompt=None,
        mode_name="prod_equivalent",
    )

    # Mode 2: pre-extracted prompt cached.
    cached_measurements = _run_warmup_and_measure(
        model, use_cached_prompt=True, cached_prompt=cached_prompt,
        mode_name="prompt_cached",
    )

    # Phase breakdowns
    _print_breakdown("prod_equivalent")
    _print_breakdown("prompt_cached")

    # Comparative summary
    print("\n=== Comparative summary ===")
    pe_mean = sum(prod_measurements) / len(prod_measurements)
    pc_mean = sum(cached_measurements) / len(cached_measurements)
    print(f"  prod_equivalent mean: {pe_mean:.0f} ms")
    print(f"  prompt_cached  mean: {pc_mean:.0f} ms")
    print(f"  delta: {pe_mean - pc_mean:.0f} ms ({(pe_mean - pc_mean) / pe_mean * 100:.1f}% would be saved by prompt caching alone)")
    print(f"  target: < 700 ms p95")
    print(f"  gap from prompt_cached mean to target: {pc_mean - 700:.0f} ms")

    # torch.profiler trace for one prod-equivalent call.
    trace_path = _profile_one_call(model)
    print(f"\nChrome trace at: {trace_path}")
    print("Open in https://ui.perfetto.dev/ or chrome://tracing for kernel-level analysis.")


if __name__ == "__main__":
    main()
