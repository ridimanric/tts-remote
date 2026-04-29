"""
Audit Qwen3-TTS for CUDA-graph capture compatibility.

Per the optimisation sequence Phase 2 Experiment 3, before implementing
CUDA-graph replay we want to know whether the inner transformer's forward
pass can be captured cleanly. Capture fails on:
  - CPU/GPU sync calls (.item(), .cpu(), .tolist()) inside the region
  - CPU-side randomness or non-CUDA RNG
  - Dynamic memory allocation that wasn't pre-allocated
  - cudaMemcpy with implicit host involvement
  - Some kernel types (e.g. cuBLAS GEMV in some configurations)

Strategy:
  1. Load the model and pre-extract a voice_clone_prompt.
  2. Run a real generate call with a forward-hook that records realistic
     inputs (single decode step) to memory.
  3. Replay one forward call eagerly with those inputs to establish a
     reference output.
  4. Attempt to capture that same forward call inside `torch.cuda.graph()`.
  5. Replay the captured graph and compare its output to the eager
     reference; report any divergence.

Output:
  - Success → CUDA graph capture is viable; report capture wall-time and
    replay wall-time; replay should be ~10-50x faster than eager for the
    same forward pass.
  - Failure → specific exception caught during capture; classify the
    failure mode and recommend next steps.

Run from /workspace/tts-remote:
    uv run python scripts/inspect_qwen_tts_cudagraph.py
"""
import time
import traceback
from typing import Any, Final

REF_AUDIO: Final[str] = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)
TEST_SENTENCE: Final[str] = "Hello, this is a test of Qwen three TTS latency."


def header(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


# Captured one-step forward args/kwargs from the hooked generate call.
_captured_args: tuple | None = None
_captured_kwargs: dict[str, Any] | None = None
_capture_count: int = 0


def main() -> None:
    import torch
    from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. CUDA-graph capture requires a GPU.")
        return

    header("Step 1: load model")
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Loaded in {load_ms:.0f} ms")

    inner_model = model.model
    print(f"inner model class: {type(inner_model).__name__}")
    fwd_fn_qual = (
        f"{type(inner_model).__module__}."
        f"{type(inner_model).__name__}.forward"
    )
    print(f"target forward: {fwd_fn_qual}")

    header("Step 2a: enumerate submodules to find the per-decode-step forward")

    # Print top-level submodules to identify candidates (talker, code_predictor,
    # speech_tokenizer, etc.).
    print("  top-level submodules of inner_model:")
    for name, sub in inner_model.named_children():
        n_params = sum(p.numel() for p in sub.parameters())
        print(f"    {name:30s}  {type(sub).__name__:30s}  params={n_params:>12,}")

    # Register a forward hook on EVERY named module to record call counts during
    # a short generate. The module called the most times during decode is the
    # one we want to capture for the autoregressive loop.
    call_counts: dict[str, int] = {}
    handles: list[Any] = []

    def make_counter(qualname: str) -> Any:
        def hook(_module: Any, _args: tuple, _output: Any) -> None:
            call_counts[qualname] = call_counts.get(qualname, 0) + 1
        return hook

    for name, sub in inner_model.named_modules():
        if name == "":
            continue
        handles.append(sub.register_forward_hook(make_counter(name)))

    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
        )
        print()
        print("  running short generate to count per-module forward calls (max_new_tokens=8)...")
        t0 = time.perf_counter()
        _ = model.generate_voice_clone(
            text=TEST_SENTENCE,
            language="English",
            voice_clone_prompt=prompt,
            max_new_tokens=8,
        )
        gen_ms = (time.perf_counter() - t0) * 1000
        print(f"  short generate finished in {gen_ms:.0f} ms")
    finally:
        for h in handles:
            h.remove()

    # Sort by call count, show top 20 most-called modules.
    print()
    print("  top 20 modules by forward-call count during 8-token generate:")
    print(f"  {'count':>7}  module")
    for qualname, count in sorted(call_counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {count:>7d}  {qualname}")

    # We need a capture target that:
    #   1. Sits BELOW transformers' create_causal_mask path (avoids the
    #      `attention_mask.all()` host-sync that breaks capture).
    #   2. Has stable inputs across calls (a single decoder layer fits — its
    #      inputs are hidden_states + already-computed mask + position_ids).
    #
    # The first decoder layer of the inner code_predictor model satisfies
    # both. From the call counts above we expect ~13 calls per outer token.
    # If this layer can be captured cleanly, we've proven the method works
    # and can apply it broadly to all layers.
    decode_target_name = "talker.code_predictor.model.layers.0"
    if decode_target_name not in dict(inner_model.named_modules()):
        print()
        print(f"  ERROR: expected target module '{decode_target_name}' not found.")
        print(f"  Module structure may have changed; inspect call_counts above and")
        print(f"  pick a layer-level module manually.")
        return
    decode_target_count = call_counts.get(decode_target_name, 0)
    print()
    print(f"  decode-step capture target: '{decode_target_name}' ({decode_target_count} calls during 8-token generate)")
    print(f"  This is a single decoder layer — sits below mask creation, so")
    print(f"  the `attention_mask.all()` blocker in transformers/masking_utils.py")
    print(f"  is not in this capture path.")
    decode_target = dict(inner_model.named_modules())[decode_target_name]

    header("Step 2b: pre-hook the decode-step module to capture realistic inputs")

    # We use register_forward_pre_hook with_kwargs=True instead of replacing
    # decode_target.forward. Replacing forward changes its signature to
    # (*args, **kwargs), which breaks HF transformers' generation kwargs
    # validation (it inspects forward's signature to know which kwargs are
    # valid).
    def pre_hook(_module: Any, args: tuple, kwargs: dict) -> None:
        global _captured_args, _captured_kwargs, _capture_count
        _capture_count += 1
        # Capture the SECOND call (first is prefill; second is the steady-
        # state decode step we want to capture).
        if _capture_count == 2 and _captured_args is None:
            def _clone(x: Any) -> Any:
                if isinstance(x, torch.Tensor):
                    return x.detach().clone()
                if isinstance(x, (list, tuple)):
                    return type(x)(_clone(v) for v in x)
                if isinstance(x, dict):
                    return {k: _clone(v) for k, v in x.items()}
                return x
            _captured_args = _clone(args)
            _captured_kwargs = _clone(kwargs)
            print(f"  captured step #{_capture_count} forward inputs from '{decode_target_name}'")
            print(f"    args: {len(args)} positional")
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    print(f"      [{i}] tensor shape={tuple(a.shape)} dtype={a.dtype} device={a.device}")
                else:
                    print(f"      [{i}] type={type(a).__name__}")
            print(f"    kwargs: {sorted(kwargs.keys())}")

    handle = decode_target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        print("  running short generate to capture step inputs (max_new_tokens=8)...")
        t0 = time.perf_counter()
        _ = model.generate_voice_clone(
            text=TEST_SENTENCE,
            language="English",
            voice_clone_prompt=prompt,
            max_new_tokens=8,
        )
        gen_ms = (time.perf_counter() - t0) * 1000
        print(f"  short generate finished in {gen_ms:.0f} ms; total target calls: {_capture_count}")
    finally:
        handle.remove()

    if _captured_args is None:
        print("ERROR: pre-hook on decode_target was not invoked. Cannot proceed.")
        return

    original_forward = decode_target.forward

    header("Step 3: replay one forward eagerly to establish reference output")
    torch.cuda.synchronize()
    with torch.no_grad():
        try:
            t0 = time.perf_counter()
            ref_output = original_forward(*_captured_args, **_captured_kwargs)
            torch.cuda.synchronize()
            ref_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"ERROR: eager replay failed before we even tried capture:")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            return
    print(f"  eager forward (single step): {ref_ms:.2f} ms")
    print(f"  eager output type: {type(ref_output).__name__}")

    header("Step 4: attempt CUDA graph capture of one forward call")
    print("  Running 3 warmup forwards on a side stream (CUDA graph requires this)...")

    # CUDA graph capture requires:
    #   1. A separate stream
    #   2. Warmup so cuBLAS/cuDNN kernels have selected their algorithms
    #   3. Static input addresses (we'll use the captured tensors as the
    #      static buffers — replay copies new inputs INTO these buffers)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            try:
                for i in range(3):
                    _ = original_forward(*_captured_args, **_captured_kwargs)
            except Exception as e:
                print(f"  ERROR during warmup: {type(e).__name__}: {e}")
                traceback.print_exc()
                return
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print("  warmup done.")

    print("  Capturing graph...")
    g = torch.cuda.CUDAGraph()
    capture_error: Exception | None = None
    capture_output: Any = None
    try:
        with torch.cuda.graph(g):
            with torch.no_grad():
                capture_output = original_forward(*_captured_args, **_captured_kwargs)
    except Exception as e:
        capture_error = e

    if capture_error is not None:
        print(f"  CAPTURE FAILED: {type(capture_error).__name__}: {capture_error}")
        print()
        traceback.print_exception(type(capture_error), capture_error, capture_error.__traceback__)
        print()
        header("Step 5: verdict")
        msg = str(capture_error)
        if "stream" in msg.lower() or "memcpy" in msg.lower():
            kind = "Stream / cudaMemcpy issue"
        elif "alloc" in msg.lower() or "memory" in msg.lower():
            kind = "Dynamic allocation inside captured region"
        elif ".item" in msg.lower() or "synchronize" in msg.lower():
            kind = "CPU/GPU sync inside captured region"
        else:
            kind = "Other (see traceback above)"
        print(f"  CUDA graph capture is NOT viable as-is.")
        print(f"  Failure category: {kind}")
        print(f"  Next step: identify and remove the offending op from the")
        print(f"  forward path, OR pivot to Experiment 4 (vLLM port).")
        return

    print("  Capture succeeded.")
    header("Step 5: replay captured graph, time it, compare output to eager")

    # Replay: tensors in _captured_args/_captured_kwargs are the "static"
    # input buffers. To use the graph with new inputs, you'd write into
    # those same buffers and call g.replay(). For this inspector we just
    # validate that replay works at all and produces the same output.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - t0) * 1000
    print(f"  graph replay (single step): {replay_ms:.3f} ms")
    print(f"  eager forward (reference):  {ref_ms:.3f} ms")
    speedup = ref_ms / replay_ms if replay_ms > 0 else float("inf")
    print(f"  speedup: {speedup:.1f}x")

    # Compare outputs (best-effort — handle nested structures)
    def _first_tensor(obj: Any) -> torch.Tensor | None:
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = _first_tensor(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for v in obj.values():
                t = _first_tensor(v)
                if t is not None:
                    return t
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                t = _first_tensor(v)
                if t is not None:
                    return t
        return None

    ref_tensor = _first_tensor(ref_output)
    cap_tensor = _first_tensor(capture_output)
    if ref_tensor is not None and cap_tensor is not None:
        if ref_tensor.shape != cap_tensor.shape:
            print(f"  output shape MISMATCH: eager={ref_tensor.shape}, captured={cap_tensor.shape}")
        else:
            max_abs = (ref_tensor.float() - cap_tensor.float()).abs().max().item()
            print(f"  output max abs diff (eager vs captured): {max_abs:.6f}")
            if max_abs > 1e-2:
                print(f"  WARNING: numerical divergence > 1e-2. Capture may be unsound.")
            else:
                print(f"  outputs match within tolerance.")
    else:
        print("  could not extract a tensor from outputs for comparison.")

    header("Step 6: verdict")
    print(f"  CUDA graph capture is VIABLE for this forward pass.")
    print(f"  Single-step replay speedup: {speedup:.1f}x vs eager.")
    print(f"  Next step: implement static-shape KV cache + per-step replay")
    print(f"  in tts-remote, then re-benchmark with profile_qwen3_tts.py.")


if __name__ == "__main__":
    main()
