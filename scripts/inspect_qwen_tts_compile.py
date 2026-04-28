"""
Audit Qwen3-TTS for torch.compile compatibility.

Per the optimisation sequence Phase 2 Experiment 2, before committing to
torch.compile we want to know whether the talker model has compile-blockers
(`.item()` calls in hot loops, Python branches on tensor values, dynamic
shapes that force recompilation each step). PocketTTS hit these and was
incompatible — Qwen3-TTS may or may not have the same issues.

This script uses `torch._dynamo.explain()` to attempt compilation of the
underlying transformer's forward pass and report every graph break with
its reason + source location. Empirical, no source-code grepping.

Compatibility shim: transformers 4.57.3 calls `torch.compiler.is_exporting()`
which doesn't exist in torch 2.6.0 (added in 2.7+). Without a shim, dynamo
errors before tracing any model code. We patch it to return a constant
False (semantically correct — we're not in torch.export() mode here).

The shim COULD theoretically affect graph capture if transformers used
`is_exporting()` in a way that should sometimes return True. To detect
this, we install the patched function with a call counter + first-call
traceback dump. After explain, we print how many times it was invoked
and warn loudly if invocations occurred — so silent semantic drift
cannot happen unnoticed.

Run from /workspace/tts-remote:
    .venv/bin/python scripts/inspect_qwen_tts_compile.py
"""
import time
import traceback
from typing import Final

REF_AUDIO: Final[str] = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT: Final[str] = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it!"
)
TEST_SENTENCE: Final[str] = "Hello, this is a test of Qwen three TTS latency."


# Counter + traceback samples for the is_exporting shim.
_is_exporting_calls: int = 0
_is_exporting_first_traceback: str | None = None


def _patched_is_exporting() -> bool:
    """Shim for torch.compiler.is_exporting (missing in torch 2.6)."""
    global _is_exporting_calls, _is_exporting_first_traceback
    _is_exporting_calls += 1
    if _is_exporting_first_traceback is None:
        # Capture the first call site for diagnostics.
        _is_exporting_first_traceback = "".join(traceback.format_stack(limit=8))
    return False


def install_compat_shim() -> None:
    """Install torch.compiler.is_exporting if missing. Tracks calls."""
    import torch.compiler
    if hasattr(torch.compiler, "is_exporting"):
        print("torch.compiler.is_exporting already exists — no shim needed.")
        return
    torch.compiler.is_exporting = _patched_is_exporting
    print("Installed torch.compiler.is_exporting shim (returns False, counts calls).")


def header(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> None:
    # Shim must be installed before transformers (or any consumer) imports.
    import torch
    install_compat_shim()

    from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]
    import torch._dynamo as dynamo

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
    print(f"inner model module path: {type(inner_model).__module__}")

    header("Step 2: build a synthetic forward-pass input via voice_clone_prompt")
    prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
    )
    print(f"prompt items: {len(prompt)} (used to seed inner-model forward)")

    header("Step 3: dynamo.explain on a single forward through generate")
    print("Running explain(...) — this will execute one full generation under")
    print("dynamo's analyser. Slower than normal generate; expect ~30-60 s.")
    print()

    def _generation_call() -> object:
        return model.generate_voice_clone(
            text=TEST_SENTENCE,
            language="English",
            voice_clone_prompt=prompt,
        )

    t0 = time.perf_counter()
    explanation: object | None = None
    explain_error: Exception | None = None
    try:
        explanation = dynamo.explain(_generation_call)()
    except Exception as e:
        explain_error = e
    explain_ms = (time.perf_counter() - t0) * 1000
    print(f"explain() finished in {explain_ms:.0f} ms")

    header("Step 3b: shim invocation report")
    print(f"  torch.compiler.is_exporting was invoked: {_is_exporting_calls} times")
    if _is_exporting_calls == 0:
        print("  Shim was never called during explain. Patch is moot — results")
        print("  are equivalent to running on a torch version with native is_exporting.")
    else:
        print()
        print("  WARNING: shim was invoked. Our patch returns constant False.")
        print("  This is semantically correct for non-export use cases, but if any")
        print("  call site expected a different answer, the trace may differ from")
        print("  what real torch.compile would produce.")
        print()
        print("  First invocation traceback:")
        if _is_exporting_first_traceback:
            for line in _is_exporting_first_traceback.splitlines():
                print(f"    {line}")
        print()
        print("  Verify: each call site should be control flow that decides between")
        print("  'real torch.compile path' (False -> our path) and 'torch.export()")
        print("  path' (True -> not us). If yes, the trace is reliable. If a call")
        print("  site flips behaviour based on this for compile users specifically,")
        print("  re-run after upgrading transformers or torch.")

    if explain_error is not None:
        header("Step 4: explain() raised — cannot produce graph-break report")
        print(f"  {type(explain_error).__name__}: {explain_error}")
        print()
        print("  This is a different failure than the version-mismatch we patched.")
        print("  Read the error above to diagnose.")
        return

    header("Step 4: graph break report")
    breaks = getattr(explanation, "break_reasons", []) or []
    op_count = getattr(explanation, "op_count", "?")
    graph_count = getattr(explanation, "graph_count", "?")
    graph_break_count = getattr(explanation, "graph_break_count", "?")

    print(f"  total ops captured:        {op_count}")
    print(f"  total subgraphs:           {graph_count}")
    print(f"  total graph breaks:        {graph_break_count}")
    print()

    if not breaks:
        print("  No graph breaks reported. Model traces cleanly.")
    else:
        print(f"  Graph breaks (showing first 30):")
        for i, br in enumerate(breaks[:30], 1):
            reason = getattr(br, "reason", str(br))
            user_stack = getattr(br, "user_stack", None)
            print(f"  [{i}] {reason}")
            if user_stack:
                for frame in user_stack[:3]:
                    print(f"        at {frame}")

    header("Step 5: verdict")
    n = graph_break_count if isinstance(graph_break_count, int) else len(breaks)
    if n == 0:
        verdict = "CLEAN — torch.compile should work without modification."
    elif n < 10:
        verdict = (
            f"MINOR ({n} breaks) — torch.compile may give partial speedup. "
            "Each break flushes the JIT and reverts to eager for that section. "
            "Inspect the breaks above; if they're in cold paths, proceed. "
            "If they're per-decode-step in hot loops, fixing them is required."
        )
    elif n < 50:
        verdict = (
            f"MODERATE ({n} breaks) — partial speedup at best. "
            "Worth a quick spike to measure, but expect significant graph-break "
            "tax. If breaks are per-step, mostly negates the win."
        )
    else:
        verdict = (
            f"FUNDAMENTALLY INCOMPATIBLE ({n} breaks) — the graph-break tax "
            "will likely exceed the fusion gains. Recommend abandoning Exp 2 "
            "and proceeding to Exp 3 (CUDA graphs) per the sequence doc."
        )
    print(f"  {verdict}")
    if _is_exporting_calls > 0:
        print()
        print(f"  CAVEAT: shim was invoked {_is_exporting_calls} times — read")
        print(f"  step 3b for whether the verdict can be trusted as-is.")


if __name__ == "__main__":
    main()
