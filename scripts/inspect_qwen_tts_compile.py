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

Run from /workspace/tts-remote:
    .venv/bin/python scripts/inspect_qwen_tts_compile.py

Output sections:
  1. Model load
  2. Forward-pass dry run (uncompiled — sanity check)
  3. dynamo.explain on the model's forward — graph break report
  4. Summary verdict: clean / minor breaks / fundamentally incompatible
"""
import time
from typing import Final

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


def main() -> None:
    import torch
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
    # Use the same path the profiler used — pre-extract a prompt, then run a
    # one-step generate to capture the actual forward signature.
    prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
    )
    print(f"prompt items: {len(prompt)} (used to seed inner-model forward)")

    header("Step 3: dynamo.explain on a single forward through generate")
    # We can't easily isolate a single decoder step without monkey-patching
    # the transformer's generate loop. Instead, run the FULL generate_voice_clone
    # under torch._dynamo.explain, which captures graph breaks across the entire
    # generation. The breaks reported here are the ones torch.compile would
    # encounter when we wrap the inner model.
    print("Running explain(...) — this will execute one full generation under")
    print("dynamo's analyser. Slower than normal generate; expect ~30-60 s.")
    print()

    # explain() returns an ExplainOutput object with break_reasons, op_count, etc.
    def _generation_call() -> object:
        return model.generate_voice_clone(
            text=TEST_SENTENCE,
            language="English",
            voice_clone_prompt=prompt,
        )

    t0 = time.perf_counter()
    try:
        explanation = dynamo.explain(_generation_call)()
    except Exception as e:
        print(f"explain() raised: {type(e).__name__}: {e}")
        print()
        print("This usually means a fundamentally non-traceable construct exists")
        print("(e.g. heavy use of Python control flow that depends on tensor")
        print("values, or .item() calls in hot paths).")
        return
    explain_ms = (time.perf_counter() - t0) * 1000
    print(f"explain() finished in {explain_ms:.0f} ms")

    header("Step 4: graph break report")
    # ExplainOutput exposes break_reasons (list of GraphCompileReason).
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
                # user_stack is typically a list of FrameSummary objects
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


if __name__ == "__main__":
    main()
