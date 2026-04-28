"""
Introspect the qwen-tts package to identify methods and source locations
we need to instrument for Phase 1 profiling.

Reads only — no model load, no GPU work. Pure metadata extraction.

Run from /workspace/tts-remote:
    .venv/bin/python scripts/inspect_qwen_tts.py
"""
import inspect


def header(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> None:
    from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]

    header("qwen-tts package metadata")
    import qwen_tts
    print(f"qwen_tts module: {qwen_tts.__file__}")
    print(f"qwen_tts version: {getattr(qwen_tts, '__version__', '?')}")

    header("Qwen3TTSModel — public callables")
    for name in sorted(dir(Qwen3TTSModel)):
        if name.startswith("_"):
            continue
        attr = getattr(Qwen3TTSModel, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                print(f"  {name}{sig}")
            except (ValueError, TypeError):
                print(f"  {name}(?)")

    header("Qwen3TTSModel — source file")
    try:
        print(inspect.getsourcefile(Qwen3TTSModel))
    except (TypeError, OSError) as e:
        print(f"  (source file not resolvable: {e})")

    header("generate_voice_clone — source")
    try:
        src_file = inspect.getsourcefile(Qwen3TTSModel.generate_voice_clone)
        print(f"file: {src_file}")
        src = inspect.getsource(Qwen3TTSModel.generate_voice_clone)
        for i, line in enumerate(src.splitlines(), 1):
            print(f"  {i:4d}: {line}")
    except (TypeError, OSError) as e:
        print(f"  (could not extract source: {e})")

    header("create_voice_clone_prompt — source")
    try:
        src = inspect.getsource(Qwen3TTSModel.create_voice_clone_prompt)
        for i, line in enumerate(src.splitlines(), 1):
            print(f"  {i:4d}: {line}")
    except (AttributeError, TypeError, OSError) as e:
        print(f"  (not available or no source: {e})")

    header("Qwen3TTSModel attributes that look like sub-models")
    # Loaded model would expose .talker, .code_predictor, .speaker_encoder etc.
    # We're inspecting the class, not an instance, so this is just hints.
    print("  (instance-level introspection happens after model load — skipping)")

    header("Look for inner generation methods on the class")
    interesting_terms = (
        "generate", "tokenize", "decode", "prefill", "audio", "wav",
        "code_pred", "subtalker", "speaker", "embed", "prompt",
    )
    for name in sorted(dir(Qwen3TTSModel)):
        lower = name.lower()
        if any(term in lower for term in interesting_terms):
            print(f"  {name}")


if __name__ == "__main__":
    main()
