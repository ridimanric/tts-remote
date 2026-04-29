"""
Inspect vLLM's compatibility with Qwen3-TTS architecture.

Per Phase 3.D / Experiment 4 of the optimisation sequence, before
committing to a vLLM port we need to know whether vLLM can host
Qwen3-TTS at all. Three possibilities:

  1. Direct support: `Qwen3TTSForConditionalGeneration` is registered
     in vllm.ModelRegistry. No surgery needed beyond config + sampling.
  2. Indirect support: vLLM doesn't recognise `Qwen3TTSForConditionalGeneration`
     but it DOES support related architectures (e.g. Qwen2/Qwen3 base
     LLMs). Surgery would mean writing a custom vLLM model definition
     that mirrors the qwen-tts internals (talker + code_predictor).
  3. Fundamentally unsuitable: nothing in the Qwen3-TTS family is
     supported, AND the doubly-autoregressive structure (talker → code
     predictor) doesn't fit vLLM's standard transformer-LM serving model.

This script queries vllm.ModelRegistry to enumerate what architectures
are supported and matches them against Qwen3-TTS's class hierarchy.

Run from /workspace/tts-remote (after setup_vllm_spike.sh):
    .venv-vllm-spike/bin/python scripts/inspect_vllm_qwen3_tts.py
"""
from typing import Any


def header(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> None:
    header("Step 1: import vllm + locate ModelRegistry")
    try:
        import vllm
    except ImportError as e:
        print(f"ERROR: vllm not importable: {e}")
        print(f"Run scripts/setup_vllm_spike.sh first.")
        return
    print(f"vllm version: {vllm.__version__}")

    # vLLM's model registry — exact location varies by version.
    # Try a few known import paths.
    registry: Any = None
    registry_path: str = ""
    for path in (
        "vllm.model_executor.models.ModelRegistry",
        "vllm.model_executor.models.registry.ModelRegistry",
        "vllm.ModelRegistry",
    ):
        try:
            module_path, _, attr = path.rpartition(".")
            module = __import__(module_path, fromlist=[attr])
            registry = getattr(module, attr)
            registry_path = path
            break
        except (ImportError, AttributeError):
            continue

    if registry is None:
        print("ERROR: could not locate vllm.ModelRegistry. Inspecting alternatives:")
        try:
            from vllm.model_executor.models import _MULTIMODAL_MODELS  # type: ignore[import-not-found]
            print("  found _MULTIMODAL_MODELS dict — vllm version may use older registry style.")
        except ImportError:
            pass
        return
    print(f"Registry located at: {registry_path}")

    header("Step 2: enumerate supported model architectures")
    # ModelRegistry exposes get_supported_archs() in newer vllm; try variants.
    supported: list[str] = []
    for method_name in ("get_supported_archs", "_get_supported_archs", "supported_archs"):
        if hasattr(registry, method_name):
            method = getattr(registry, method_name)
            if callable(method):
                supported = list(method())
            else:
                supported = list(method)
            print(f"Used: {method_name}")
            break

    if not supported:
        # Fall back: dir() the registry, look for arch list.
        for attr in dir(registry):
            value = getattr(registry, attr)
            if isinstance(value, (list, tuple, set, dict)) and value:
                if any("for" in str(item).lower() and "generation" in str(item).lower() for item in value):
                    supported = list(value)
                    print(f"Used fallback attribute: {attr}")
                    break

    if not supported:
        print("ERROR: could not enumerate supported architectures from registry.")
        print("Inspect the registry object manually:")
        print(f"  attrs: {[a for a in dir(registry) if not a.startswith('_')]}")
        return

    print(f"Total supported architectures: {len(supported)}")

    header("Step 3: search for Qwen3-TTS and related architectures")
    # Search patterns, ranked by relevance to our use case.
    patterns = (
        ("Qwen3TTSForConditionalGeneration", "EXACT — direct support, minimal surgery needed"),
        ("Qwen3TTS", "PARTIAL — Qwen3-TTS family present, may need custom registration"),
        ("Qwen3TTSTalker", "PARTIAL — talker submodel only; would need code_predictor as separate model"),
        ("Qwen3", "INDIRECT — base Qwen3 LLM, structurally similar; would need full custom model"),
        ("Qwen2", "INDIRECT — older Qwen, similar transformer; would need full custom model"),
        ("Qwen", "INDIRECT — any Qwen family at all"),
    )

    found: dict[str, list[str]] = {p[0]: [] for p in patterns}
    for arch in supported:
        arch_str = str(arch)
        for pattern, _ in patterns:
            if pattern.lower() in arch_str.lower():
                found[pattern].append(arch_str)
                break  # take the most-specific match only

    for pattern, description in patterns:
        matches = found[pattern]
        marker = "MATCH" if matches else "no    "
        print(f"  [{marker}] '{pattern}' — {description}")
        for m in matches[:5]:
            print(f"           -> {m}")

    header("Step 4: verdict")
    if found["Qwen3TTSForConditionalGeneration"]:
        verdict = "DIRECT SUPPORT — Qwen3-TTS is registered. Port is straightforward."
    elif found["Qwen3TTS"]:
        verdict = "PARTIAL SUPPORT — Qwen3TTS family exists, may need custom registration."
    elif found["Qwen3TTSTalker"]:
        verdict = (
            "PARTIAL SUPPORT — talker submodel exists. Code_predictor would need\n"
            "  separate handling. Possible but requires the doubly-autoregressive\n"
            "  flow to be implemented as two vllm models."
        )
    elif found["Qwen3"]:
        verdict = (
            "INDIRECT SUPPORT — base Qwen3 LLM is supported. Writing a custom\n"
            "  vllm model definition for Qwen3-TTS would build on this. Significant\n"
            "  work but tractable."
        )
    elif found["Qwen2"]:
        verdict = (
            "INDIRECT SUPPORT — Qwen2 supported, structurally similar. Custom\n"
            "  vllm model definition needed; Qwen3-TTS architecture differences\n"
            "  may be load-bearing."
        )
    elif found["Qwen"]:
        verdict = (
            "WEAK INDIRECT SUPPORT — only older Qwen variants. Custom model\n"
            "  development would need to bridge multiple architectural changes."
        )
    else:
        verdict = "NO SUPPORT — Qwen family not in vllm. Pivot or commit to a major upstream."
    print(f"  {verdict}")

    header("Step 5: search for related multimodal/TTS support")
    # vLLM has been adding multimodal support; check for TTS or audio-output models.
    audio_patterns = ("TTS", "Audio", "Speech", "Voice", "Mel", "Codec")
    audio_hits: dict[str, list[str]] = {}
    for arch in supported:
        arch_str = str(arch)
        for pattern in audio_patterns:
            if pattern.lower() in arch_str.lower():
                audio_hits.setdefault(pattern, []).append(arch_str)

    if not audio_hits:
        print("  no audio/TTS-related architectures found in vllm registry.")
    else:
        print("  audio/TTS-related architectures found:")
        for pattern, matches in audio_hits.items():
            print(f"    {pattern}:")
            for m in matches:
                print(f"      - {m}")

    header("Done")
    print("Next step depends on the verdict above. If DIRECT or PARTIAL,")
    print("the next inspection should attempt to actually load the model")
    print("via vLLM. If INDIRECT, evaluate whether the custom-model surface")
    print("is smaller than implementing Experiment 3 from the parked state.")


if __name__ == "__main__":
    main()
