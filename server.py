"""
TTS Inference Server for GPU Deployment
========================================
Runs on a remote GPU server. Exposes voice cloning synthesis over HTTP.
Only engines with installed packages are available.

Setup (on GPU server):
    apt-get install -y build-essential python3-dev ffmpeg vim
    uv python install 3.12 && uv python pin 3.12
    uv add fastapi uvicorn scipy numpy soundfile torch
    uv add coqui-tts "transformers<5"   # XTTS-v2
    uv add f5-tts                        # F5-TTS
    uv pip install qwen-tts              # Qwen3-TTS
    uv pip install orpheus-speech        # Orpheus TTS

Run:
    uv run python server.py
"""
import io
import os
import wave
import time
import base64
import asyncio
import logging
import tempfile
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("tts-inference")

CANONICAL_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_vram_usage_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def clear_gpu_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def to_wav_bytes(wav_array: np.ndarray, source_rate: int, target_rate: int) -> bytes:
    if source_rate != target_rate:
        import soxr
        wav_array = soxr.resample(wav_array.astype(np.float64), source_rate, target_rate, quality="VHQ")
    if wav_array.dtype != np.int16:
        peak = np.abs(wav_array).max() if len(wav_array) > 0 else 0
        if peak <= 1.0:
            wav_array = (wav_array * 32767).astype(np.int16)
        else:
            wav_array = wav_array.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_rate)
        wf.writeframes(wav_array.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Engine implementations — only registered if their package is importable
# ---------------------------------------------------------------------------

LOADERS: dict[str, callable] = {}
SYNTHESIZERS: dict[str, callable] = {}

# --- XTTS-v2 (coqui-tts) ---
try:
    import TTS  # noqa: F401 - just check if importable
    def _load_xtts() -> object:
        from TTS.api import TTS as CoquiTTS
        logger.info("Loading XTTS-v2...")
        start = time.perf_counter()
        model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        logger.info(f"XTTS-v2 loaded in {(time.perf_counter() - start) * 1000:.0f}ms")
        return model

    def _synthesize_xtts(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None) -> bytes:
        if voice_path:
            wav = model.tts(text=text, speaker_wav=voice_path, language="en")
        else:
            wav = model.tts(text=text, language="en", speaker="Ana Florence")
        return to_wav_bytes(np.array(wav), 24000, CANONICAL_SAMPLE_RATE)

    LOADERS["xtts"] = _load_xtts
    SYNTHESIZERS["xtts"] = _synthesize_xtts
    logger.info("XTTS-v2 engine: available")
except ImportError:
    logger.info("XTTS-v2 engine: not installed (coqui-tts missing)")

# --- F5-TTS ---
try:
    import f5_tts  # noqa: F401
    def _load_f5() -> object:
        from f5_tts.api import F5TTS
        logger.info("Loading F5-TTS...")
        start = time.perf_counter()
        model = F5TTS()
        logger.info(f"F5-TTS loaded in {(time.perf_counter() - start) * 1000:.0f}ms")
        return model

    def _synthesize_f5(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None) -> bytes:
        wav, sr, _ = model.infer(
            ref_file=voice_path or "",
            ref_text=ref_text or "",
            gen_text=text,
        )
        return to_wav_bytes(np.array(wav), sr, CANONICAL_SAMPLE_RATE)

    LOADERS["f5"] = _load_f5
    SYNTHESIZERS["f5"] = _synthesize_f5
    logger.info("F5-TTS engine: available")
except ImportError:
    logger.info("F5-TTS engine: not installed (f5-tts missing)")

# --- Qwen3-TTS ---
try:
    import qwen_tts as _qwen_check  # noqa: F401
    def _load_qwen3() -> object:
        import torch
        from qwen_tts import Qwen3TTSModel
        logger.info("Loading Qwen3-TTS...")
        start = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        logger.info(f"Qwen3-TTS loaded in {(time.perf_counter() - start) * 1000:.0f}ms")
        return model

    def _synthesize_qwen3(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None) -> bytes:
        if voice_path:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=voice_path,
                x_vector_only_mode=False,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
                x_vector_only_mode=False,
            )
        wav_array = wavs[0] if isinstance(wavs, list) else wavs
        if hasattr(wav_array, 'cpu'):
            wav_array = wav_array.cpu().numpy()
        return to_wav_bytes(np.array(wav_array), sr, CANONICAL_SAMPLE_RATE)

    LOADERS["qwen3"] = _load_qwen3
    SYNTHESIZERS["qwen3"] = _synthesize_qwen3
    logger.info("Qwen3-TTS engine: available")
except ImportError:
    logger.info("Qwen3-TTS engine: not installed (qwen-tts missing)")

# --- Orpheus TTS ---
try:
    from orpheus_tts import OrpheusModel as _orpheus_check  # noqa: F401
    def _load_orpheus() -> object:
        from orpheus_tts import OrpheusModel
        logger.info("Loading Orpheus TTS...")
        start = time.perf_counter()
        model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        logger.info(f"Orpheus TTS loaded in {(time.perf_counter() - start) * 1000:.0f}ms")
        return model

    def _synthesize_orpheus(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None) -> bytes:
        # Orpheus uses preset voices, not reference audio cloning
        syn_tokens = model.generate_speech(prompt=text, voice="tara")
        all_audio = []
        for chunk in syn_tokens:
            all_audio.append(chunk)
        if not all_audio:
            return b""
        audio_data = b"".join(all_audio)
        wav_array = np.frombuffer(audio_data, dtype=np.int16)
        return to_wav_bytes(wav_array, 24000, CANONICAL_SAMPLE_RATE)

    LOADERS["orpheus"] = _load_orpheus
    SYNTHESIZERS["orpheus"] = _synthesize_orpheus
    logger.info("Orpheus TTS engine: available")
except ImportError:
    logger.info("Orpheus TTS engine: not installed (orpheus-speech missing)")


# ---------------------------------------------------------------------------
# Engine manager (LRU-1: one model in VRAM at a time)
# ---------------------------------------------------------------------------

ENGINES: dict[str, object] = {}  # loaded model instances
ENGINE_LOCKS: dict[str, asyncio.Lock] = {}
_current_gpu_engine: Optional[str] = None


def _get_engine(name: str) -> object:
    global _current_gpu_engine

    if name not in LOADERS:
        available = list(LOADERS.keys())
        raise ValueError(f"Unknown engine: '{name}'. Available: {available}")

    # Already loaded
    if name in ENGINES:
        return ENGINES[name]

    # Evict previous engine
    if _current_gpu_engine and _current_gpu_engine in ENGINES and _current_gpu_engine != name:
        logger.info(f"LRU-1: evicting '{_current_gpu_engine}' to load '{name}'")
        del ENGINES[_current_gpu_engine]
        clear_gpu_cache()

    # Load new engine
    ENGINES[name] = LOADERS[name]()
    ENGINE_LOCKS[name] = asyncio.Lock()
    _current_gpu_engine = name
    return ENGINES[name]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    engine: str
    voice_audio_b64: Optional[str] = None
    ref_text: Optional[str] = None  # Reference audio transcript (used by F5-TTS)
    sample_rate: int = CANONICAL_SAMPLE_RATE


class SynthesizeResponse(BaseModel):
    audio_b64: str
    synthesis_ms: float
    vram_mb: Optional[float] = None
    engine: str
    sample_rate: int


app = FastAPI(title="TTS Inference Server")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "loaded_engines": list(ENGINES.keys()),
        "available_engines": list(LOADERS.keys()),
        "vram_mb": get_vram_usage_mb(),
    }


@app.get("/engines")
async def list_engines() -> dict:
    return {"available": list(LOADERS.keys()), "loaded": list(ENGINES.keys())}


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    # Load engine (may download model on first call)
    try:
        model = await asyncio.wait_for(
            asyncio.to_thread(_get_engine, req.engine),
            timeout=300.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail=f"Engine '{req.engine}' load timed out")
    except Exception as e:
        logger.error(f"Engine load failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Engine load failed: {e}")

    lock = ENGINE_LOCKS[req.engine]

    # Write voice profile to temp file if provided
    voice_path: Optional[str] = None
    tmp_file = None
    if req.voice_audio_b64:
        try:
            voice_bytes = base64.b64decode(req.voice_audio_b64)
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.write(voice_bytes)
            tmp_file.close()
            voice_path = tmp_file.name
        except Exception as e:
            logger.warning(f"Voice profile decode failed: {e}")

    try:
        synthesizer = SYNTHESIZERS[req.engine]
        async with lock:
            start = time.perf_counter()
            audio_bytes = await asyncio.wait_for(
                asyncio.to_thread(synthesizer, model, req.text, voice_path, req.ref_text),
                timeout=120.0,
            )
            synthesis_ms = (time.perf_counter() - start) * 1000

        vram = get_vram_usage_mb()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        logger.info(f"[{req.engine}] {len(req.text)} chars in {synthesis_ms:.0f}ms, VRAM={vram}MB")

        return SynthesizeResponse(
            audio_b64=audio_b64,
            synthesis_ms=round(synthesis_ms, 1),
            vram_mb=round(vram, 1) if vram is not None else None,
            engine=req.engine,
            sample_rate=CANONICAL_SAMPLE_RATE,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Synthesis timed out (120s)")
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TTS_PORT", "9000"))
    logger.info(f"Starting TTS Inference Server on port {port}")
    logger.info(f"Available engines: {list(LOADERS.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=port)
