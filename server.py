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
    uv add faster-whisper                # STT (Whisper)

Run:
    uv run python server.py
"""
import io
import os
import wave
import time
import base64
import asyncio
import hashlib
import logging
import tempfile
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel

# STT (faster-whisper)
_stt_model = None
_stt_lock = asyncio.Lock()
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small.en")

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
        if peak > 0 and peak <= 1.0:
            wav_array = (wav_array / peak * 0.95 * 32767).astype(np.int16)
        elif peak > 1.0:
            wav_array = (wav_array / peak * 0.95 * 32767).astype(np.int16)
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

    def _synthesize_xtts(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None, engine_params: Optional[dict] = None, voice_prompt: object = None) -> bytes:
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

    def _synthesize_f5(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None, engine_params: Optional[dict] = None, voice_prompt: object = None) -> bytes:
        p = engine_params or {}
        wav, sr, _ = model.infer(
            ref_file=voice_path or "",
            ref_text=ref_text or "",
            gen_text=text,
            nfe_step=p.get("nfe_step", 32),
            cfg_strength=p.get("cfg_strength", 2),
            sway_sampling_coef=p.get("sway_sampling_coef", -1),
            speed=p.get("speed", 1.0),
            seed=p.get("seed"),
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
        # Use SDPA (PyTorch native FlashAttention v2 kernels) for faster inference
        load_kwargs: dict = {
            "device_map": "cuda:0",
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        logger.info("Qwen3-TTS: using SDPA (PyTorch native flash attention)")
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            **load_kwargs,
        )
        logger.info(f"Qwen3-TTS loaded in {(time.perf_counter() - start) * 1000:.0f}ms")
        return model

    # Tuned defaults from quality iteration (best: t=0.7, st=0.85 with ref_text)
    QWEN3_DEFAULTS: dict = {"temperature": 0.7, "subtalker_temperature": 0.85}

    def _synthesize_qwen3(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None, engine_params: Optional[dict] = None, voice_prompt: object = None) -> bytes:
        p = {**QWEN3_DEFAULTS, **(engine_params or {})}
        language = p.pop("language", "English")

        # Use pre-extracted speaker embedding if available (skips per-call extraction)
        if voice_prompt is not None:
            clone_kwargs: dict = dict(text=text, language=language, voice_clone_prompt=voice_prompt)
        else:
            # Fall back to extracting from audio on every call
            use_xvector = not ref_text
            clone_kwargs: dict = dict(text=text, language=language, x_vector_only_mode=use_xvector)
            if not use_xvector:
                clone_kwargs["ref_text"] = ref_text
            if voice_path:
                clone_kwargs["ref_audio"] = voice_path
            else:
                clone_kwargs["ref_audio"] = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"

        # Pass generation kwargs
        for k in ("temperature", "top_k", "top_p", "repetition_penalty", "subtalker_temperature", "subtalker_top_k", "subtalker_top_p"):
            if k in p:
                clone_kwargs[k] = p[k]
        wavs, sr = model.generate_voice_clone(**clone_kwargs)
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

    def _synthesize_orpheus(model: object, text: str, voice_path: Optional[str], ref_text: Optional[str] = None, engine_params: Optional[dict] = None, voice_prompt: object = None) -> bytes:
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


def _get_stt_model() -> object:
    global _stt_model
    if _stt_model is None:
        from faster_whisper import WhisperModel
        logger.info(f"Loading STT model: {STT_MODEL_SIZE}")
        _stt_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type="float16")
        logger.info(f"STT model loaded: {STT_MODEL_SIZE}")
    return _stt_model


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

class PrimeVoiceRequest(BaseModel):
    voice_audio_b64: str
    engine: str
    ref_text: Optional[str] = None


class PrimeVoiceResponse(BaseModel):
    cache_key: str
    size_bytes: int


# In-memory voice cache: cache_key -> {"path": tmpfs_path, "prompt": pre-extracted voice_clone_prompt}
_voice_cache: dict[str, dict] = {}
MAX_VOICE_CACHE_ENTRIES = 20
MAX_VOICE_FILE_BYTES = 10 * 1024 * 1024  # 10MB per voice file


class SynthesizeRequest(BaseModel):
    text: str
    engine: str
    voice_audio_b64: Optional[str] = None
    voice_cache_key: Optional[str] = None  # Use cached voice instead of uploading WAV
    ref_text: Optional[str] = None  # Reference audio transcript (used by F5-TTS)
    engine_params: Optional[dict] = None  # Engine-specific params (temperature, cfg_strength, etc.)
    sample_rate: int = CANONICAL_SAMPLE_RATE


class SynthesizeResponse(BaseModel):
    audio_b64: str
    synthesis_ms: float
    vram_mb: Optional[float] = None
    engine: str
    sample_rate: int


app = FastAPI(title="TTS Inference Server")


@app.on_event("shutdown")
async def cleanup_voice_cache() -> None:
    """Clean up cached voice files from /dev/shm on server shutdown."""
    for entry in _voice_cache.values():
        path = entry["path"] if isinstance(entry, dict) else entry
        if os.path.exists(path):
            os.unlink(path)
    _voice_cache.clear()
    logger.info("Voice cache cleaned up")


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


@app.post("/prime-voice", response_model=PrimeVoiceResponse)
async def prime_voice(req: PrimeVoiceRequest) -> PrimeVoiceResponse:
    """Upload a voice profile once. Returns a cache_key for subsequent /synthesize calls."""
    try:
        voice_bytes = base64.b64decode(req.voice_audio_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")

    # Size check
    if len(voice_bytes) > MAX_VOICE_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"Voice file too large ({len(voice_bytes)} bytes, max {MAX_VOICE_FILE_BYTES})")

    # Validate WAV format
    try:
        with wave.open(io.BytesIO(voice_bytes), "rb") as wf:
            if wf.getnchannels() not in (1, 2) or wf.getsampwidth() not in (1, 2, 4):
                raise HTTPException(status_code=400, detail="Unsupported WAV format")
    except wave.Error:
        raise HTTPException(status_code=400, detail="Invalid WAV file")

    cache_key = hashlib.sha256(voice_bytes).hexdigest()

    if cache_key not in _voice_cache:
        # Evict oldest entry if cache is full
        if len(_voice_cache) >= MAX_VOICE_CACHE_ENTRIES:
            oldest_key = next(iter(_voice_cache))
            evicted = _voice_cache.pop(oldest_key)
            if os.path.exists(evicted["path"]):
                os.unlink(evicted["path"])
            logger.info(f"Voice cache full, evicted: {oldest_key[:16]}...")

        # Save to /dev/shm (tmpfs — RAM only, no disk persistence)
        cache_path = f"/dev/shm/voice_cache_{cache_key[:16]}.wav"
        with open(cache_path, "wb") as f:
            f.write(voice_bytes)

        cache_entry: dict = {"path": cache_path, "prompt": None}

        # Pre-extract speaker embedding for Qwen3 (skips re-extraction per sentence)
        if req.engine == "qwen3" and "qwen3" in LOADERS:
            try:
                model = await asyncio.wait_for(
                    asyncio.to_thread(_get_engine, "qwen3"), timeout=300.0
                )
                use_xvector = not req.ref_text
                mode_label = "x-vector only" if use_xvector else "ref_text-aware"
                logger.info(f"Pre-extracting Qwen3 speaker embedding ({mode_label}) for cache_key={cache_key[:16]}...")
                start = time.perf_counter()
                extract_kwargs: dict = {
                    "ref_audio": cache_path,
                    "x_vector_only_mode": use_xvector,
                }
                if req.ref_text:
                    extract_kwargs["ref_text"] = req.ref_text
                prompt_items = await asyncio.to_thread(
                    model.create_voice_clone_prompt,
                    **extract_kwargs,
                )
                extract_ms = (time.perf_counter() - start) * 1000
                cache_entry["prompt"] = prompt_items
                logger.info(f"Speaker embedding extracted in {extract_ms:.0f}ms ({mode_label})")
            except Exception as e:
                logger.warning(f"Speaker embedding extraction failed (will extract per-call): {e}")

        _voice_cache[cache_key] = cache_entry
        logger.info(f"Voice profile cached: key={cache_key[:16]}..., size={len(voice_bytes)} bytes")
    else:
        logger.info(f"Voice profile already cached: key={cache_key[:16]}...")

    return PrimeVoiceResponse(cache_key=cache_key, size_bytes=len(voice_bytes))


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

    # Resolve voice profile: prefer cache_key, fall back to inline b64
    voice_path: Optional[str] = None
    voice_prompt = None  # Pre-extracted speaker embedding (Qwen3)
    tmp_file = None
    if req.voice_cache_key and req.voice_cache_key in _voice_cache:
        cache_entry = _voice_cache[req.voice_cache_key]
        voice_path = cache_entry["path"]
        voice_prompt = cache_entry.get("prompt")
    elif req.voice_audio_b64:
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
                asyncio.to_thread(synthesizer, model, req.text, voice_path, req.ref_text, req.engine_params, voice_prompt),
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


@app.post("/synthesize-stream")
async def synthesize_stream(req: SynthesizeRequest) -> StreamingResponse:
    """Streaming synthesis: returns chunked raw PCM int16 frames."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        model = await asyncio.wait_for(
            asyncio.to_thread(_get_engine, req.engine),
            timeout=300.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail=f"Engine '{req.engine}' load timed out")

    lock = ENGINE_LOCKS[req.engine]

    voice_path: Optional[str] = None
    voice_prompt = None
    tmp_file = None
    if req.voice_cache_key and req.voice_cache_key in _voice_cache:
        cache_entry = _voice_cache[req.voice_cache_key]
        voice_path = cache_entry["path"]
        voice_prompt = cache_entry.get("prompt")
    elif req.voice_audio_b64:
        try:
            voice_bytes = base64.b64decode(req.voice_audio_b64)
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.write(voice_bytes)
            tmp_file.close()
            voice_path = tmp_file.name
        except Exception as e:
            logger.warning(f"Voice profile decode failed: {e}")

    async def generate_chunks():
        """Run synthesis and yield PCM chunks."""
        try:
            synthesizer = SYNTHESIZERS[req.engine]
            start = time.perf_counter()
            async with lock:
                audio_bytes = await asyncio.wait_for(
                    asyncio.to_thread(synthesizer, model, req.text, voice_path, req.ref_text, req.engine_params, voice_prompt),
                    timeout=120.0,
                )
            synthesis_ms = (time.perf_counter() - start) * 1000

            # Extract raw PCM from WAV
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                pcm_data = wf.readframes(wf.getnframes())

            # Yield in chunks of 2400 samples (100ms at 24kHz)
            chunk_size = 2400 * 2  # 2 bytes per int16 sample
            for offset in range(0, len(pcm_data), chunk_size):
                yield pcm_data[offset:offset + chunk_size]

            logger.info(f"[{req.engine}] streamed {len(req.text)} chars in {synthesis_ms:.0f}ms, {len(pcm_data) // chunk_size + 1} chunks")
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}", exc_info=True)
        finally:
            if tmp_file and os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={"X-Engine": req.engine},
    )


@app.post("/transcribe")
async def transcribe(request: Request) -> dict:
    """Transcribe raw PCM audio (16kHz, 16-bit, mono) using GPU-accelerated Whisper."""
    audio_bytes = await request.body()
    if len(audio_bytes) < 16000:  # < 0.5s
        return {"transcript": "", "inference_ms": 0.0}

    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _run_transcribe(model, audio):
        segments, info = model.transcribe(
            audio, beam_size=5, no_speech_threshold=0.6, log_prob_threshold=-1.0,
        )
        return [s.text for s in segments if s.no_speech_prob <= 0.6]

    async with _stt_lock:
        start = time.perf_counter()
        model = _get_stt_model()
        parts = await asyncio.wait_for(
            asyncio.to_thread(_run_transcribe, model, audio_array),
            timeout=30.0,
        )
        inference_ms = (time.perf_counter() - start) * 1000

    transcript = " ".join(parts).strip()
    logger.info(f"[STT] {len(audio_bytes)} bytes -> \"{transcript}\" in {inference_ms:.0f}ms")
    return {"transcript": transcript, "inference_ms": round(inference_ms, 1)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TTS_PORT", "9000"))
    logger.info(f"Starting TTS Inference Server on port {port}")
    logger.info(f"Available engines: {list(LOADERS.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=port)
