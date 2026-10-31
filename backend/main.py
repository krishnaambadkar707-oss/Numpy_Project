import os
import uuid
import numpy as np
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.audio_processor import (
    load_wav_from_bytes,
    export_wav_bytes,
    analyze_signal,
    reduce_noise,
    smooth_signal,
    frequency_analysis,
    compare_signals,
    downsample_waveform,
    generate_demo_wav
)
from backend.translate_routes import router as translate_router

app = FastAPI(
    title="Audio Signal Processor API",
    description="NumPy-powered Audio Processing API providing noise reduction, moving average smoothing, FFT frequency analysis, and MSE metrics.",
    version="1.0.0",
    redirect_slashes=False
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount translation router under both /api/translate and /translate (in case proxy strips /api)
app.include_router(translate_router, prefix="/api/translate")
app.include_router(translate_router, prefix="/translate")

# In-memory storage for active audio sessions (capped for production memory safety)
AUDIO_STORE: Dict[str, Dict[str, Any]] = {}
MAX_SESSIONS = 50
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max upload

is_posix = os.name != "nt"
tmp_base = "/tmp" if is_posix else os.environ.get("TEMP", os.environ.get("TMP", "C:/tmp"))
SESSIONS_DIR = os.path.join(tmp_base, "audio_sessions")
try:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
except Exception:
    pass

def save_session_to_disk(audio_id: str, data: Dict[str, Any]):
    """Persists audio session arrays to disk so serverless instances can recover sessions."""
    try:
        session_file = os.path.join(SESSIONS_DIR, f"{audio_id}.npz")
        np.savez_compressed(
            session_file,
            original_signal=data["original_signal"],
            processed_signal=data["processed_signal"],
            sample_rate=np.array([data["sample_rate"]]),
            n_channels=np.array([data.get("n_channels", 1)]),
            n_frames=np.array([data.get("n_frames", len(data["original_signal"]))]),
            threshold=np.array([data.get("threshold", 500)]),
            window_size=np.array([data.get("window_size", 5)])
        )
    except Exception:
        pass

def load_session_from_disk(audio_id: str) -> Optional[Dict[str, Any]]:
    """Restores audio session from disk if this lambda container didn't process the upload."""
    session_file = os.path.join(SESSIONS_DIR, f"{audio_id}.npz")
    if os.path.exists(session_file):
        try:
            with np.load(session_file) as npz:
                return {
                    "original_signal": npz["original_signal"],
                    "processed_signal": npz["processed_signal"],
                    "sample_rate": int(npz["sample_rate"][0]),
                    "n_channels": int(npz["n_channels"][0]),
                    "n_frames": int(npz["n_frames"][0]),
                    "threshold": float(npz["threshold"][0]),
                    "window_size": int(npz["window_size"][0])
                }
        except Exception:
            pass
    return None

def clean_old_sessions():
    """Keeps in-memory storage bounded to prevent memory leaks in production."""
    if len(AUDIO_STORE) >= MAX_SESSIONS:
        oldest_keys = list(AUDIO_STORE.keys())[:10]
        for key in oldest_keys:
            AUDIO_STORE.pop(key, None)
            session_file = os.path.join(SESSIONS_DIR, f"{key}.npz")
            if os.path.exists(session_file):
                try:
                    os.remove(session_file)
                except OSError:
                    pass

class ProcessRequest(BaseModel):
    audio_id: str
    threshold: float = 500.0
    window_size: int = 5

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Audio Signal Processor API"}

@app.post("/upload")
@app.post("/api/upload")
async def upload_audio(request: Request, filename: Optional[str] = "uploaded.wav"):
    contents = await request.body()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file payload.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds maximum 25MB limit.")
    
    clean_old_sessions()
    
    try:
        signal, sample_rate, n_channels, n_frames = load_wav_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse WAV file: {str(e)}")

    audio_id = str(uuid.uuid4())
    
    # Analyze original
    orig_metrics = analyze_signal(signal)
    orig_waveform = downsample_waveform(signal)
    
    # Process with default parameters (threshold 500, window_size 5)
    cleaned = reduce_noise(signal, threshold=500)
    smoothed = smooth_signal(cleaned, window_size=5)
    proc_metrics = analyze_signal(smoothed)
    proc_waveform = downsample_waveform(smoothed)
    freqs, magnitudes = frequency_analysis(smoothed, sample_rate)
    mse = compare_signals(signal, smoothed)

    # Store in memory & persist to disk for serverless session recovery
    session_data = {
        "original_signal": signal,
        "processed_signal": smoothed,
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "n_frames": n_frames,
        "threshold": 500,
        "window_size": 5
    }
    AUDIO_STORE[audio_id] = session_data
    save_session_to_disk(audio_id, session_data)

    return {
        "audio_id": audio_id,
        "filename": filename,
        "sample_rate": sample_rate,
        "duration_sec": round(len(signal) / sample_rate, 2),
        "total_samples": len(signal),
        "channels": n_channels,
        "original_metrics": orig_metrics,
        "processed_metrics": proc_metrics,
        "mse": mse,
        "threshold": 500,
        "window_size": 5,
        "original_waveform": orig_waveform,
        "processed_waveform": proc_waveform,
        "frequencies": freqs,
        "magnitudes": magnitudes
    }

@app.get("/demo")
@app.get("/api/demo")
def get_demo_audio():
    signal, sample_rate = generate_demo_wav(duration_sec=3.0, sample_rate=22050)
    audio_id = str(uuid.uuid4())
    
    orig_metrics = analyze_signal(signal)
    orig_waveform = downsample_waveform(signal)
    
    cleaned = reduce_noise(signal, threshold=500)
    smoothed = smooth_signal(cleaned, window_size=5)
    proc_metrics = analyze_signal(smoothed)
    proc_waveform = downsample_waveform(smoothed)
    freqs, magnitudes = frequency_analysis(smoothed, sample_rate)
    mse = compare_signals(signal, smoothed)

    session_data = {
        "original_signal": signal,
        "processed_signal": smoothed,
        "sample_rate": sample_rate,
        "n_channels": 1,
        "n_frames": len(signal),
        "threshold": 500,
        "window_size": 5
    }
    AUDIO_STORE[audio_id] = session_data
    save_session_to_disk(audio_id, session_data)

    return {
        "audio_id": audio_id,
        "filename": "demo_synthetic_audio.wav",
        "sample_rate": sample_rate,
        "duration_sec": 3.0,
        "total_samples": len(signal),
        "channels": 1,
        "original_metrics": orig_metrics,
        "processed_metrics": proc_metrics,
        "mse": mse,
        "threshold": 500,
        "window_size": 5,
        "original_waveform": orig_waveform,
        "processed_waveform": proc_waveform,
        "frequencies": freqs,
        "magnitudes": magnitudes
    }

@app.post("/process")
@app.post("/api/process")
def process_audio(req: ProcessRequest):
    session = AUDIO_STORE.get(req.audio_id)
    if not session:
        session = load_session_from_disk(req.audio_id)
        if session:
            AUDIO_STORE[req.audio_id] = session

    if not session:
        raise HTTPException(status_code=404, detail="Audio session not found. Please upload or generate audio first.")

    original = session["original_signal"]
    sample_rate = session["sample_rate"]

    cleaned = reduce_noise(original, threshold=req.threshold)
    smoothed = smooth_signal(cleaned, window_size=req.window_size)

    session["processed_signal"] = smoothed
    session["threshold"] = req.threshold
    session["window_size"] = req.window_size
    save_session_to_disk(req.audio_id, session)

    orig_metrics = analyze_signal(original)
    proc_metrics = analyze_signal(smoothed)
    proc_waveform = downsample_waveform(smoothed)
    freqs, magnitudes = frequency_analysis(smoothed, sample_rate)
    mse = compare_signals(original, smoothed)

    return {
        "audio_id": req.audio_id,
        "threshold": req.threshold,
        "window_size": req.window_size,
        "original_metrics": orig_metrics,
        "processed_metrics": proc_metrics,
        "mse": mse,
        "processed_waveform": proc_waveform,
        "frequencies": freqs,
        "magnitudes": magnitudes
    }

@app.get("/audio/{audio_id}/{audio_type}")
@app.get("/api/audio/{audio_id}/{audio_type}")
def get_audio_file(audio_id: str, audio_type: str):
    session = AUDIO_STORE.get(audio_id)
    if not session:
        session = load_session_from_disk(audio_id)
        if session:
            AUDIO_STORE[audio_id] = session

    if not session:
        raise HTTPException(status_code=404, detail="Audio session not found.")
    
    session = AUDIO_STORE[audio_id]
    if audio_type == "original":
        signal = session["original_signal"]
    elif audio_type == "processed":
        signal = session["processed_signal"]
    else:
        raise HTTPException(status_code=400, detail="Invalid audio type. Must be 'original' or 'processed'.")

    sample_rate = session["sample_rate"]
    wav_bytes = export_wav_bytes(signal, sample_rate)

    return Response(content=wav_bytes, media_type="audio/wav")
