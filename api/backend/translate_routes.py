"""
translate_routes.py - FastAPI APIRouter for the Audio Translator service.

Endpoints:
- GET  /api/translate/languages    -> List supported languages and default neural voices
- POST /api/translate/start        -> Initiates audio translation job in background
- GET  /api/translate/status/{id}  -> Polls real-time job progress and transcripts
- GET  /api/translate/result/{id}  -> Streams the translated audio result file
"""

import os
import time
import uuid
import shutil
import tempfile
import threading
from typing import Dict, Any, Optional

is_posix = os.name != "nt"
tmp_base = "/tmp" if is_posix else os.environ.get("TEMP", os.environ.get("TMP", "C:/tmp"))
if is_posix and os.path.exists(tmp_base) and os.access(tmp_base, os.W_OK):
    tempfile.tempdir = tmp_base
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Response

from backend.translator import (
    convert_to_wav_16k_mono,
    get_audio_duration_seconds,
    transcribe,
    translate_text,
    synthesize,
    convert_mp3_to_wav,
    list_supported_languages,
    SUPPORTED_LANGUAGES,
    ALLOWED_EXTENSIONS
)

router = APIRouter(tags=["Audio Translator"])

# ---------------------------------------------------------------------------
# In-Memory Job Storage & TTL Cleanup
# ---------------------------------------------------------------------------
MAX_TRANSLATE_JOBS = 50
JOB_TTL_SECONDS = 3600  # 1 hour expiration
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max upload
MAX_DURATION_SECONDS = 300.0  # 5 minutes max duration

TRANSLATE_JOBS: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def cleanup_old_jobs():
    """Prunes expired jobs (>1 hour) and bounds storage to MAX_TRANSLATE_JOBS."""
    now = time.time()
    with _jobs_lock:
        expired_keys = [
            jid for jid, job in TRANSLATE_JOBS.items()
            if now - job.get("created_at", now) > JOB_TTL_SECONDS
        ]
        for jid in expired_keys:
            job = TRANSLATE_JOBS.pop(jid, None)
            if job and job.get("temp_dir") and os.path.isdir(job["temp_dir"]):
                shutil.rmtree(job["temp_dir"], ignore_errors=True)

        # If still over capacity, remove oldest
        if len(TRANSLATE_JOBS) >= MAX_TRANSLATE_JOBS:
            sorted_keys = sorted(
                TRANSLATE_JOBS.keys(),
                key=lambda k: TRANSLATE_JOBS[k].get("created_at", 0)
            )
            for jid in sorted_keys[:10]:
                job = TRANSLATE_JOBS.pop(jid, None)
                if job and job.get("temp_dir") and os.path.isdir(job["temp_dir"]):
                    shutil.rmtree(job["temp_dir"], ignore_errors=True)


# ---------------------------------------------------------------------------
# Background Pipeline Execution
# ---------------------------------------------------------------------------
def run_translation_pipeline(job_id: str, input_bytes: bytes, original_filename: str, target_lang: str):
    """
    Executes the multi-stage audio translation pipeline in a background thread:
    1. Converting: audio format decode & resample to 16 kHz mono WAV via ffmpeg.
    2. Transcribing: Speech-to-text & language detection via faster-whisper.
    3. Translating: Text translation to target language.
    4. Synthesizing: Neural text-to-speech via edge-tts.
    5. Finalizing: Audio packaging and cleanup.
    """
    with _jobs_lock:
        job = TRANSLATE_JOBS.get(job_id)
    if not job:
        return

    temp_wav_path = None
    temp_dir = job.get("temp_dir")

    try:
        # Step 1: Converting
        with _jobs_lock:
            job["status"] = "converting"
            job["progress"] = 15

        temp_wav_path = convert_to_wav_16k_mono(input_bytes, original_filename)

        # Check duration limit (max 5 minutes)
        duration = get_audio_duration_seconds(temp_wav_path)
        if duration > MAX_DURATION_SECONDS:
            raise ValueError(
                f"Audio duration of {duration:.1f}s exceeds the maximum limit of {int(MAX_DURATION_SECONDS)}s (5 minutes)."
            )

        # Step 2: Transcribing
        with _jobs_lock:
            job["status"] = "transcribing"
            job["progress"] = 40

        transcript, detected_lang, confidence = transcribe(temp_wav_path)

        with _jobs_lock:
            job["detected_language"] = detected_lang
            job["detected_language_confidence"] = round(confidence, 3)
            job["transcript"] = transcript

        # Step 3: Translating
        with _jobs_lock:
            job["status"] = "translating"
            job["progress"] = 65

        translated_text = translate_text(
            text=transcript,
            source_lang=detected_lang,
            target_lang=target_lang
        )

        with _jobs_lock:
            job["translated_text"] = translated_text

        # Step 4: Synthesizing
        with _jobs_lock:
            job["status"] = "synthesizing"
            job["progress"] = 85

        mp3_audio = synthesize(
            text=translated_text,
            target_lang=target_lang
        )

        # Save result audio into temp_dir
        result_filename = f"translated_{job_id}.mp3"
        result_path = os.path.join(temp_dir, result_filename)
        with open(result_path, "wb") as f:
            f.write(mp3_audio)

        # Step 5: Done
        with _jobs_lock:
            job["status"] = "done"
            job["progress"] = 100
            job["result_audio_bytes"] = mp3_audio
            job["result_audio_path"] = result_path
            job["result_media_type"] = "audio/mpeg"

    except Exception as exc:
        error_msg = str(exc)
        with _jobs_lock:
            job["status"] = "error"
            job["error"] = error_msg
            job["progress"] = 0
    finally:
        # Clean up intermediate WAV file
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                # If temp_wav_path was created inside a dedicated conversion dir, remove it
                conv_dir = os.path.dirname(temp_wav_path)
                if conv_dir != temp_dir and os.path.isdir(conv_dir):
                    shutil.rmtree(conv_dir, ignore_errors=True)
                else:
                    os.remove(temp_wav_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@router.get("/languages")
def get_supported_languages():
    """Returns the list of supported target languages with default voices."""
    return {"languages": list_supported_languages()}


@router.post("/start")
async def start_translation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_lang: str = Form(...)
):
    """
    Accepts an audio file and target language, validates constraints, and starts
    the background translation process.
    """
    cleanup_old_jobs()

    # 1. Validate target language
    code = target_lang.lower().strip()
    if code not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language '{target_lang}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
        )

    # 2. Validate filename and extension
    original_filename = file.filename or "audio.wav"
    _, ext = os.path.splitext(original_filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 3. Read payload and validate size
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file payload.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum 25MB limit ({len(contents) / (1024 * 1024):.1f} MB uploaded)."
        )

    # 4. Create Job Record
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix=f"trans_job_{job_id[:8]}_")

    with _jobs_lock:
        TRANSLATE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "target_language": code,
            "original_filename": original_filename,
            "detected_language": None,
            "detected_language_confidence": None,
            "transcript": None,
            "translated_text": None,
            "result_audio_bytes": None,
            "result_audio_path": None,
            "result_media_type": "audio/mpeg",
            "error": None,
            "created_at": time.time(),
            "temp_dir": temp_dir
        }

    # 5. Enqueue background task
    background_tasks.add_task(
        run_translation_pipeline,
        job_id,
        contents,
        original_filename,
        code
    )

    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Polls the status of an ongoing or completed audio translation job.
    """
    with _jobs_lock:
        job = TRANSLATE_JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Translation job not found or expired.")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "target_language": job["target_language"],
        "detected_language": job.get("detected_language"),
        "detected_language_confidence": job.get("detected_language_confidence"),
        "transcript": job.get("transcript"),
        "translated_text": job.get("translated_text"),
        "error": job.get("error")
    }


@router.get("/result/{job_id}")
def get_job_result(job_id: str):
    """
    Streams the synthesized audio result for a completed translation job.
    """
    with _jobs_lock:
        job = TRANSLATE_JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Translation job not found or expired.")

    if job["status"] == "error":
        raise HTTPException(status_code=400, detail=f"Job failed with error: {job.get('error')}")

    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job is not yet complete (current status: {job['status']}).")

    audio_bytes = job.get("result_audio_bytes")
    audio_path = job.get("result_audio_path")
    media_type = job.get("result_media_type", "audio/mpeg")

    if audio_bytes is None and audio_path and os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Synthesized audio result is unavailable.")

    ext = "mp3" if "mpeg" in media_type else "wav"
    headers = {
        "Content-Disposition": f'inline; filename="translated_{job_id}.{ext}"',
        "Cache-Control": "no-cache"
    }

    return Response(content=audio_bytes, media_type=media_type, headers=headers)

