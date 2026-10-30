"""
translator.py - Pure logic module for audio translation pipeline.

Provides:
1. convert_to_wav_16k_mono: Validates and converts input audio to 16 kHz mono 16-bit PCM WAV using ffmpeg.
2. transcribe: Transcribes speech using faster-whisper with lazy model caching and language detection.
3. translate_text: Translates text across languages with sentence boundary chunking, retries, and same-language skip.
4. synthesize: Synthesizes translated text to audio using edge-tts.
5. list_supported_languages: Lists supported languages and default neural voices.
"""

import os
import sys

# Prevent OpenMP runtime crash (OMP: Error #15) on Windows / Anaconda environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import time
import shutil
import asyncio
import tempfile
import threading
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Protocol
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Supported Languages and Voice Mappings
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "voice": "en-US-JennyNeural"},
    "hi": {"name": "Hindi", "voice": "hi-IN-SwaraNeural"},
    "mr": {"name": "Marathi", "voice": "mr-IN-AarohiNeural"},
    "ta": {"name": "Tamil", "voice": "ta-IN-PallaviNeural"},
    "te": {"name": "Telugu", "voice": "te-IN-ShrutiNeural"},
    "bn": {"name": "Bengali", "voice": "bn-IN-TanishaaNeural"},
    "gu": {"name": "Gujarati", "voice": "gu-IN-DhwaniNeural"},
    "kn": {"name": "Kannada", "voice": "kn-IN-SapnaNeural"},
    "ml": {"name": "Malayalam", "voice": "ml-IN-SobhanaNeural"},
    "pa": {"name": "Punjabi", "voice": "pa-IN-GurpreetNeural"},
    "ur": {"name": "Urdu", "voice": "ur-PK-UzmaNeural"},
    "es": {"name": "Spanish", "voice": "es-ES-ElviraNeural"},
    "fr": {"name": "French", "voice": "fr-FR-DeniseNeural"},
    "de": {"name": "German", "voice": "de-DE-KatjaNeural"},
    "it": {"name": "Italian", "voice": "it-IT-ElsaNeural"},
    "pt": {"name": "Portuguese", "voice": "pt-BR-FranciscaNeural"},
    "ru": {"name": "Russian", "voice": "ru-RU-SvetlanaNeural"},
    "ja": {"name": "Japanese", "voice": "ja-JP-NanamiNeural"},
    "ko": {"name": "Korean", "voice": "ko-KR-SunHiNeural"},
    "zh": {"name": "Chinese", "voice": "zh-CN-XiaoxiaoNeural"},
    "ar": {"name": "Arabic", "voice": "ar-SA-ZariyahNeural"},
}

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def list_supported_languages() -> List[Dict[str, str]]:
    """Returns a list of supported languages with code, display name, and default TTS voice."""
    return [
        {"code": code, "name": meta["name"], "voice": meta["voice"]}
        for code, meta in SUPPORTED_LANGUAGES.items()
    ]


# ---------------------------------------------------------------------------
# 1. Audio Conversion via FFmpeg with Pure-Python WAV Fallback
# ---------------------------------------------------------------------------
def get_ffmpeg_cmd() -> Optional[str]:
    """
    Discovers the FFmpeg executable across multiple sources:
    1. System PATH ('ffmpeg' or 'ffmpeg.exe')
    2. imageio-ffmpeg bundled binary (pip install imageio-ffmpeg)
    3. FFMPEG_PATH / FFMPEG_BINARY environment variables
    4. Common Windows install locations (WinGet, Scoop, Chocolatey, Program Files)
    """
    # 1. System PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    # 3. Environment variables
    for env_var in ("FFMPEG_PATH", "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        p = os.environ.get(env_var)
        if p and os.path.exists(p):
            return p

    # 4. Common Windows directories
    win_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"),
    ]
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_dir = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
        if os.path.isdir(winget_dir):
            for root, _, files in os.walk(winget_dir):
                if "ffmpeg.exe" in files:
                    win_paths.append(os.path.join(root, "ffmpeg.exe"))

    for path in win_paths:
        if os.path.exists(path):
            return path

    return None


def _convert_wav_fallback_pure_python(input_bytes: bytes) -> str:
    """
    Pure Python + NumPy fallback for WAV files when FFmpeg is not installed.
    Loads WAV, downmixes multi-channel audio to mono, resamples to 16 kHz using
    NumPy interpolation, and writes a 16 kHz 16-bit PCM WAV temporary file.
    """
    import io
    import wave
    import numpy as np

    buffer = io.BytesIO(input_bytes)
    try:
        with wave.open(buffer, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_frames = wf.readframes(n_frames)
    except Exception as e:
        raise ValueError(f"Failed to decode WAV file: {e}")

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 4:
        dtype = np.int32
    else:
        dtype = np.int16

    signal = np.frombuffer(raw_frames, dtype=dtype)
    if sampwidth == 1:
        signal = (signal.astype(np.int32) - 128) * 256
        signal = signal.astype(np.int16)
    elif sampwidth == 4:
        signal = (signal / 65536).astype(np.int16)

    # Downmix to mono
    if n_channels > 1:
        signal = signal[::n_channels]

    # Resample to 16,000 Hz if needed
    if sample_rate != 16000 and len(signal) > 0:
        duration = len(signal) / float(sample_rate)
        target_len = int(duration * 16000)
        orig_indices = np.linspace(0, duration, len(signal), endpoint=False)
        target_indices = np.linspace(0, duration, target_len, endpoint=False)
        signal = np.interp(target_indices, orig_indices, signal).astype(np.int16)
    else:
        signal = signal.astype(np.int16)

    temp_dir = tempfile.mkdtemp(prefix="audio_trans_wav_")
    output_path = os.path.join(temp_dir, "converted_16k_mono.wav")
    with wave.open(output_path, "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(16000)
        out_wf.writeframes(signal.tobytes())

    return output_path


def _convert_audio_with_pyav(input_bytes: bytes) -> str:
    """
    Decodes audio (MP3, M4A, OGG, FLAC, WEBM) directly in Python using PyAV,
    resampling to 16 kHz mono 16-bit PCM WAV. Does not require external ffmpeg CLI.
    """
    import io
    import wave
    import av

    _patch_pyav_metadata_errors()
    buffer = io.BytesIO(input_bytes)
    container = av.open(buffer)
    if not container.streams.audio:
        raise ValueError("Audio stream not found in container.")
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    frames_bytes = []
    for frame in container.decode(stream):
        for r_frame in resampler.resample(frame):
            frames_bytes.append(r_frame.to_ndarray().tobytes())

    raw_pcm = b"".join(frames_bytes)
    if not raw_pcm:
        raise ValueError("Decoded audio stream contains no audio data.")

    temp_dir = tempfile.mkdtemp(prefix="audio_trans_pyav_")
    output_path = os.path.join(temp_dir, "converted_16k_mono.wav")
    with wave.open(output_path, "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(16000)
        out_wf.writeframes(raw_pcm)

    return output_path


def convert_to_wav_16k_mono(input_bytes: bytes, filename: str) -> str:
    """
    Validates input audio and converts it to a 16 kHz mono 16-bit PCM WAV file.
    Uses FFmpeg if available, and gracefully falls back to pure Python/NumPy for WAV files.

    Args:
        input_bytes: Raw binary audio data.
        filename: Original user filename (used to check extension, not trusted for disk path).

    Returns:
        Path to the converted 16 kHz mono WAV temporary file.

    Raises:
        RuntimeError: If ffmpeg is missing and input is not a WAV file.
        ValueError: If file type is unsupported or if audio cannot be decoded.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if not input_bytes:
        raise ValueError("Audio payload is empty.")

    ffmpeg_bin = get_ffmpeg_cmd()

    # If FFmpeg is missing, attempt pure-Python fallback for WAV files or PyAV for encoded audio
    if not ffmpeg_bin:
        if ext == ".wav":
            return _convert_wav_fallback_pure_python(input_bytes)
        try:
            return _convert_audio_with_pyav(input_bytes)
        except Exception:
            raise RuntimeError(
                "ffmpeg is not installed or not found in system PATH. "
                "Please install ffmpeg (e.g. run 'winget install Gyan.FFmpeg'), "
                "or upload a .wav audio file."
            )

    # Create temporary directory for safe execution
    temp_dir = tempfile.mkdtemp(prefix="audio_trans_")
    input_path = os.path.join(temp_dir, f"input{ext}")
    output_path = os.path.join(temp_dir, "converted_16k_mono.wav")

    try:
        with open(input_path, "wb") as f:
            f.write(input_bytes)

        # Run ffmpeg to convert to 16 kHz, 1 channel, 16-bit PCM WAV
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            output_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            err_msg = result.stderr.strip() or "Unknown ffmpeg decoding error."
            lines = [line for line in err_msg.splitlines() if "Error" in line or "Invalid" in line]
            detail = lines[-1] if lines else err_msg[-200:]
            raise ValueError(f"Failed to decode audio file. Corrupted or invalid audio: {detail}")

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("Audio conversion produced an empty or missing output file.")

        return output_path

    finally:
        # Clean up input temporary file to save disk space
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


def get_audio_duration_seconds(file_path: str) -> float:
    """
    Returns audio duration in seconds using wave module for WAV files,
    or ffprobe / ffmpeg if available.
    """
    # 1. Try pure-Python wave module (instant for WAV files)
    try:
        import wave
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return float(frames) / float(rate)
    except Exception:
        pass

    # 2. Try ffprobe if found
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
        except Exception:
            pass

    # 3. Try ffmpeg duration output
    ffmpeg_bin = get_ffmpeg_cmd()
    if ffmpeg_bin:
        cmd = [ffmpeg_bin, "-i", file_path]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = res.stderr or ""
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
            if match:
                hours, mins, secs = match.groups()
                return float(hours) * 3600 + float(mins) * 60 + float(secs)
        except Exception:
            pass

    # 4. Fallback to file size estimate for 16 kHz 16-bit mono PCM (32000 bytes/sec)
    if file_path.endswith(".wav") and os.path.exists(file_path):
        size = os.path.getsize(file_path)
        return max(0.0, (size - 44) / 32000.0)

    return 0.0


# ---------------------------------------------------------------------------
# 2. Speech-to-Text via faster-whisper (Lazy Thread-Safe Cache)
# ---------------------------------------------------------------------------
def _patch_pyav_metadata_errors():
    """
    Fixes compatibility issue between faster-whisper and PyAV >= 14.0.0.
    In PyAV 14+, 'metadata_errors' was removed from av.open().
    faster-whisper's decode_audio() calls av.open(..., metadata_errors="ignore").
    This patch strips 'metadata_errors' when av.open rejects it, preventing
    TypeError: open() got an unexpected keyword argument 'metadata_errors'.
    """
    try:
        import av
        if getattr(av, "_patched_for_metadata_errors", False):
            return

        orig_open = av.open

        def _safe_av_open(*args, **kwargs):
            try:
                return orig_open(*args, **kwargs)
            except TypeError as exc:
                if "metadata_errors" in str(exc) and "metadata_errors" in kwargs:
                    kwargs.pop("metadata_errors", None)
                    return orig_open(*args, **kwargs)
                raise

        av.open = _safe_av_open
        av._patched_for_metadata_errors = True

        # Also patch faster_whisper.audio.av if already imported
        if "faster_whisper.audio" in sys.modules:
            fw_audio = sys.modules["faster_whisper.audio"]
            if hasattr(fw_audio, "av"):
                fw_audio.av.open = _safe_av_open
    except Exception:
        pass


# Apply immediately
_patch_pyav_metadata_errors()

_model_lock = threading.Lock()
_cached_whisper_model = None


def get_whisper_model():
    """Lazily loads and caches the faster-whisper model in a thread-safe manner."""
    _patch_pyav_metadata_errors()
    global _cached_whisper_model
    if _cached_whisper_model is None:
        with _model_lock:
            if _cached_whisper_model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as e:
                    raise RuntimeError(
                        "faster-whisper is not installed. Please install faster-whisper: "
                        "pip install faster-whisper"
                    ) from e

                model_size = os.environ.get("WHISPER_MODEL", "base")
                compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
                device = os.environ.get("WHISPER_DEVICE", "cpu")
                cpu_threads = min(4, os.cpu_count() or 1)

                # Ensure download_root is writable in serverless environments (e.g. Vercel, AWS Lambda)
                download_root = os.environ.get("WHISPER_DOWNLOAD_ROOT")
                if not download_root and (os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")):
                    download_root = "/tmp/whisper_models"

                _cached_whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    download_root=download_root
                )
    return _cached_whisper_model


def transcribe(wav_path: str) -> Tuple[str, str, float]:
    """
    Transcribes a 16 kHz WAV file using faster-whisper with automatic language detection.
    Optimized for low-latency CPU inference via greedy decoding (beam_size=1) and VAD filtering.

    Args:
        wav_path: Path to the 16 kHz mono WAV file.

    Returns:
        Tuple of (transcript_text, detected_language_code, confidence_score)

    Raises:
        ValueError: If no speech could be detected.
        RuntimeError: If model inference fails.
    """
    _patch_pyav_metadata_errors()
    model = get_whisper_model()
    # Fast greedy decoding (beam_size=1) + Voice Activity Detection skips silent frames
    segments, info = model.transcribe(
        wav_path,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    collected_segments = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            collected_segments.append(text)

    transcript = " ".join(collected_segments).strip()
    if not transcript:
        raise ValueError("No speech could be detected in the provided audio file.")

    detected_lang = getattr(info, "language", "") or "en"
    confidence = float(getattr(info, "language_probability", 1.0))

    return transcript, detected_lang.lower(), confidence


# ---------------------------------------------------------------------------
# 3. Text Translation Provider Interface
# ---------------------------------------------------------------------------
class TranslationProvider(ABC):
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translates text from source_lang to target_lang."""
        pass


class GoogleTranslatorProvider(TranslationProvider):
    """
    High-speed, resilient translation provider with multi-tier fallback:
    Tier 1: Direct Google Translate GTX endpoint (sub-200ms, no 429 rate limit).
    Tier 2: deep-translator GoogleTranslator.
    Tier 3: deep-translator MyMemoryTranslator.
    """

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        src = source_lang.lower().strip() if source_lang else "auto"
        tgt = target_lang.lower().strip()

        # Tier 1: Fast direct Google Translate GTX API (instantaneous)
        try:
            import urllib.request
            import urllib.parse
            import json

            encoded_text = urllib.parse.quote(text)
            url = (
                f"https://translate.googleapis.com/translate_a/single?"
                f"client=gtx&sl={src}&tl={tgt}&dt=t&q={encoded_text}"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=4.0) as response:
                if response.status == 200:
                    raw_json = json.loads(response.read().decode("utf-8"))
                    if raw_json and isinstance(raw_json, list) and len(raw_json) > 0:
                        parts = [seg[0] for seg in raw_json[0] if seg and len(seg) > 0 and seg[0]]
                        translated = "".join(parts).strip()
                        if translated:
                            return translated
        except Exception:
            pass

        # Tier 2: deep-translator GoogleTranslator fallback
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=src, target=tgt).translate(text)
        except Exception:
            pass

        # Tier 3: deep-translator MyMemoryTranslator fallback
        try:
            from deep_translator import MyMemoryTranslator
            return MyMemoryTranslator(source=src, target=tgt).translate(text)
        except Exception as e:
            raise RuntimeError(f"All translation providers failed: {e}")


_default_translation_provider: TranslationProvider = GoogleTranslatorProvider()


def set_translation_provider(provider: TranslationProvider):
    """Allows swapping the translation provider implementation (e.g. for testing or enterprise APIs)."""
    global _default_translation_provider
    _default_translation_provider = provider


def _chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """
    Splits long text into sentence-bounded chunks of at most max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    # Split by common sentence delimiters while keeping the punctuation
    sentence_endings = re.compile(r'([.?!।\n]+)')
    tokens = sentence_endings.split(text)

    sentences = []
    current_sentence = ""
    for i in range(0, len(tokens) - 1, 2):
        current_sentence = tokens[i] + tokens[i + 1]
        sentences.append(current_sentence.strip())
    if len(tokens) % 2 == 1 and tokens[-1].strip():
        sentences.append(tokens[-1].strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single sentence exceeds max_chars, split by words
            if len(sentence) > max_chars:
                words = sentence.split()
                sub_chunk = ""
                for w in words:
                    if len(sub_chunk) + len(w) + 1 <= max_chars:
                        sub_chunk = f"{sub_chunk} {w}".strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = w
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text]


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    provider: Optional[TranslationProvider] = None,
    max_chars: int = 4000
) -> str:
    """
    Translates text with sentence boundary chunking, retries with exponential backoff,
    and same-language skip.

    Args:
        text: Input text string.
        source_lang: Detected or specified source language code (e.g. 'hi', 'en').
        target_lang: Target language code (e.g. 'mr', 'en').
        provider: Optional custom TranslationProvider (defaults to GoogleTranslatorProvider).
        max_chars: Maximum character length per translation chunk (default 4000).

    Returns:
        Translated text string.

    Raises:
        RuntimeError: If translation service is unreachable after retries.
    """
    clean_text = text.strip()
    if not clean_text:
        return ""

    src = source_lang.strip().lower()
    tgt = target_lang.strip().lower()

    # Skip translation if source matches target
    if src == tgt:
        return clean_text

    active_provider = provider or _default_translation_provider
    chunks = _chunk_text(clean_text, max_chars=max_chars)
    translated_chunks = []

    for chunk in chunks:
        # Retry with exponential backoff
        max_retries = 3
        backoff_sec = 0.5
        success = False
        last_error = None

        for attempt in range(max_retries):
            try:
                res = active_provider.translate(chunk, src, tgt)
                if res:
                    translated_chunks.append(res.strip())
                success = True
                break
            except Exception as e:
                last_error = e
                time.sleep(backoff_sec)
                backoff_sec *= 2.0

        if not success:
            raise RuntimeError(
                f"Translation service unreachable or failed after {max_retries} attempts: {last_error}"
            )

    return " ".join(translated_chunks)


# ---------------------------------------------------------------------------
# 4. Text-to-Speech via edge-tts
# ---------------------------------------------------------------------------
async def _edge_tts_synthesize(text: str, voice: str) -> bytes:
    """Async helper to stream audio bytes from edge-tts."""
    try:
        import edge_tts
    except ImportError as e:
        raise RuntimeError(
            "edge-tts is not installed. Please install edge-tts: pip install edge-tts"
        ) from e

    communicate = edge_tts.Communicate(text, voice)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_chunks.append(chunk["data"])

    if not audio_chunks:
        raise RuntimeError(f"edge-tts produced empty audio for voice '{voice}'.")

    return b"".join(audio_chunks)


def synthesize(text: str, target_lang: str, voice: Optional[str] = None) -> bytes:
    """
    Synthesizes text into MP3 audio bytes using edge-tts.

    Args:
        text: Text to synthesize.
        target_lang: Target language code (must be in SUPPORTED_LANGUAGES).
        voice: Optional specific edge-tts voice name (defaults to mapping for target_lang).

    Returns:
        MP3 audio bytes.

    Raises:
        ValueError: If target_lang is unsupported or text is empty.
        RuntimeError: If synthesis fails.
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Cannot synthesize empty text.")

    code = target_lang.lower().strip()
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Target language '{target_lang}' is not supported. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
        )

    selected_voice = voice or SUPPORTED_LANGUAGES[code]["voice"]

    # Run the async coroutine safely in a new or existing event loop
    try:
        # Check if an event loop is already active in this thread
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside an already running event loop (e.g. FastAPI async thread) -> run in worker thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            def _run():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(_edge_tts_synthesize(clean_text, selected_voice))
                finally:
                    new_loop.close()
            return pool.submit(_run).result()
    else:
        # Standard synchronous execution
        return asyncio.run(_edge_tts_synthesize(clean_text, selected_voice))


def convert_mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """
    Converts MP3 audio bytes to standard WAV bytes using ffmpeg.
    Useful if frontend or downstream DSP components require WAV format.
    """
    ffmpeg_bin = get_ffmpeg_cmd()
    if not ffmpeg_bin:
        return mp3_bytes

    temp_dir = tempfile.mkdtemp(prefix="mp3_to_wav_")
    mp3_path = os.path.join(temp_dir, "input.mp3")
    wav_path = os.path.join(temp_dir, "output.wav")

    try:
        with open(mp3_path, "wb") as f:
            f.write(mp3_bytes)

        cmd = [ffmpeg_bin, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", wav_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(wav_path):
            with open(wav_path, "rb") as f:
                return f.read()
        return mp3_bytes
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
