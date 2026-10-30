# Audio Signal Processor & Translator Studio

A full-stack, real-time Audio Signal Processing (DSP) and Multi-Language Audio Translation application powered by **NumPy**, **FastAPI**, **faster-whisper**, **deep-translator**, and **edge-tts**.

---

## 🌟 Key Features

### 1. NumPy-Powered DSP Studio
- **Noise Reduction Gating**: Eliminates background hiss and ambient noise below an adjustable threshold using vectorized NumPy amplitude gating.
- **Moving-Average Smoothing**: Smooths sharp high-frequency transients using uniform 1D convolution (`numpy.convolve`).
- **FFT Spectrum Analysis**: Computes real-time Fast Fourier Transform (`numpy.fft.rfft`) magnitude spectra across frequency bins.
- **MSE Quality Analytics**: Computes Mean Squared Error (MSE), RMS, peak amplitude, and DC offset comparing original vs processed signals.
- **Interactive Visualizations**: High-performance HTML5 Canvas waveforms and frequency spectrum charts with hover inspection tooltips.
- **Zero-Dependency Client Fallback**: Operates fully in browser via JavaScript DSP if the backend server is offline.

### 2. Multi-Language Audio Translator
- **Universal Audio Ingestion**: Accepts `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, and `.webm` up to 25 MB and 5 minutes duration.
- **FFmpeg Ingestion Engine**: Normalizes and converts incoming audio streams into standardized 16 kHz mono 16-bit PCM WAV.
- **Automatic Language Detection & STT**: Employs `faster-whisper` (CTranslate2 INT8 engine) with lazy thread-safe model caching to transcribe audio and detect the spoken source language with confidence metrics.
- **Sentence-Bounded Translation**: Translates transcripts using sentence-boundary chunking (<4000 characters) and exponential backoff retry. Automatically skips translation if source language matches target language.
- **Neural Text-to-Speech**: Generates natural speech in the target language using Microsoft `edge-tts` across 21+ languages (English, Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Punjabi, Urdu, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic).
- **Asynchronous Background Processing**: Background worker with job lifecycle tracking (`queued` → `converting` → `transcribing` → `translating` → `synthesizing` → `done` / `error`), polling UI, and in-memory TTL cleanup.

---

## 🏗️ Architecture & Pipeline

```
[ User Audio (any format/lang) ]
               │
               ▼
   [ FFmpeg 16 kHz Mono WAV ]
               │
               ▼
 [ faster-whisper (STT + Lang Detect) ] ────► [ Source Language + Transcript ]
               │
               ▼
[ Sentence-Bounded Text Translation ] ────► [ Translated Text ]
               │
               ▼
    [ edge-tts (Neural Voice) ]
               │
               ▼
  [ Output Audio (MP3 / WAV) ] ───► [ Playback & Download in Browser ]
```

---

## 📡 REST API Reference

### DSP Endpoints
- `GET /api/health`: Health status.
- `GET /api/demo`: Generates a synthetic 3-second multi-tone demo signal with added Gaussian noise.
- `POST /api/upload?filename=...`: Uploads a WAV audio file (up to 25 MB) and processes with default DSP parameters.
- `POST /api/process`: Re-processes an audio session with custom threshold and window size parameters.
- `GET /api/audio/{audio_id}/{audio_type}`: Streams `original` or `processed` audio as `audio/wav`.

### Audio Translator Endpoints
- `GET /api/translate/languages`: Returns supported target languages, language codes, and default neural voices.
- `POST /api/translate/start`: Accepts multipart form (`file`, `target_lang`). Validates format, size, duration, and enqueues a background job. Returns `{"job_id": "...", "status": "queued"}`.
- `GET /api/translate/status/{job_id}`: Polls current job status (`queued`, `converting`, `transcribing`, `translating`, `synthesizing`, `done`, `error`), progress percentage (0-100), detected language, transcript, and translated text.
- `GET /api/translate/result/{job_id}`: Downloads or streams the translated audio result (`audio/mpeg`).

---

## ⚙️ Environment Variables & Memory Implications

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`. |
| `WHISPER_DEVICE` | `cpu` | Device: `cpu` or `cuda`. |
| `WHISPER_COMPUTE_TYPE` | `int8` | Precision: `int8`, `float16`, `float32`. |
| `HOST` | `127.0.0.1` (`0.0.0.0` in Docker) | Host binding IP. |
| `PORT` | `8000` (`10000` on Render) | Server port. |
| `ENV` | `development` (`production` in Docker) | Environment mode. |

### Memory Implications of `WHISPER_MODEL`
Faster-whisper loads weights into RAM using CTranslate2:
- **`tiny` / `tiny.en`**: ~39M parameters, ~500 MB RAM usage. Recommended for memory-constrained hosts (e.g., Render free tier with 512 MB RAM).
- **`base` (Default)**: ~74M parameters, ~1.0 GB RAM usage. Excellent balance of speed and accuracy for multi-language detection.
- **`small`**: ~244M parameters, ~2.0 GB RAM usage. Higher accuracy, best with at least 2 GB RAM.
- **`medium`**: ~769M parameters, ~4.5 GB RAM usage. Requires dedicated machine or GPU.

> **Tip for Render Free Tier:** Render free web services have 512 MB RAM. Set `WHISPER_MODEL=tiny` in `render.yaml` or Render dashboard to prevent OOM termination.

---

## 🚀 Running Locally

### 1. Prerequisites & Installing FFmpeg

FFmpeg is required for decoding incoming audio and converting to 16 kHz mono WAV.

#### Windows
Install via package managers or manual download:
```powershell
# Option A: Windows Package Manager (winget) - Recommended
winget install Gyan.FFmpeg

# Option B: Chocolatey
choco install ffmpeg

# Option C: Scoop
scoop install ffmpeg
```
After installation, verify in a new terminal:
```powershell
ffmpeg -version
```

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Debian / Ubuntu)
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

---

### 2. Python Setup

1. **Clone the repository and enter directory:**
   ```bash
   cd "Audio Signal Processor np_project"
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the studio server:**
   ```bash
   python server.py
   ```
   Open your browser at: **`http://127.0.0.1:8000`**

---

## 🧪 Running the Test Suites

Both test suites run completely offline with mocked ML/network modules:

```bash
# 1. Existing DSP and API test suite
python test_project.py

# 2. New Audio Translator test suite
python test_translate.py
```

---

## 🐳 Docker & Cloud Deployment

### Docker Build & Run
```bash
docker build -t audio-processor .
docker run -p 8000:10000 -e WHISPER_MODEL=tiny audio-processor
```

### Render Cloud Deployment
This project is pre-configured to deploy on Render seamlessly via either **Docker** (recommended) or **Native Python**:

1. **Option A: Docker Deployment (Recommended via `render.yaml`)**:
   - In Render, click **New > Blueprint** and select your repository.
   - Render automatically uses `render.yaml` with `runtime: docker` and sets `plan: free` with `WHISPER_MODEL=tiny` (optimized for Render's 512 MB RAM limit).
2. **Option B: Manual Web Service**:
   - In Render, click **New > Web Service**.
   - If using **Docker Runtime**:
     - Environment: `Docker`
   - If using **Python Runtime**:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `python server.py`
     - Add Environment Variable: `WHISPER_MODEL` = `tiny`
   - Render routes traffic to port `10000` automatically. The server automatically binds to `0.0.0.0:$PORT`.

---

## ⚠️ Known Limitations

1. **Generic Neural TTS Voice vs Speaker Voice Cloning**: The current pipeline synthesizes speech using Microsoft Edge neural voices matched to the target language rather than cloning the original speaker's timbre or pitch.
2. **Audio Duration & Timing Differences**: Target language syllables and cadence naturally differ in length from the source language (e.g., German translations are typically longer than English). The translated audio is a natural rendering of the text, not time-stretched to fit original word boundaries.
3. **Accuracy in Noisy Audio**: Automatic speech recognition accuracy decreases when speech is accompanied by severe background noise, heavy music, or cross-talk.
4. **Internet Connectivity for Translation & TTS**: Text translation and Edge-TTS voice synthesis require outbound internet connectivity to their respective cloud endpoints.

---

## 📋 Manual Test Checklist

| Scenario | Input File | Target Lang | Expected Result |
|---|---|---|---|
| **Hindi to English** | Spoken Hindi `.mp3` | `en` | Detects `hi`, transcribes Hindi, translates to English, plays English voice audio. |
| **English to Marathi** | Spoken English `.wav` | `mr` | Detects `en`, transcribes English, translates to Marathi, plays Marathi voice audio. |
| **Marathi to Hindi** | Spoken Marathi `.m4a` | `hi` | Detects `mr`, transcribes Marathi, translates to Hindi, plays Hindi voice audio. |
| **Same-Language Skip** | Spoken English `.wav` | `en` | Detects `en`, skips translation, synthesizes English speech directly. |
| **Silent Audio File** | 3s silence `.wav` | `es` | Graceful error: *"No speech could be detected in the provided audio file."* |
| **Oversized File (>25 MB)** | 30 MB audio file | `fr` | Blocked immediately with HTTP 413 *"File size exceeds maximum 25MB limit."* |
| **Corrupt Audio File** | Non-audio file renamed `.mp3` | `de` | FFmpeg decode fails gracefully with informative error message. |
