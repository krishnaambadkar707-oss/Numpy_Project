# Walkthrough - Full-Stack Audio Signal Processor Web Application

We have successfully analyzed `numpy_project.py` and transformed it into a full-stack Web Application featuring a **Python FastAPI backend** and a modern **Glassmorphism HTML5/CSS3/JS frontend dashboard**.

---

## 🛠️ Architecture & Components Created

```
Project/
├── backend/
│   ├── __init__.py
│   ├── audio_processor.py   # Core NumPy DSP functions (FFT, MSE, noise reduction, smoothing)
│   └── main.py              # FastAPI REST API endpoints
├── frontend/
│   ├── index.html           # SPA Dashboard HTML structure
│   ├── style.css            # Dark mode glassmorphism design system & visual styling
│   └── app.js               # Canvas waveform/FFT renderer, audio players, & API client
├── numpy_project.py         # Original standalone CLI script (preserved)
├── requirements.txt         # Project dependencies (FastAPI, Uvicorn, NumPy, python-multipart)
└── server.py                # Unified server entry point serving API and static frontend UI
```

---

## ⚙️ Key Technical Features

### 1. Backend (`backend/audio_processor.py` & `backend/main.py`)
- **NumPy DSP Engine**:
  - `load_wav_from_bytes`: Reads 16-bit PCM WAV bytes into NumPy arrays, auto-converting stereo to mono.
  - `reduce_noise`: Applies absolute threshold filtering to silence noise floor.
  - `smooth_signal`: Applies moving average kernel convolution (`np.convolve`).
  - `frequency_analysis`: Computes Fast Fourier Transform (`np.fft.fft` & `np.fft.fftfreq`) for frequency spectrum magnitude visualization.
  - `compare_signals`: Calculates Mean Squared Error (MSE) between original and processed audio.
  - `generate_demo_wav`: Synthesizes a 440 Hz fundamental tone + harmonics + artificial noise for instant one-click testing.
- **RESTful Endpoints**:
  - `GET /api/health` -> Health check status.
  - `GET /api/demo` -> Generates synthetic demo audio.
  - `POST /api/upload` -> Accepts `.wav` audio upload.
  - `POST /api/process` -> Executes real-time noise reduction & smoothing parameter updates.
  - `GET /api/audio/{id}/{type}` -> Streams 16-bit PCM WAV bytes for HTML5 audio players.

### 2. Frontend (`frontend/index.html`, `style.css`, `app.js`)
- **Glassmorphism Dark UI**: Vibrant neon cyan/violet accents, backdrop filters, responsive grid layout.
- **Drag-and-Drop WAV Uploader**: Interactive dropzone with instant file metadata breakdown (Sample Rate, Duration, Samples, Channels).
- **One-Click Demo Generator**: Instant testing without needing local `.wav` files.
- **Real-Time Sliders**: Interactive controls for Noise Gate Threshold (`0 - 2000`) and Moving Average Window Size (`1 - 31`).
- **Side-by-Side Audio Players**: HTML5 audio players for instant before vs after sound comparison.
- **Analytics Metric Cards**: Real-time breakdown of Peak Amplitude, Mean DC Offset, Signal Variance, and MSE.
- **Canvas Visualizations**:
  - **Dual Waveform Graph**: Overlaid rendering of original signal (cyan) vs processed signal (glowing violet).
  - **FFT Frequency Spectrum**: Rendered Fast Fourier Transform spectral magnitude curve across frequency bins.

---

## 🧪 Verification & Test Results

1. **NumPy Signal Processor Unit Test**: Verified signal generation, noise reduction, moving average convolution, FFT magnitude calculation, and MSE calculation.
2. **FastAPI Route Testing**: Executed end-to-end endpoint tests for file upload, audio processing, demo audio generation, and binary WAV streaming.
3. **Full-Stack Static Mounting**: Verified that `server.py` correctly mounts static frontend assets (`index.html`, `style.css`, `app.js`) and API routes.

---

## 🚀 How to Run the Web Application

1. Open your terminal in the project directory:
   ```bash
   python server.py
   ```

2. Open your browser and navigate to:
   ```text
   http://127.0.0.1:8000
   ```

3. Click **"Generate Demo Audio (Sine + Noise)"** or upload your own `.wav` file to test noise reduction, smoothing filters, FFT spectrum graphs, and MSE metrics in real time!
