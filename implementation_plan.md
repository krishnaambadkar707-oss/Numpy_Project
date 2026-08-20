# Implementation Plan - Audio Signal Processor Web Application

Transform the existing NumPy audio signal processing script (`numpy_project.py`) into a modern, full-stack Web Application featuring a Python FastAPI backend and a interactive, glassmorphism dark-themed frontend dashboard.

## User Review Required

> [!IMPORTANT]
> - **Backend Framework**: We propose using **FastAPI** with `uvicorn` and `numpy` for fast asynchronous REST API endpoints.
> - **Demo Sample Generator**: To make testing effortless, the application includes a built-in "Generate Sample WAV (Sine Wave + Noise)" feature so users can test immediately without providing their own WAV file.
> - **Frontend Stack**: Built with clean Vanilla HTML5, CSS3 (Glassmorphism design system, CSS variables, dark mode styling), and Canvas/Chart.js for interactive waveform and frequency spectrum plots.

## Proposed Changes

### Backend Component (`backend/`)

#### [NEW] [audio_processor.py](file:///c:/Users/krish/OneDrive/Desktop/NUMPY/Project/backend/audio_processor.py)
- Modularized audio processing engine leveraging NumPy.
- Implement WAV loading (`load_wav_from_bytes`), WAV saving (`export_wav_bytes`), noise reduction (`reduce_noise`), moving average smoothing (`smooth_signal`), FFT frequency analysis (`frequency_analysis`), signal statistics (`analyze_signal`), MSE comparison (`compare_signals`), and sample signal generator (`generate_demo_wav`).

#### [NEW] [main.py](file:///c:/Users/krish/OneDrive/Desktop/NUMPY/Project/backend/main.py)
- FastAPI web server exposing the following API routes:
  - `POST /api/upload`: Upload and validate `.wav` audio files.
  - `POST /api/process`: Process uploaded/generated audio with configurable `threshold` and `window_size` parameters. Returns JSON waveform data downsampled for rendering, FFT spectrum data, statistical metrics, and MSE.
  - `GET /api/demo-audio`: Generate and return a synthetic 440 Hz audio signal with artificial high-frequency noise.
  - `GET /api/audio/{file_id}`: Stream audio file for HTML5 `<audio>` player.

#### [NEW] [requirements.txt](file:///c:/Users/krish/OneDrive/Desktop/NUMPY/Project/requirements.txt)
- Dependencies: `fastapi`, `uvicorn`, `numpy`, `python-multipart`.

---

### Frontend Component (`frontend/`)

#### [NEW] [index.html](file:///c:/Users/krish/OneDrive/Desktop/NUMPY/Project/frontend/index.html)
- Single Page Application (SPA) dashboard layout.
- Sections:
  - **Header & Navbar**: Title, status badges, quick help tooltips.
  - **Audio Source Selection**: Drag-and-drop file upload area + "Generate Demo WAV" button.
  - **Parameter Control Panel**: Sliders for Noise Reduction Threshold (0-2000) and Smoothing Window Size (1-31).
  - **Audio Playback Cards**: Side-by-side HTML5 audio players for original vs processed audio.
  - **Analytics Grid**: Metric cards displaying Max Amplitude, Mean Value, Variance, and MSE.
  - **Visualization Canvas**: Dual interactive charts for Waveform comparison and Frequency Spectrum (FFT).

#### [NEW] [style.css](file:///c:/Users/krish/OneDrive/Desktop/NUMPY/Project/frontend/style.css)
- Dark mode glassmorphism visual aesthetics (`backdrop-filter`, glowing accents, modern typography).
- Responsive grid and flexbox layout for mobile and desktop screens.
- Smooth transitions and hover animations for interactive controls.

#### [NEW] [app.js](file:///c:/Users/krish/OneDrive/Desktop/NUMPY/Project/frontend/app.js)
- Event listeners for file upload, demo generation, and slider updates.
- Canvas-based custom rendering for smooth waveform and FFT visualization.
- Asynchronous API client fetching processed audio and updating metrics in real-time.

---

### Integration & Execution (`server.py`)

#### [NEW] [server.py](file:///c:/Users/krish/OneDrive/Desktop/NUMPY/Project/server.py)
- Unified application runner that serves both backend API and static frontend assets via FastAPI / Uvicorn on local port 8000.

## Verification Plan

### Automated Tests
- Run Python unit tests for `audio_processor.py` (verifying FFT shape, noise reduction thresholding, smooth signal convolution, MSE computation).
- Verify FastAPI endpoint response status codes and payload structure.

### Manual Verification
- Launch the full-stack server using `python server.py`.
- Test demo audio generation, upload custom WAV files, adjust threshold and window size sliders, verify real-time plot updates and audio playback quality.
