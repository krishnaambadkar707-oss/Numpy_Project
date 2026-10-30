import io
import wave
import numpy as np

def load_wav_from_bytes(wav_bytes: bytes):
    """
    Parses WAV file bytes and returns a numpy int16 array and sample rate.
    """
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        
        audio_data = wav.readframes(n_frames)
        
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 4:
        dtype = np.int32
    else:
        dtype = np.int16

    signal = np.frombuffer(audio_data, dtype=dtype)
    
    # If 8-bit uint8, convert to int16 centered at 0
    if sampwidth == 1:
        signal = (signal.astype(np.int32) - 128) * 256
        signal = signal.astype(np.int16)
    elif sampwidth == 4:
        signal = (signal / 65536).astype(np.int16)

    # Convert Stereo to Mono
    if n_channels > 1:
        signal = signal[::n_channels]

    return signal, sample_rate, n_channels, n_frames

def export_wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    """
    Converts a NumPy signal array into a 16-bit PCM WAV byte stream.
    """
    signal_int16 = np.clip(signal, -32768, 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(signal_int16.tobytes())
        
    return buffer.getvalue()

def analyze_signal(signal: np.ndarray):
    """
    Calculates key statistical metrics of the audio signal.
    """
    if len(signal) == 0:
        return {"max_amplitude": 0, "mean_value": 0.0, "variance": 0.0, "rms": 0.0}

    amplitude = int(np.max(np.abs(signal)))
    mean_val = float(np.mean(signal))
    variance = float(np.var(signal))
    rms = float(np.sqrt(np.mean(signal.astype(np.float64)**2)))

    return {
        "max_amplitude": amplitude,
        "mean_value": round(mean_val, 4),
        "variance": round(variance, 2),
        "rms": round(rms, 2)
    }

def reduce_noise(signal: np.ndarray, threshold: float = 500) -> np.ndarray:
    """
    Applies noise gating by zeroing amplitudes below the specified threshold.
    """
    cleaned = np.copy(signal)
    cleaned[np.abs(cleaned) < threshold] = 0
    return cleaned

def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Applies a moving average smoothing filter to the signal.
    """
    if window_size < 2:
        return np.copy(signal)
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal.astype(np.float64), kernel, mode='same')
    return np.clip(smoothed, -32768, 32767).astype(np.int16)

def frequency_analysis(signal: np.ndarray, sample_rate: int, num_bins: int = 150):
    """
    Calculates FFT frequency spectrum and downsamples magnitude for rendering.
    """
    if len(signal) == 0:
        return [], []

    # Compute FFT
    n = len(signal)
    fft_result = np.fft.fft(signal.astype(np.float64))
    frequencies = np.fft.fftfreq(n, d=1.0 / sample_rate)
    magnitude = np.abs(fft_result)

    # Take positive frequencies only
    half_n = n // 2
    pos_freqs = frequencies[:half_n]
    pos_mags = magnitude[:half_n]

    # Downsample for visualization
    if len(pos_freqs) > num_bins:
        indices = np.linspace(0, len(pos_freqs) - 1, num_bins, dtype=int)
        freq_out = pos_freqs[indices].round(2).tolist()
        mag_out = pos_mags[indices].round(2).tolist()
    else:
        freq_out = pos_freqs.round(2).tolist()
        mag_out = pos_mags.round(2).tolist()

    return freq_out, mag_out

def compare_signals(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE) between original and processed signals.
    """
    if len(original) != len(processed):
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

    mse = float(np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2))
    return round(mse, 2)

def downsample_waveform(signal: np.ndarray, max_points: int = 600):
    """
    Downsamples signal for fast frontend canvas rendering.
    """
    n = len(signal)
    if n <= max_points:
        return signal.tolist()
    
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return signal[indices].tolist()

def generate_demo_wav(duration_sec: float = 3.0, sample_rate: int = 22050):
    """
    Generates a synthetic audio signal: fundamental 440Hz A4 note + harmonics + noise.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # Fundamental (440 Hz) + 2nd Harmonic (880 Hz)
    clean_signal = 12000 * np.sin(2 * np.pi * 440 * t) + 4000 * np.sin(2 * np.pi * 880 * t)
    
    # Add random background noise
    noise = np.random.normal(0, 800, len(t))
    
    # Combine signal and noise
    noisy_signal = (clean_signal + noise).astype(np.int16)
    
    return noisy_signal, sample_rate
