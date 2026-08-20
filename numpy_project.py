# =========================================================
# AUDIO SIGNAL PROCESSOR
# Using NumPy + wave module
# Features:
# 1. Load WAV audio
# 2. Waveform analysis
# 3. Noise reduction
# 4. Signal smoothing
# 5. Frequency analysis (FFT)
# 6. Save processed audio
# =========================================================

import os
import wave
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: LOAD AUDIO FILE

def load_wav(file_name):
    wav = wave.open(file_name, 'rb')

    sample_rate = wav.getframerate()
    n_frames = wav.getnframes()
    n_channels = wav.getnchannels()

    audio_data = wav.readframes(n_frames)

    wav.close()

    # Convert bytes to NumPy array
    signal = np.frombuffer(audio_data, dtype=np.int16)

    # Stereo → Mono
    if n_channels == 2:
        signal = signal[::2]

    return signal, sample_rate

# STEP 2: SAVE AUDIO FILE

def save_wav(file_name, signal, sample_rate):
    signal = signal.astype(np.int16)

    wav = wave.open(file_name, 'wb')

    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)

    wav.writeframes(signal.tobytes())

    wav.close()

# STEP 3: WAVEFORM ANALYSIS

def analyze_signal(signal):
    print("\n===== SIGNAL ANALYSIS =====")

    amplitude = np.max(np.abs(signal))
    mean_value = np.mean(signal)
    variance = np.var(signal)

    print("Maximum Amplitude :", amplitude)
    print("Mean Value         :", mean_value)
    print("Variance           :", variance)

# STEP 4: NOISE REDUCTION
# Remove very small amplitudes

def reduce_noise(signal, threshold=500):

    cleaned_signal = np.copy(signal)

    cleaned_signal[np.abs(cleaned_signal) < threshold] = 0

    return cleaned_signal

# STEP 5: SIGNAL SMOOTHING
# Moving Average Filter

def smooth_signal(signal, window_size=5):

    kernel = np.ones(window_size) / window_size

    smoothed_signal = np.convolve(signal, kernel, mode='same')

    return smoothed_signal

# STEP 6: FREQUENCY ANALYSIS
# FFT

def frequency_analysis(signal, sample_rate):

    fft_result = np.fft.fft(signal)

    frequencies = np.fft.fftfreq(len(signal), d=1/sample_rate)

    magnitude = np.abs(fft_result)

    return frequencies, magnitude

# STEP 7: PLOT WAVEFORM

def plot_waveform(original, processed):

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(original)
    plt.title("Original Signal")

    plt.subplot(2, 1, 2)
    plt.plot(processed)
    plt.title("Processed Signal")

    plt.tight_layout()
    plt.show()

# STEP 8: PLOT FREQUENCY SPECTRUM

def plot_frequency(frequencies, magnitude):

    plt.figure(figsize=(12, 5))

    plt.plot(frequencies[:len(frequencies)//2],
             magnitude[:len(magnitude)//2])

    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.show()

# STEP 9: COMPARE SIGNALS
# MSE Calculation

def compare_signals(original, processed):

    mse = np.mean((original - processed) ** 2)

    print("\n===== SIGNAL COMPARISON =====")
    print("Mean Squared Error (MSE):", mse)

def create_sample_wav(file_name, duration=3, sample_rate=22050):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440 Hz sine wave + 880 Hz harmonic + noise
    clean = 12000 * np.sin(2 * np.pi * 440 * t) + 4000 * np.sin(2 * np.pi * 880 * t)
    noise = np.random.normal(0, 800, len(t))
    signal = (clean + noise).astype(np.int16)
    save_wav(file_name, signal, sample_rate)

# MAIN PROGRAM

def main():

    input_file = "input.wav"
    output_file = "processed.wav"

    if not os.path.exists(input_file):
        print(f"File '{input_file}' not found. Generating sample audio file '{input_file}'...")
        create_sample_wav(input_file)

    print("Loading audio...")


    signal, sample_rate = load_wav(input_file)

    print("Sample Rate:", sample_rate)
    print("Total Samples:", len(signal))

    # Analysis

    analyze_signal(signal)

    # Noise Reduction

    cleaned_signal = reduce_noise(signal, threshold=500)

    # Smoothing

    smoothed_signal = smooth_signal(cleaned_signal, window_size=7)

    # Frequency Analysis

    frequencies, magnitude = frequency_analysis(
        smoothed_signal,
        sample_rate
    )

    # Comparison

    compare_signals(signal, smoothed_signal)

    # Save Processed Audio

    save_wav(output_file, smoothed_signal, sample_rate)

    print("\nProcessed audio saved as:", output_file)

    # Visualization

    plot_waveform(signal, smoothed_signal)

    plot_frequency(frequencies, magnitude)

# Run Program

if __name__ == "__main__":
    main()