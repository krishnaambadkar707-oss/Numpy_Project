import sys
import unittest
import numpy as np
from fastapi.testclient import TestClient

# Import backend modules
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
from backend.main import app, AUDIO_STORE
from server import app as server_app

class TestAudioProcessorDSP(unittest.TestCase):
    """Unit tests for NumPy audio processing functions in backend/audio_processor.py"""

    def setUp(self):
        self.signal, self.sample_rate = generate_demo_wav(duration_sec=2.0, sample_rate=22050)

    def test_demo_signal_generation(self):
        self.assertEqual(len(self.signal), 44100)
        self.assertEqual(self.sample_rate, 22050)
        self.assertIsInstance(self.signal, np.ndarray)

    def test_wav_export_and_load_cycle(self):
        wav_bytes = export_wav_bytes(self.signal, self.sample_rate)
        self.assertGreater(len(wav_bytes), 44)
        
        loaded_signal, sr, n_channels, n_frames = load_wav_from_bytes(wav_bytes)
        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(n_channels, 1)
        self.assertEqual(len(loaded_signal), len(self.signal))

    def test_analyze_signal_metrics(self):
        metrics = analyze_signal(self.signal)
        self.assertIn("max_amplitude", metrics)
        self.assertIn("mean_value", metrics)
        self.assertIn("variance", metrics)
        self.assertIn("rms", metrics)
        self.assertGreater(metrics["max_amplitude"], 0)

    def test_reduce_noise_gating(self):
        noisy_signal = np.array([100, 300, 700, -200, -800, 1500], dtype=np.int16)
        cleaned = reduce_noise(noisy_signal, threshold=500)
        expected = np.array([0, 0, 700, 0, -800, 1500], dtype=np.int16)
        np.testing.assert_array_equal(cleaned, expected)

    def test_smooth_signal_convolution(self):
        raw = np.array([0, 1000, 0, 1000, 0], dtype=np.int16)
        smoothed = smooth_signal(raw, window_size=3)
        self.assertEqual(len(smoothed), len(raw))
        # Peak amplitude after smoothing should be lower than original raw peak
        self.assertLess(np.max(smoothed), np.max(raw))

    def test_frequency_analysis_fft(self):
        freqs, mags = frequency_analysis(self.signal, self.sample_rate, num_bins=50)
        self.assertEqual(len(freqs), 50)
        self.assertEqual(len(mags), 50)
        self.assertGreaterEqual(freqs[0], 0)

    def test_compare_signals_mse(self):
        sig1 = np.array([100, 200, 300], dtype=np.int16)
        sig2 = np.array([110, 190, 310], dtype=np.int16)
        mse = compare_signals(sig1, sig2)
        # Expected MSE = ((10^2 + (-10)^2 + (-10)^2) / 3) = 100.0
        self.assertEqual(mse, 100.0)

    def test_downsample_waveform(self):
        downsampled = downsample_waveform(self.signal, max_points=100)
        self.assertEqual(len(downsampled), 100)


class TestFastAPIEndpoints(unittest.TestCase):
    """Integration tests for REST API endpoints in backend/main.py & server.py"""

    def setUp(self):
        self.client = TestClient(app)
        self.server_client = TestClient(server_app)

    def test_api_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_api_demo_audio_generation(self):
        res = self.client.get("/api/demo")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("audio_id", data)
        self.assertIn("original_waveform", data)
        self.assertIn("processed_waveform", data)
        self.assertIn("frequencies", data)
        self.assertIn("magnitudes", data)

    def test_api_upload_and_process_flow(self):
        # 1. Generate WAV bytes
        sig, sr = generate_demo_wav(duration_sec=1.0, sample_rate=22050)
        wav_bytes = export_wav_bytes(sig, sr)

        # 2. Upload binary payload
        upload_res = self.client.post(
            "/api/upload?filename=test_input.wav",
            content=wav_bytes,
            headers={"Content-Type": "application/octet-stream"}
        )
        self.assertEqual(upload_res.status_code, 200)
        upload_data = upload_res.json()
        audio_id = upload_data["audio_id"]
        self.assertEqual(upload_data["filename"], "test_input.wav")

        # 3. Process DSP parameters
        proc_res = self.client.post(
            "/api/process",
            json={"audio_id": audio_id, "threshold": 600, "window_size": 7}
        )
        self.assertEqual(proc_res.status_code, 200)
        proc_data = proc_res.json()
        self.assertEqual(proc_data["threshold"], 600)
        self.assertEqual(proc_data["window_size"], 7)

        # 4. Stream original & processed audio WAV
        orig_audio_res = self.client.get(f"/api/audio/{audio_id}/original")
        self.assertEqual(orig_audio_res.status_code, 200)
        self.assertEqual(orig_audio_res.headers["content-type"], "audio/wav")

        proc_audio_res = self.client.get(f"/api/audio/{audio_id}/processed")
        self.assertEqual(proc_audio_res.status_code, 200)
        self.assertEqual(proc_audio_res.headers["content-type"], "audio/wav")

    def test_api_upload_file_size_limit(self):
        # 26 MB payload (exceeding 25 MB max limit)
        oversized = b"0" * (26 * 1024 * 1024)
        res = self.client.post(
            "/api/upload?filename=huge.wav",
            content=oversized,
            headers={"Content-Type": "application/octet-stream"}
        )
        self.assertEqual(res.status_code, 413)

    def test_static_frontend_serving(self):
        # Verify index.html, style.css, app.js
        res_html = self.server_client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("<title>Audio Signal Processor - NumPy Powered Studio</title>", res_html.text)

        res_css = self.server_client.get("/style.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("--bg-dark", res_css.text)

        res_js = self.server_client.get("/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("drawWaveform", res_js.text)


if __name__ == "__main__":
    print("=========================================================")
    print("Running Audio Signal Processor Test Suite...")
    print("=========================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioProcessorDSP)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFastAPIEndpoints))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll project tests passed cleanly!")
        sys.exit(0)
    else:
        print("\nTest suite failed!")
        sys.exit(1)
