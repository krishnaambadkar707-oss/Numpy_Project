"""
test_translate.py - Comprehensive Unit and Integration Tests for Audio Translator.

Tests:
1. Unit tests for backend/translator.py:
   - convert_to_wav_16k_mono: ffmpeg invocation, error handling, missing ffmpeg, unsupported formats.
   - transcribe: faster-whisper invocation, result mapping, no speech error.
   - translate_text: same-language skip, chunking long text, retry backoff, error handling.
   - synthesize: voice mapping, unsupported language, edge-tts streaming.
   - list_supported_languages: metadata structure check.
2. Integration / API tests for backend/translate_routes.py using FastAPI TestClient:
   - GET /api/translate/languages
   - POST /api/translate/start validation (size limits, invalid format, invalid lang)
   - Happy-path job lifecycle (queued -> done)
   - Same-language skip job execution
   - Error handling propagation (e.g. no speech)
   - Static mount non-shadowing verification via server.py
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from backend.translator import (
    convert_to_wav_16k_mono,
    transcribe,
    translate_text,
    synthesize,
    list_supported_languages,
    SUPPORTED_LANGUAGES,
    TranslationProvider,
    _chunk_text
)
from backend.main import app
from backend.translate_routes import (
    TRANSLATE_JOBS,
    run_translation_pipeline,
    cleanup_old_jobs
)
from server import app as server_app


# ---------------------------------------------------------------------------
# Unit Tests for Pure Logic in translator.py
# ---------------------------------------------------------------------------
class TestTranslatorLogic(unittest.TestCase):
    """Unit tests for backend/translator.py pure logic."""

    def test_list_supported_languages(self):
        langs = list_supported_languages()
        self.assertGreaterEqual(len(langs), 20)
        codes = [l["code"] for l in langs]
        # Must contain at least the required languages
        required = ["en", "hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "ur", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar"]
        for r in required:
            self.assertIn(r, codes, f"Language {r} must be in supported languages.")
            self.assertIn("name", SUPPORTED_LANGUAGES[r])
            self.assertIn("voice", SUPPORTED_LANGUAGES[r])

    def test_convert_to_wav_missing_ffmpeg(self):
        with patch("backend.translator.get_ffmpeg_cmd", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                convert_to_wav_16k_mono(b"fake audio data", "test.mp3")
            self.assertIn("ffmpeg is not installed", str(ctx.exception).lower())

    def test_convert_to_wav_pure_python_fallback_for_wav(self):
        # When ffmpeg is missing, WAV uploads must succeed via pure Python / NumPy fallback
        from backend.audio_processor import generate_demo_wav, export_wav_bytes
        sig, sr = generate_demo_wav(duration_sec=1.0, sample_rate=22050)
        wav_bytes = export_wav_bytes(sig, sr)

        with patch("backend.translator.get_ffmpeg_cmd", return_value=None):
            wav_path = convert_to_wav_16k_mono(wav_bytes, "test.wav")
            self.assertTrue(os.path.exists(wav_path))
            # Verify converted file is 16 kHz mono
            import wave
            with wave.open(wav_path, "rb") as wf:
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)

    def test_convert_to_wav_unsupported_format(self):
        with patch("backend.translator.get_ffmpeg_cmd", return_value="/usr/bin/ffmpeg"):
            with self.assertRaises(ValueError) as ctx:
                convert_to_wav_16k_mono(b"fake audio data", "test.txt")
            self.assertIn("Unsupported audio format", str(ctx.exception))

    def test_convert_to_wav_empty_payload(self):
        with patch("backend.translator.get_ffmpeg_cmd", return_value="/usr/bin/ffmpeg"):
            with self.assertRaises(ValueError) as ctx:
                convert_to_wav_16k_mono(b"", "test.mp3")
            self.assertIn("empty", str(ctx.exception))

    def test_convert_to_wav_success(self):
        with patch("backend.translator.get_ffmpeg_cmd", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            # Mock os.path.exists and os.path.getsize for the output file
            with patch("os.path.exists", return_value=True), \
                 patch("os.path.getsize", return_value=1024):
                wav_path = convert_to_wav_16k_mono(b"dummy audio content", "speech.mp3")
                self.assertTrue(wav_path.endswith(".wav"))
                self.assertTrue(mock_run.called)
                args = mock_run.call_args[0][0]
                self.assertIn("/usr/bin/ffmpeg", args[0])
                self.assertIn("-ar", args)
                self.assertIn("16000", args)
                self.assertIn("-ac", args)
                self.assertIn("1", args)

    def test_convert_to_wav_corrupt_file(self):
        with patch("backend.translator.get_ffmpeg_cmd", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="[mp3 @ 0x123] Header missing: Invalid data found when processing input"
            )
            with self.assertRaises(ValueError) as ctx:
                convert_to_wav_16k_mono(b"bad binary data", "corrupt.mp3")
            self.assertIn("Failed to decode audio file", str(ctx.exception))

    def test_transcribe_success(self):
        mock_segment1 = MagicMock(text="नमस्ते")
        mock_segment2 = MagicMock(text="आप कैसे हैं?")
        mock_info = MagicMock(language="hi", language_probability=0.98)

        mock_whisper_instance = MagicMock()
        mock_whisper_instance.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        with patch("backend.translator.get_whisper_model", return_value=mock_whisper_instance):
            text, lang, prob = transcribe("fake.wav")
            self.assertEqual(text, "नमस्ते आप कैसे हैं?")
            self.assertEqual(lang, "hi")
            self.assertAlmostEqual(prob, 0.98, places=2)

    def test_transcribe_no_speech_detected(self):
        mock_info = MagicMock(language="en", language_probability=0.5)
        mock_whisper_instance = MagicMock()
        mock_whisper_instance.transcribe.return_value = ([], mock_info)

        with patch("backend.translator.get_whisper_model", return_value=mock_whisper_instance):
            with self.assertRaises(ValueError) as ctx:
                transcribe("silent.wav")
            self.assertIn("No speech could be detected", str(ctx.exception))

    def test_translate_text_same_language_skip(self):
        # When source == target, should return text untouched without calling translation provider
        mock_provider = MagicMock()
        result = translate_text("Hello world", "en", "en", provider=mock_provider)
        self.assertEqual(result, "Hello world")
        self.assertFalse(mock_provider.translate.called)

        result_case_insensitive = translate_text("Namaste", "HI", "hi", provider=mock_provider)
        self.assertEqual(result_case_insensitive, "Namaste")
        self.assertFalse(mock_provider.translate.called)

    def test_translate_text_different_languages(self):
        class MockProvider(TranslationProvider):
            def translate(self, text: str, source_lang: str, target_lang: str) -> str:
                return f"Translated({text}) to {target_lang}"

        provider = MockProvider()
        res = translate_text("Hello", "en", "es", provider=provider)
        self.assertEqual(res, "Translated(Hello) to es")

    def test_translate_text_chunking(self):
        # Verify text chunking splits at sentences
        long_text = ". ".join([f"Sentence {i} with some content" for i in range(200)]) + "."
        chunks = _chunk_text(long_text, max_chars=500)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 500)

        # Translate all chunks with chunk size 500
        mock_provider = MagicMock()
        mock_provider.translate.side_effect = lambda t, s, tgt: f"T({t})"
        translated = translate_text(long_text, "en", "fr", provider=mock_provider, max_chars=500)
        self.assertTrue(translated.startswith("T("))
        self.assertEqual(mock_provider.translate.call_count, len(chunks))

    def test_translate_text_retry_and_failure(self):
        mock_provider = MagicMock()
        mock_provider.translate.side_effect = ConnectionError("Service unreachable")

        with patch("time.sleep", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                translate_text("Test error", "en", "de", provider=mock_provider)
            self.assertIn("Translation service unreachable", str(ctx.exception))
            self.assertEqual(mock_provider.translate.call_count, 3)

    def test_synthesize_unsupported_language(self):
        with self.assertRaises(ValueError) as ctx:
            synthesize("Test", target_lang="klingon")
        self.assertIn("Target language 'klingon' is not supported", str(ctx.exception))

    def test_synthesize_empty_text(self):
        with self.assertRaises(ValueError) as ctx:
            synthesize("   ", target_lang="en")
        self.assertIn("Cannot synthesize empty text", str(ctx.exception))

    def test_synthesize_success(self):
        mock_audio_bytes = b"ID3\x03\x00\x00\x00mocked_mp3_stream"
        with patch("backend.translator._edge_tts_synthesize", return_value=mock_audio_bytes):
            res = synthesize("Hello world", target_lang="en")
            self.assertEqual(res, mock_audio_bytes)

    def test_pyav_metadata_errors_patch_compatibility(self):
        """Verifies that av.open handles metadata_errors='ignore' without TypeError."""
        try:
            import av
            from backend.translator import _patch_pyav_metadata_errors
            _patch_pyav_metadata_errors()
            # Calling av.open with metadata_errors must not raise TypeError
            try:
                av.open("nonexistent_test_audio.wav", mode="r", metadata_errors="ignore")
            except (FileNotFoundError, av.error.FileNotFoundError):
                pass  # Expected: file does not exist, but TypeError was prevented!
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# API Route Integration Tests
# ---------------------------------------------------------------------------
class TestTranslateRoutes(unittest.TestCase):
    """Integration tests for FastAPI translate routes."""

    def setUp(self):
        self.client = TestClient(app)
        self.server_client = TestClient(server_app)
        TRANSLATE_JOBS.clear()

    def test_get_languages(self):
        res = self.client.get("/api/translate/languages")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("languages", data)
        langs = data["languages"]
        self.assertGreaterEqual(len(langs), 20)
        sample = langs[0]
        self.assertIn("code", sample)
        self.assertIn("name", sample)
        self.assertIn("voice", sample)

    def test_start_validation_unsupported_language(self):
        fake_wav = b"RIFF" + b"\x00" * 40
        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "invalid_lang_xyz"},
            files={"file": ("audio.wav", fake_wav, "audio/wav")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported target language", res.json()["detail"])

    def test_start_validation_unsupported_format(self):
        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "en"},
            files={"file": ("sample.txt", b"plain text", "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file format", res.json()["detail"])

    def test_start_validation_empty_file(self):
        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "en"},
            files={"file": ("audio.wav", b"", "audio/wav")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Empty audio file payload", res.json()["detail"])

    def test_start_validation_file_too_large(self):
        huge_bytes = b"0" * (26 * 1024 * 1024)
        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "en"},
            files={"file": ("huge.wav", huge_bytes, "audio/wav")}
        )
        self.assertEqual(res.status_code, 413)
        self.assertIn("exceeds maximum 25MB limit", res.json()["detail"])

    @patch("backend.translate_routes.convert_to_wav_16k_mono")
    @patch("backend.translate_routes.get_audio_duration_seconds", return_value=12.5)
    @patch("backend.translate_routes.transcribe", return_value=("Hello world", "en", 0.99))
    @patch("backend.translate_routes.translate_text", return_value="नमस्ते दुनिया")
    @patch("backend.translate_routes.synthesize", return_value=b"mocked_mp3_data_hi")
    def test_happy_path_pipeline_lifecycle(self, mock_synth, mock_trans_txt, mock_whisper, mock_dur, mock_conv):
        # Create a real temporary WAV file for convert_to_wav_16k_mono return
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.write(b"RIFF" + b"\x00" * 100)
        temp_wav.close()
        mock_conv.return_value = temp_wav.name

        fake_audio = b"fake audio content"
        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "hi"},
            files={"file": ("speech.mp3", fake_audio, "audio/mpeg")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("job_id", data)
        job_id = data["job_id"]

        # Run pipeline directly or let background task finish
        # In TestClient, BackgroundTasks run during request completion
        status_res = self.client.get(f"/api/translate/status/{job_id}")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertEqual(status_data["status"], "done")
        self.assertEqual(status_data["progress"], 100)
        self.assertEqual(status_data["detected_language"], "en")
        self.assertEqual(status_data["transcript"], "Hello world")
        self.assertEqual(status_data["translated_text"], "नमस्ते दुनिया")

        # Fetch result audio
        result_res = self.client.get(f"/api/translate/result/{job_id}")
        self.assertEqual(result_res.status_code, 200)
        self.assertEqual(result_res.content, b"mocked_mp3_data_hi")
        self.assertEqual(result_res.headers["content-type"], "audio/mpeg")

    @patch("backend.translate_routes.convert_to_wav_16k_mono")
    @patch("backend.translate_routes.get_audio_duration_seconds", return_value=3.0)
    @patch("backend.translate_routes.transcribe", return_value=("Hello again", "en", 0.95))
    @patch("backend.translate_routes.synthesize", return_value=b"mocked_mp3_en")
    def test_same_language_skip_flow(self, mock_synth, mock_whisper, mock_dur, mock_conv):
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.write(b"RIFF" + b"\x00" * 100)
        temp_wav.close()
        mock_conv.return_value = temp_wav.name

        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "en"},
            files={"file": ("english.wav", b"dummy audio", "audio/wav")}
        )
        job_id = res.json()["job_id"]
        status_res = self.client.get(f"/api/translate/status/{job_id}")
        status_data = status_res.json()
        self.assertEqual(status_data["status"], "done")
        self.assertEqual(status_data["detected_language"], "en")
        # Same language: translated_text == transcript
        self.assertEqual(status_data["transcript"], "Hello again")
        self.assertEqual(status_data["translated_text"], "Hello again")

    @patch("backend.translate_routes.convert_to_wav_16k_mono")
    @patch("backend.translate_routes.get_audio_duration_seconds", return_value=350.0)
    def test_audio_duration_exceeded(self, mock_dur, mock_conv):
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.write(b"RIFF" + b"\x00" * 100)
        temp_wav.close()
        mock_conv.return_value = temp_wav.name

        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "es"},
            files={"file": ("long.wav", b"dummy audio", "audio/wav")}
        )
        job_id = res.json()["job_id"]
        status_res = self.client.get(f"/api/translate/status/{job_id}")
        status_data = status_res.json()
        self.assertEqual(status_data["status"], "error")
        self.assertIn("exceeds the maximum limit", status_data["error"])

    @patch("backend.translate_routes.convert_to_wav_16k_mono")
    @patch("backend.translate_routes.get_audio_duration_seconds", return_value=5.0)
    @patch("backend.translate_routes.transcribe", side_effect=ValueError("No speech could be detected in the provided audio file."))
    def test_pipeline_no_speech_error(self, mock_whisper, mock_dur, mock_conv):
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.write(b"RIFF" + b"\x00" * 100)
        temp_wav.close()
        mock_conv.return_value = temp_wav.name

        res = self.client.post(
            "/api/translate/start",
            data={"target_lang": "hi"},
            files={"file": ("silent.wav", b"dummy audio", "audio/wav")}
        )
        job_id = res.json()["job_id"]
        status_res = self.client.get(f"/api/translate/status/{job_id}")
        status_data = status_res.json()
        self.assertEqual(status_data["status"], "error")
        self.assertIn("No speech could be detected", status_data["error"])

        # Result request on failed job returns 400
        result_res = self.client.get(f"/api/translate/result/{job_id}")
        self.assertEqual(result_res.status_code, 400)
        self.assertIn("Job failed with error", result_res.json()["detail"])

    def test_server_mount_does_not_shadow_api(self):
        # Crucial check: verify server_app (which mounts static files on '/') does not shadow /api/translate
        res = self.server_client.get("/api/translate/languages")
        self.assertEqual(res.status_code, 200)
        self.assertIn("languages", res.json())

        # Verify static frontend files for translator are served cleanly
        res_css = self.server_client.get("/translate.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("#translator-section", res_css.text)

        res_js = self.server_client.get("/translate.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("Audio Translator Client Module", res_js.text)


if __name__ == "__main__":
    print("=========================================================")
    print("Running Audio Translator Test Suite...")
    print("=========================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTranslatorLogic)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTranslateRoutes))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll translator tests passed cleanly!")
        sys.exit(0)
    else:
        print("\nTranslator test suite failed!")
        sys.exit(1)
