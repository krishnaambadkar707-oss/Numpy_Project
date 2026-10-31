/**
 * Audio Translator Client Module (translate.js)
 * Namespaced inside an IIFE to prevent global scope pollution.
 */
(function () {
    'use strict';

    // State Variables
    let selectedFile = null;
    let activeJobId = null;
    let pollInterval = null;
    let supportedLanguages = [];
    let isTranslating = false;

    // Elements
    let section, dropzone, fileInput, btnBrowse, fileNameDisplay;
    let langSelect, btnTranslate, btnReset;
    let progressBox, progressBar, progressStatusText, progressPercentText;
    let stepsList, resultContainer, errorBox, offlineBox;
    let transcriptBox, translatedTextBox, btnCopyOrig, btnCopyTrans;
    let audioPlayer, btnDownloadAudio, srcLangBadge, tgtLangBadge;

    const STEP_INDICES = {
        'queued': 0,
        'converting': 1,
        'transcribing': 2,
        'translating': 3,
        'synthesizing': 4,
        'done': 5
    };

    const STEP_LABELS = {
        'queued': 'Job Queued in Worker Queue...',
        'converting': 'Converting Audio to 16 kHz Mono (FFmpeg)...',
        'transcribing': 'Transcribing Speech & Detecting Language (Whisper)...',
        'translating': 'Translating Transcript to Target Language...',
        'synthesizing': 'Synthesizing Neural Audio (Edge-TTS)...',
        'done': 'Translation Complete!',
        'error': 'Translation Failed'
    };

    /**
     * Resolve API base URL: check window.apiBaseUrl if set by app.js,
     * or fallback to relative '/' if served via http, or http://127.0.0.1:8000.
     */
    function getApiBaseUrl() {
        if (typeof window.apiBaseUrl !== 'undefined' && window.apiBaseUrl !== null) {
            return window.apiBaseUrl;
        }
        if (window.location.protocol.startsWith('http')) {
            return '';
        }
        return 'http://127.0.0.1:8000';
    }

    const DEFAULT_LANGUAGES = [
        { code: 'en', name: 'English' },
        { code: 'hi', name: 'Hindi' },
        { code: 'mr', name: 'Marathi' },
        { code: 'ta', name: 'Tamil' },
        { code: 'te', name: 'Telugu' },
        { code: 'bn', name: 'Bengali' },
        { code: 'gu', name: 'Gujarati' },
        { code: 'kn', name: 'Kannada' },
        { code: 'ml', name: 'Malayalam' },
        { code: 'pa', name: 'Punjabi' },
        { code: 'ur', name: 'Urdu' },
        { code: 'es', name: 'Spanish' },
        { code: 'fr', name: 'French' },
        { code: 'de', name: 'German' },
        { code: 'it', name: 'Italian' },
        { code: 'pt', name: 'Portuguese' },
        { code: 'ru', name: 'Russian' },
        { code: 'ja', name: 'Japanese' },
        { code: 'ko', name: 'Korean' },
        { code: 'zh', name: 'Chinese' },
        { code: 'ar', name: 'Arabic' }
    ];

    function populateLanguages(langs) {
        if (!langSelect) return;
        const currentVal = langSelect.value || 'en';
        langSelect.innerHTML = '';
        langs.forEach(lang => {
            const opt = document.createElement('option');
            opt.value = lang.code;
            opt.textContent = `${lang.name} (${lang.code})`;
            if (lang.code === currentVal) opt.selected = true;
            langSelect.appendChild(opt);
        });
    }

    /**
     * Initializes DOM references and event listeners once DOM is ready.
     */
    function init() {
        section = document.getElementById('translator-section');
        if (!section) return;

        dropzone = document.getElementById('trans-dropzone');
        fileInput = document.getElementById('trans-file-input');
        btnBrowse = document.getElementById('trans-btn-browse');
        fileNameDisplay = document.getElementById('trans-selected-file');
        langSelect = document.getElementById('trans-lang-select');
        btnTranslate = document.getElementById('trans-btn-submit');
        btnReset = document.getElementById('trans-btn-reset');

        progressBox = document.getElementById('trans-progress-box');
        progressBar = document.getElementById('trans-progress-bar');
        progressStatusText = document.getElementById('trans-status-text');
        progressPercentText = document.getElementById('trans-percent-text');
        stepsList = document.getElementById('trans-steps-list');

        resultContainer = document.getElementById('trans-result-container');
        errorBox = document.getElementById('trans-error-box');
        offlineBox = document.getElementById('trans-offline-box');

        transcriptBox = document.getElementById('trans-transcript-text');
        translatedTextBox = document.getElementById('trans-translated-text');
        btnCopyOrig = document.getElementById('trans-btn-copy-orig');
        btnCopyTrans = document.getElementById('trans-btn-copy-trans');

        audioPlayer = document.getElementById('trans-audio-player');
        btnDownloadAudio = document.getElementById('trans-btn-download');
        srcLangBadge = document.getElementById('trans-src-lang-badge');
        tgtLangBadge = document.getElementById('trans-tgt-lang-badge');

        attachEventListeners();
        populateLanguages(DEFAULT_LANGUAGES);
        loadLanguages();
    }

    function attachEventListeners() {
        // Browse button
        btnBrowse.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
        });

        // Dropzone click
        dropzone.addEventListener('click', (e) => {
            if (e.target !== btnBrowse) {
                fileInput.click();
            }
        });

        // Drag and drop
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFileSelected(e.dataTransfer.files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelected(e.target.files[0]);
            }
        });

        // Translate button
        btnTranslate.addEventListener('click', startTranslationJob);

        // Reset button
        btnReset.addEventListener('click', resetTranslator);

        // Copy buttons
        if (btnCopyOrig) {
            btnCopyOrig.addEventListener('click', () => copyToClipboard(transcriptBox.textContent, btnCopyOrig));
        }
        if (btnCopyTrans) {
            btnCopyTrans.addEventListener('click', () => copyToClipboard(translatedTextBox.textContent, btnCopyTrans));
        }
    }

    /**
     * Validates and sets the selected audio file.
     */
    function handleFileSelected(file) {
        hideError();
        const allowedExts = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'];
        const name = file.name.toLowerCase();
        const isValidExt = allowedExts.some(ext => name.endsWith(ext));

        if (!isValidExt) {
            showError(`Unsupported file format. Please upload an audio file (${allowedExts.join(', ')}).`);
            return;
        }

        const maxBytes = 25 * 1024 * 1024;
        if (file.size > maxBytes) {
            showError(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 25 MB limit.`);
            return;
        }

        selectedFile = file;
        fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        fileNameDisplay.style.display = 'inline-flex';
        btnTranslate.disabled = false;
    }

    /**
     * Loads available languages from GET /api/translate/languages (or /translate/languages).
     * Retries automatically if backend is warming up (e.g. serverless cold start).
     */
    async function loadLanguages(retryCount = 0) {
        const baseUrl = getApiBaseUrl();
        const urls = [
            `${baseUrl}/api/translate/languages`,
            `${baseUrl}/translate/languages`
        ];

        let loaded = false;
        for (const url of urls) {
            try {
                const controller = new AbortController();
                const timeout = setTimeout(() => controller.abort(), 6000);
                const res = await fetch(url, { signal: controller.signal });
                clearTimeout(timeout);

                if (res.ok) {
                    const data = await res.json();
                    if (data && Array.isArray(data.languages) && data.languages.length > 0) {
                        supportedLanguages = data.languages;
                        populateLanguages(supportedLanguages);
                        if (offlineBox) offlineBox.style.display = 'none';
                        if (selectedFile) btnTranslate.disabled = false;
                        loaded = true;
                        break;
                    }
                }
            } catch (err) {
                // Try next endpoint fallback
            }
        }

        if (!loaded) {
            if (retryCount < 3) {
                setTimeout(() => loadLanguages(retryCount + 1), 1500);
                return;
            }
            console.warn('Translator: Backend not reachable for languages after retries.');
            if (offlineBox) {
                offlineBox.style.display = 'flex';
                offlineBox.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <span><strong>Translator Backend Offline:</strong> The Audio Translator requires the Python backend server. Run <code>python server.py</code> to enable multi-language audio translation.</span>
                `;
            }
            btnTranslate.disabled = true;
        }
    }

    /**
     * Submits audio file and target language to POST /api/translate/start.
     */
    async function startTranslationJob() {
        if (!selectedFile) {
            showError('Please select or drop an audio file first.');
            return;
        }
        if (isTranslating) return; // Prevent double-click

        const targetLang = langSelect.value;
        if (!targetLang) {
            showError('Please choose a target language.');
            return;
        }

        hideError();
        setProcessingState(true);
        resultContainer.style.display = 'none';
        progressBox.style.display = 'block';
        updateProgressUI('queued', 5);

        const baseUrl = getApiBaseUrl();
        const urls = [
            `${baseUrl}/api/translate/start`,
            `${baseUrl}/translate/start`
        ];

        let success = false;
        let lastError = null;

        for (const url of urls) {
            const formData = new FormData();
            formData.append('file', selectedFile, selectedFile.name);
            formData.append('target_lang', targetLang);

            try {
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    activeJobId = data.job_id;
                    if (data.status === 'done') {
                        updateProgressUI('done', 100);
                        onJobComplete(data, url.includes('/api/'));
                        success = true;
                        break;
                    }
                    startPolling(activeJobId, url.includes('/api/'));
                    success = true;
                    break;
                } else if (response.status !== 404 && response.status !== 405) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData.detail || `Server returned error status ${response.status}`);
                }
            } catch (err) {
                lastError = err;
            }
        }

        if (!success) {
            showError(`Failed to initiate translation: ${lastError ? lastError.message : 'Backend unreachable'}`);
            setProcessingState(false);
            progressBox.style.display = 'none';
        }
    }

    /**
     * Polls GET /api/translate/status/{job_id} with low latency (600ms).
     */
    function startPolling(jobId, useApiPrefix = true) {
        if (pollInterval) clearInterval(pollInterval);

        const checkStatus = async () => {
            const baseUrl = getApiBaseUrl();
            const prefix = useApiPrefix ? '/api/translate' : '/translate';
            const url = `${baseUrl}${prefix}/status/${jobId}`;

            try {
                let res = await fetch(url);
                if (!res.ok && res.status === 404 && useApiPrefix) {
                    // Fallback to non-prefixed status endpoint
                    res = await fetch(`${baseUrl}/translate/status/${jobId}`);
                }

                if (!res.ok) {
                    throw new Error(`Failed to query job status (${res.status})`);
                }

                const job = await res.json();
                updateProgressUI(job.status, job.progress);

                if (job.status === 'done') {
                    if (pollInterval) {
                        clearInterval(pollInterval);
                        pollInterval = null;
                    }
                    onJobComplete(job, useApiPrefix);
                } else if (job.status === 'error') {
                    if (pollInterval) {
                        clearInterval(pollInterval);
                        pollInterval = null;
                    }
                    showError(job.error || 'Translation failed during execution.');
                    setProcessingState(false);
                }
            } catch (err) {
                if (pollInterval) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                }
                showError(`Network error while checking job status: ${err.message}`);
                setProcessingState(false);
            }
        };

        // Immediate first check, then poll every 600ms for fast responsive updates
        checkStatus();
        pollInterval = setInterval(checkStatus, 600);
    }

    /**
     * Updates progress bar and step indicators.
     */
    function updateProgressUI(status, progress) {
        const percent = Math.min(100, Math.max(0, progress || 0));
        progressBar.style.width = `${percent}%`;
        progressPercentText.textContent = `${percent}%`;
        progressStatusText.textContent = STEP_LABELS[status] || status;

        const currentStepIdx = STEP_INDICES[status] ?? 0;
        const stepItems = stepsList.querySelectorAll('.trans-step-item');

        stepItems.forEach((item, idx) => {
            item.classList.remove('active', 'completed');
            if (idx < currentStepIdx) {
                item.classList.add('completed');
            } else if (idx === currentStepIdx) {
                item.classList.add('active');
            }
        });
    }

    /**
     * Renders job results when status === 'done'.
     */
    function onJobComplete(job, useApiPrefix = true) {
        setProcessingState(false);

        // Language badges
        const srcCode = (job.detected_language || 'en').toUpperCase();
        const confPercent = job.detected_language_confidence
            ? ` (${Math.round(job.detected_language_confidence * 100)}% conf)`
            : '';
        srcLangBadge.textContent = `Source: ${srcCode}${confPercent}`;

        const tgtCode = (job.target_language || langSelect.value).toUpperCase();
        tgtLangBadge.textContent = `Target: ${tgtCode}`;

        // Text display
        transcriptBox.textContent = job.transcript || 'No transcript available.';
        translatedTextBox.textContent = job.translated_text || 'No translated text available.';

        // Audio playback
        const baseUrl = getApiBaseUrl();
        const prefix = useApiPrefix ? '/api/translate' : '/translate';
        const audioUrl = `${baseUrl}${prefix}/result/${job.job_id}?t=${Date.now()}`;
        audioPlayer.src = audioUrl;
        audioPlayer.load();

        // Download button
        btnDownloadAudio.href = audioUrl;
        btnDownloadAudio.download = `translated_${job.target_language}_${job.job_id.slice(0, 8)}.mp3`;

        resultContainer.style.display = 'block';
    }

    /**
     * Resets form to initial state.
     */
    function resetTranslator() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
        selectedFile = null;
        activeJobId = null;
        isTranslating = false;
        fileInput.value = '';
        fileNameDisplay.textContent = '';
        fileNameDisplay.style.display = 'none';

        hideError();
        progressBox.style.display = 'none';
        resultContainer.style.display = 'none';
        audioPlayer.pause();
        audioPlayer.removeAttribute('src');

        setProcessingState(false);
        updateProgressUI('queued', 0);
    }

    function setProcessingState(processing) {
        isTranslating = processing;
        btnTranslate.disabled = processing || !selectedFile;
        btnBrowse.disabled = processing;
        fileInput.disabled = processing;
        langSelect.disabled = processing;
        btnReset.disabled = processing;
        dropzone.style.pointerEvents = processing ? 'none' : 'auto';
        dropzone.style.opacity = processing ? '0.6' : '1';
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.style.display = 'block';
    }

    function hideError() {
        errorBox.textContent = '';
        errorBox.style.display = 'none';
    }

    function copyToClipboard(text, buttonEl) {
        if (!navigator.clipboard) {
            alert('Clipboard access not available in this browser context.');
            return;
        }
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = buttonEl.innerHTML;
            buttonEl.innerHTML = `✓ Copied!`;
            setTimeout(() => {
                buttonEl.innerHTML = originalHTML;
            }, 1800);
        }).catch(err => {
            console.error('Copy failed:', err);
        });
    }

    // Auto-init on DOMContentLoaded or immediate if already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

