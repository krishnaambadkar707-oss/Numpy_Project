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
     * Loads available languages from GET /api/translate/languages.
     */
    async function loadLanguages() {
        const baseUrl = getApiBaseUrl();
        const url = `${baseUrl}/api/translate/languages`;

        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 3000);
            const res = await fetch(url, { signal: controller.signal });
            clearTimeout(timeout);

            if (!res.ok) throw new Error('Failed to fetch languages.');

            const data = await res.json();
            supportedLanguages = data.languages || [];

            // Populate select options
            langSelect.innerHTML = '';
            supportedLanguages.forEach(lang => {
                const opt = document.createElement('option');
                opt.value = lang.code;
                opt.textContent = `${lang.name} (${lang.code})`;
                if (lang.code === 'en') opt.selected = true;
                langSelect.appendChild(opt);
            });

            if (offlineBox) offlineBox.style.display = 'none';
        } catch (err) {
            console.warn('Translator: Backend not reachable for languages:', err.message);
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
        const url = `${baseUrl}/api/translate/start`;

        const formData = new FormData();
        formData.append('file', selectedFile, selectedFile.name);
        formData.append('target_lang', targetLang);

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned error status ${response.status}`);
            }

            const data = await response.json();
            activeJobId = data.job_id;
            startPolling(activeJobId);
        } catch (err) {
            showError(`Failed to initiate translation: ${err.message}`);
            setProcessingState(false);
            progressBox.style.display = 'none';
        }
    }

    /**
     * Polls GET /api/translate/status/{job_id} with low latency (600ms).
     */
    function startPolling(jobId) {
        if (pollInterval) clearInterval(pollInterval);

        const checkStatus = async () => {
            const baseUrl = getApiBaseUrl();
            const url = `${baseUrl}/api/translate/status/${jobId}`;

            try {
                const res = await fetch(url);
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
                    onJobComplete(job);
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
    function onJobComplete(job) {
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
        const audioUrl = `${baseUrl}/api/translate/result/${job.job_id}?t=${Date.now()}`;
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

