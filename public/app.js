// State variables
let currentAudioId = null;
let originalSignal = null; // Stored Int16Array for client-side mode
let originalWaveform = [];
let processedWaveform = [];
let frequencies = [];
let magnitudes = [];
let sampleRate = 22050;
let durationSec = 3.0;

// Backend Connection Mode
let apiBaseUrl = null; // null if standalone client mode, or "http://127.0.0.1:8000" / "" if backend active

// DOM Elements
const banner = document.getElementById('server-status-banner');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('audio-file-input');
const btnBrowse = document.getElementById('btn-browse');
const btnLoadDemo = document.getElementById('btn-load-demo');
const fileInfoCard = document.getElementById('file-info-card');

const sliderThreshold = document.getElementById('slider-threshold');
const sliderWindow = document.getElementById('slider-window');
const valThreshold = document.getElementById('val-threshold');
const valWindow = document.getElementById('val-window');
const btnProcess = document.getElementById('btn-process');

const audioPlayerOrig = document.getElementById('audio-player-original');
const audioPlayerProc = document.getElementById('audio-player-processed');

const metricMaxAmp = document.getElementById('metric-max-amp');
const metricMaxAmpOrig = document.getElementById('metric-max-amp-orig');
const metricMean = document.getElementById('metric-mean');
const metricMeanOrig = document.getElementById('metric-mean-orig');
const metricVariance = document.getElementById('metric-variance');
const metricVarOrig = document.getElementById('metric-var-orig');
const metricMse = document.getElementById('metric-mse');

const waveformCanvas = document.getElementById('waveform-canvas');
const fftCanvas = document.getElementById('fft-canvas');

// Hover tooltip state
let hoverWaveformX = null;
let hoverFftX = null;

// Initialize Sliders
sliderThreshold.addEventListener('input', () => {
    valThreshold.textContent = sliderThreshold.value;
});

sliderWindow.addEventListener('input', () => {
    valWindow.textContent = sliderWindow.value;
});

// File Upload Listeners
btnBrowse.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.click();
});

dropZone.addEventListener('click', (e) => {
    if (e.target !== btnBrowse) {
        fileInput.click();
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Demo Button
btnLoadDemo.addEventListener('click', loadDemoAudio);

// Process Button
btnProcess.addEventListener('click', processAudioFilters);

// Window Resize for Canvas Redraw
window.addEventListener('resize', () => {
    drawWaveform();
    drawFFT();
});

// Canvas Mouse Interactions
waveformCanvas.addEventListener('mousemove', (e) => {
    const rect = waveformCanvas.getBoundingClientRect();
    hoverWaveformX = e.clientX - rect.left;
    drawWaveform();
});

waveformCanvas.addEventListener('mouseleave', () => {
    hoverWaveformX = null;
    drawWaveform();
});

fftCanvas.addEventListener('mousemove', (e) => {
    const rect = fftCanvas.getBoundingClientRect();
    hoverFftX = e.clientX - rect.left;
    drawFFT();
});

fftCanvas.addEventListener('mouseleave', () => {
    hoverFftX = null;
    drawFFT();
});

// -------------------------------------------------------------
// SERVER CONNECTIVITY DETECTION
// -------------------------------------------------------------
async function detectBackendServer() {
    // Check local server endpoints
    const endpoints = [
        window.location.protocol.startsWith('http') ? '/api/health' : null,
        'http://127.0.0.1:8000/api/health'
    ].filter(Boolean);

    for (const url of endpoints) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 6000);
            const res = await fetch(url, { signal: controller.signal });
            clearTimeout(timeoutId);

            if (res.ok) {
                apiBaseUrl = url.replace('/api/health', '');
                showBanner('success', '⚡ Server Mode Connected — Python NumPy FastAPI Backend is active!');
                return true;
            }
        } catch (err) {
            // Server not reachable on this URL
        }
    }

    // Backend server not running -> Fallback to Client Mode
    apiBaseUrl = null;
    showBanner('info', 'ℹ️ Viewing in Local Browser Mode. To connect the Python NumPy backend, run <code>python server.py</code> in terminal and open <code>http://127.0.0.1:8000</code>.');
    return false;
}

function showBanner(type, messageHTML) {
    if (!banner) return;
    banner.className = `status-banner banner-${type}`;
    banner.innerHTML = messageHTML;
    banner.style.display = 'flex';
}

// -------------------------------------------------------------
// 1. UPLOAD WAV FILE (Server or Client-Side Fallback)
// -------------------------------------------------------------
async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
        alert('Please select a valid .WAV audio file.');
        return;
    }

    btnBrowse.textContent = 'Uploading...';

    // If server mode is active
    if (apiBaseUrl !== null) {
        try {
            const uploadUrl = `${apiBaseUrl}/api/upload?filename=${encodeURIComponent(file.name)}`;
            const response = await fetch(uploadUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/octet-stream' },
                body: file
            });

            if (!response.ok) {
                throw new Error('Server upload failed.');
            }

            const data = await response.json();
            updateDashboard(data);
            return;
        } catch (error) {
            console.warn('Server upload failed, falling back to client DSP:', error);
        } finally {
            btnBrowse.textContent = 'Browse Files';
        }
    }

    // Client-Side Fallback for Upload
    try {
        const arrayBuffer = await file.arrayBuffer();
        const parsed = parseWavClientSide(arrayBuffer);
        
        originalSignal = parsed.signal;
        sampleRate = parsed.sampleRate;
        durationSec = parseFloat((originalSignal.length / sampleRate).toFixed(2));
        currentAudioId = 'client_' + Date.now();

        const data = runClientDsp(file.name, originalSignal, sampleRate, 500, 5);
        updateDashboard(data);
    } catch (error) {
        alert('Failed to parse WAV file in browser: ' + error.message);
    } finally {
        btnBrowse.textContent = 'Browse Files';
    }
}

// -------------------------------------------------------------
// 2. LOAD DEMO AUDIO
// -------------------------------------------------------------
async function loadDemoAudio() {
    btnLoadDemo.disabled = true;
    btnLoadDemo.style.opacity = '0.6';

    if (apiBaseUrl !== null) {
        try {
            const response = await fetch(`${apiBaseUrl}/api/demo`);
            if (response.ok) {
                const data = await response.json();
                updateDashboard(data);
                return;
            }
        } catch (error) {
            console.warn('Server demo failed, using client DSP fallback:', error);
        } finally {
            btnLoadDemo.disabled = false;
            btnLoadDemo.style.opacity = '1';
        }
    }

    // Client-Side Demo Generation Fallback
    try {
        sampleRate = 22050;
        durationSec = 3.0;
        currentAudioId = 'demo_client_' + Date.now();
        originalSignal = generateDemoSignalClient(durationSec, sampleRate);

        const data = runClientDsp('demo_synthetic_audio.wav', originalSignal, sampleRate, 500, 5);
        updateDashboard(data);
    } catch (error) {
        alert('Error generating demo audio: ' + error.message);
    } finally {
        btnLoadDemo.disabled = false;
        btnLoadDemo.style.opacity = '1';
    }
}

// -------------------------------------------------------------
// 3. PROCESS FILTERS
// -------------------------------------------------------------
async function processAudioFilters() {
    if (!currentAudioId) {
        alert('Please load or upload an audio file first.');
        return;
    }

    btnProcess.disabled = true;
    btnProcess.textContent = 'Processing Signal...';

    const threshold = parseFloat(sliderThreshold.value);
    const windowSize = parseInt(sliderWindow.value);

    if (apiBaseUrl !== null && !currentAudioId.startsWith('client_') && !currentAudioId.startsWith('demo_client_')) {
        try {
            const response = await fetch(`${apiBaseUrl}/api/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    audio_id: currentAudioId,
                    threshold: threshold,
                    window_size: windowSize
                })
            });

            if (response.ok) {
                const data = await response.json();
                processedWaveform = data.processed_waveform || [];
                frequencies = data.frequencies || [];
                magnitudes = data.magnitudes || [];

                updateMetrics(data.original_metrics, data.processed_metrics, data.mse);
                audioPlayerProc.src = `${apiBaseUrl}/api/audio/${currentAudioId}/processed?t=${Date.now()}`;
                audioPlayerProc.load();
                drawWaveform();
                drawFFT();
                return;
            }
        } catch (error) {
            console.warn('Server processing failed, running client DSP:', error);
        } finally {
            btnProcess.disabled = false;
            btnProcess.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Apply Process Filters`;
        }
    }

    // Client-Side Processing Fallback
    if (originalSignal) {
        const data = runClientDsp('audio.wav', originalSignal, sampleRate, threshold, windowSize);
        processedWaveform = data.processed_waveform;
        frequencies = data.frequencies;
        magnitudes = data.magnitudes;

        updateMetrics(data.original_metrics, data.processed_metrics, data.mse);

        const procBlob = createWavBlob(data.processed_signal_raw, sampleRate);
        audioPlayerProc.src = URL.createObjectURL(procBlob);
        audioPlayerProc.load();

        drawWaveform();
        drawFFT();
    }

    btnProcess.disabled = false;
    btnProcess.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Apply Process Filters`;
}

// -------------------------------------------------------------
// CLIENT-SIDE DSP ENGINE (Pure JS Fallback)
// -------------------------------------------------------------
function generateDemoSignalClient(durationSec, sampleRate) {
    const numSamples = Math.floor(durationSec * sampleRate);
    const signal = new Int16Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        const clean = 12000 * Math.sin(2 * Math.PI * 440 * t) + 4000 * Math.sin(2 * Math.PI * 880 * t);
        const noise = (Math.random() - 0.5) * 1600;
        signal[i] = Math.max(-32768, Math.min(32767, Math.round(clean + noise)));
    }
    return signal;
}

function runClientDsp(filename, origSignal, sr, threshold, windowSize) {
    // Noise Reduction
    const cleaned = new Int16Array(origSignal.length);
    for (let i = 0; i < origSignal.length; i++) {
        cleaned[i] = Math.abs(origSignal[i]) < threshold ? 0 : origSignal[i];
    }

    // Moving Average Smoothing
    const smoothed = new Int16Array(origSignal.length);
    const halfWin = Math.floor(windowSize / 2);
    for (let i = 0; i < origSignal.length; i++) {
        let sum = 0;
        let count = 0;
        for (let w = -halfWin; w <= halfWin; w++) {
            const idx = i + w;
            if (idx >= 0 && idx < origSignal.length) {
                sum += cleaned[idx];
                count++;
            }
        }
        smoothed[i] = Math.round(sum / count);
    }

    // Metrics
    const origMetrics = calcSignalMetrics(origSignal);
    const procMetrics = calcSignalMetrics(smoothed);
    const mse = calcMse(origSignal, smoothed);

    // FFT & Downsampling
    const numBins = 150;
    const fftRes = calcFftClient(smoothed, sr, numBins);

    return {
        audio_id: currentAudioId,
        filename: filename,
        sample_rate: sr,
        duration_sec: parseFloat((origSignal.length / sr).toFixed(2)),
        total_samples: origSignal.length,
        channels: 1,
        threshold: threshold,
        window_size: windowSize,
        original_metrics: origMetrics,
        processed_metrics: procMetrics,
        mse: mse,
        original_waveform: downsampleArray(origSignal, 600),
        processed_waveform: downsampleArray(smoothed, 600),
        processed_signal_raw: smoothed,
        frequencies: fftRes.freqs,
        magnitudes: fftRes.mags
    };
}

function calcSignalMetrics(signal) {
    if (signal.length === 0) return { max_amplitude: 0, mean_value: 0, variance: 0, rms: 0 };
    let maxAmp = 0;
    let sum = 0;
    let sumSq = 0;

    for (let i = 0; i < signal.length; i++) {
        const val = signal[i];
        const absVal = Math.abs(val);
        if (absVal > maxAmp) maxAmp = absVal;
        sum += val;
        sumSq += val * val;
    }

    const mean = sum / signal.length;
    const variance = (sumSq / signal.length) - (mean * mean);
    const rms = Math.sqrt(sumSq / signal.length);

    return {
        max_amplitude: maxAmp,
        mean_value: parseFloat(mean.toFixed(4)),
        variance: parseFloat(variance.toFixed(2)),
        rms: parseFloat(rms.toFixed(2))
    };
}

function calcMse(orig, proc) {
    let sumSqDiff = 0;
    const len = Math.min(orig.length, proc.length);
    for (let i = 0; i < len; i++) {
        const diff = orig[i] - proc[i];
        sumSqDiff += diff * diff;
    }
    return parseFloat((sumSqDiff / len).toFixed(2));
}

function downsampleArray(arr, maxPoints) {
    if (arr.length <= maxPoints) return Array.from(arr);
    const result = [];
    const step = (arr.length - 1) / (maxPoints - 1);
    for (let i = 0; i < maxPoints; i++) {
        const idx = Math.round(i * step);
        result.push(arr[idx]);
    }
    return result;
}

function calcFftClient(signal, sr, numBins) {
    const freqs = [];
    const mags = [];
    const maxFreq = sr / 2;
    const sampleSize = Math.min(2048, signal.length);

    for (let i = 0; i < numBins; i++) {
        const targetFreq = (maxFreq / numBins) * i;
        // Simple Goertzel / DFT Magnitude estimation for visualization
        let real = 0;
        let imag = 0;
        const k = 2 * Math.PI * targetFreq / sr;
        const step = Math.max(1, Math.floor(sampleSize / 256));

        for (let n = 0; n < sampleSize; n += step) {
            const angle = k * n;
            real += signal[n] * Math.cos(angle);
            imag -= signal[n] * Math.sin(angle);
        }

        const mag = Math.sqrt(real * real + imag * imag) / (sampleSize / step);
        freqs.push(Math.round(targetFreq));
        mags.push(parseFloat(mag.toFixed(2)));
    }
    return { freqs, mags };
}

function parseWavClientSide(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const sr = view.getUint32(24, true);
    const numChannels = view.getUint16(22, true);
    const bitsPerSample = view.getUint16(34, true);
    const dataOffset = 44;
    const numSamples = Math.floor((arrayBuffer.byteLength - dataOffset) / (bitsPerSample / 8));

    const rawSignal = new Int16Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
        if (bitsPerSample === 16) {
            rawSignal[i] = view.getInt16(dataOffset + i * 2, true);
        } else if (bitsPerSample === 8) {
            rawSignal[i] = (view.getUint8(dataOffset + i) - 128) * 256;
        }
    }

    let finalSignal = rawSignal;
    if (numChannels > 1) {
        const mono = new Int16Array(Math.floor(numSamples / numChannels));
        for (let i = 0; i < mono.length; i++) {
            mono[i] = rawSignal[i * numChannels];
        }
        finalSignal = mono;
    }
    return { signal: finalSignal, sampleRate: sr };
}

function createWavBlob(int16Array, sr) {
    const buffer = new ArrayBuffer(44 + int16Array.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + int16Array.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, int16Array.length * 2, true);

    for (let i = 0; i < int16Array.length; i++) {
        view.setInt16(44 + i * 2, int16Array[i], true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// -------------------------------------------------------------
// DASHBOARD UPDATE & VISUALIZATIONS
// -------------------------------------------------------------
function updateDashboard(data) {
    currentAudioId = data.audio_id;
    originalWaveform = data.original_waveform || [];
    processedWaveform = data.processed_waveform || [];
    frequencies = data.frequencies || [];
    magnitudes = data.magnitudes || [];
    sampleRate = data.sample_rate || 22050;
    durationSec = data.duration_sec || 3.0;

    // Metadata Card
    fileInfoCard.style.display = 'block';
    document.getElementById('info-filename').textContent = data.filename;
    document.getElementById('info-samplerate').textContent = data.sample_rate.toLocaleString() + ' Hz';
    document.getElementById('info-duration').textContent = data.duration_sec + ' s';
    document.getElementById('info-samples').textContent = data.total_samples.toLocaleString();
    document.getElementById('info-channels').textContent = data.channels === 1 ? 'Mono (1)' : `Stereo (${data.channels})`;

    // Sliders
    sliderThreshold.value = data.threshold || 500;
    valThreshold.textContent = sliderThreshold.value;
    sliderWindow.value = data.window_size || 5;
    valWindow.textContent = sliderWindow.value;

    // Audio Players
    if (apiBaseUrl !== null && !currentAudioId.startsWith('client_') && !currentAudioId.startsWith('demo_client_')) {
        audioPlayerOrig.src = `${apiBaseUrl}/api/audio/${currentAudioId}/original`;
        audioPlayerProc.src = `${apiBaseUrl}/api/audio/${currentAudioId}/processed`;
    } else if (originalSignal) {
        const origBlob = createWavBlob(originalSignal, sampleRate);
        const procBlob = createWavBlob(data.processed_signal_raw || originalSignal, sampleRate);
        audioPlayerOrig.src = URL.createObjectURL(origBlob);
        audioPlayerProc.src = URL.createObjectURL(procBlob);
    }
    audioPlayerOrig.load();
    audioPlayerProc.load();

    // Metrics
    updateMetrics(data.original_metrics, data.processed_metrics, data.mse);

    // Draw Visualizations
    drawWaveform();
    drawFFT();
}

function updateMetrics(orig, proc, mse) {
    if (!proc) return;

    metricMaxAmp.textContent = proc.max_amplitude.toLocaleString();
    metricMaxAmpOrig.textContent = `Orig: ${orig ? orig.max_amplitude.toLocaleString() : '--'}`;

    metricMean.textContent = proc.mean_value;
    metricMeanOrig.textContent = `Orig: ${orig ? orig.mean_value : '--'}`;

    metricVariance.textContent = proc.variance.toLocaleString();
    metricVarOrig.textContent = `Orig: ${orig ? orig.variance.toLocaleString() : '--'}`;

    metricMse.textContent = mse !== undefined ? mse.toLocaleString() : '--';
}

function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    
    return { ctx, width: rect.width, height: rect.height };
}

function drawWaveform() {
    if (!waveformCanvas || !waveformCanvas.parentElement) return;
    const { ctx, width, height } = setupCanvas(waveformCanvas);

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    const numGridCols = 5;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#64748b';

    for (let i = 0; i <= numGridCols; i++) {
        const x = (width / numGridCols) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        const timeVal = ((durationSec / numGridCols) * i).toFixed(2) + 's';
        if (i === 0) ctx.fillText(timeVal, x + 4, height - 6);
        else if (i === numGridCols) ctx.fillText(timeVal, x - 30, height - 6);
        else ctx.fillText(timeVal, x - 12, height - 6);
    }

    if (originalWaveform.length === 0) return;

    const midY = height / 2;
    let maxAmp = 1;
    for (let i = 0; i < originalWaveform.length; i++) {
        const absVal = Math.abs(originalWaveform[i]);
        if (absVal > maxAmp) maxAmp = absVal;
    }

    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.45)';
    ctx.lineWidth = 1.5;
    const stepOrig = width / (originalWaveform.length - 1);
    for (let i = 0; i < originalWaveform.length; i++) {
        const x = i * stepOrig;
        const y = midY - (originalWaveform[i] / maxAmp) * (height * 0.42);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (processedWaveform.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = '#d946ef';
        ctx.lineWidth = 2;
        ctx.shadowColor = '#d946ef';
        ctx.shadowBlur = 6;
        const stepProc = width / (processedWaveform.length - 1);
        for (let i = 0; i < processedWaveform.length; i++) {
            const x = i * stepProc;
            const y = midY - (processedWaveform[i] / maxAmp) * (height * 0.42);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    if (hoverWaveformX !== null && hoverWaveformX >= 0 && hoverWaveformX <= width) {
        const sampleIndex = Math.floor((hoverWaveformX / width) * originalWaveform.length);
        const origVal = originalWaveform[sampleIndex] || 0;
        const procVal = processedWaveform[sampleIndex] || 0;
        const hoverTime = ((hoverWaveformX / width) * durationSec).toFixed(3);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(hoverWaveformX, 0);
        ctx.lineTo(hoverWaveformX, height);
        ctx.stroke();
        ctx.setLineDash([]);

        const tooltipText = `Time: ${hoverTime}s | Orig: ${origVal} | Proc: ${procVal}`;
        ctx.font = '11px JetBrains Mono, monospace';
        const textWidth = ctx.measureText(tooltipText).width;
        let boxX = hoverWaveformX + 8;
        if (boxX + textWidth + 12 > width) boxX = hoverWaveformX - textWidth - 20;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 1;
        ctx.beginPath();
        drawRoundedRect(ctx, boxX, 10, textWidth + 12, 24, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#f8fafc';
        ctx.fillText(tooltipText, boxX + 6, 26);
    }
}

function drawFFT() {
    if (!fftCanvas || !fftCanvas.parentElement) return;
    const { ctx, width, height } = setupCanvas(fftCanvas);

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = 1;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#64748b';

    const numFreqCols = 5;
    const maxFreq = frequencies.length > 0 ? frequencies[frequencies.length - 1] : (sampleRate / 2);

    for (let i = 0; i <= numFreqCols; i++) {
        const x = (width / numFreqCols) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        const freqVal = Math.round((maxFreq / numFreqCols) * i).toLocaleString() + ' Hz';
        if (i === 0) ctx.fillText(freqVal, x + 4, height - 6);
        else if (i === numFreqCols) ctx.fillText(freqVal, x - 45, height - 6);
        else ctx.fillText(freqVal, x - 20, height - 6);
    }

    if (frequencies.length === 0 || magnitudes.length === 0) return;

    const maxMag = Math.max(...magnitudes, 1);
    const step = width / (magnitudes.length - 1);

    const gradient = ctx.createLinearGradient(0, height, 0, 0);
    gradient.addColorStop(0, 'rgba(0, 242, 254, 0.15)');
    gradient.addColorStop(0.5, 'rgba(79, 172, 254, 0.6)');
    gradient.addColorStop(1, 'rgba(217, 70, 239, 0.95)');

    ctx.fillStyle = gradient;

    ctx.beginPath();
    ctx.moveTo(0, height);

    for (let i = 0; i < magnitudes.length; i++) {
        const x = i * step;
        const barHeight = (magnitudes[i] / maxMag) * (height * 0.82);
        const y = height - barHeight - 15;
        ctx.lineTo(x, y);
    }

    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < magnitudes.length; i++) {
        const x = i * step;
        const barHeight = (magnitudes[i] / maxMag) * (height * 0.82);
        const y = height - barHeight - 15;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (hoverFftX !== null && hoverFftX >= 0 && hoverFftX <= width) {
        const index = Math.floor((hoverFftX / width) * magnitudes.length);
        const freqHz = frequencies[index] !== undefined ? Math.round(frequencies[index]) : 0;
        const magVal = magnitudes[index] !== undefined ? Math.round(magnitudes[index]) : 0;

        ctx.strokeStyle = 'rgba(217, 70, 239, 0.5)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(hoverFftX, 0);
        ctx.lineTo(hoverFftX, height);
        ctx.stroke();
        ctx.setLineDash([]);

        const tooltipText = `Freq: ${freqHz.toLocaleString()} Hz | Mag: ${magVal.toLocaleString()}`;
        ctx.font = '11px JetBrains Mono, monospace';
        const textWidth = ctx.measureText(tooltipText).width;
        let boxX = hoverFftX + 8;
        if (boxX + textWidth + 12 > width) boxX = hoverFftX - textWidth - 20;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
        ctx.strokeStyle = '#d946ef';
        ctx.lineWidth = 1;
        ctx.beginPath();
        drawRoundedRect(ctx, boxX, 10, textWidth + 12, 24, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#f8fafc';
        ctx.fillText(tooltipText, boxX + 6, 26);
    }
}

function drawRoundedRect(ctx, x, y, w, h, r) {
    if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(x, y, w, h, r);
    } else {
        ctx.rect(x, y, w, h);
    }
}

// Auto Initialize & Detect Backend Server on Page Load
window.addEventListener('DOMContentLoaded', async () => {
    await detectBackendServer();
    loadDemoAudio();
});
