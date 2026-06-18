/**
 * Lingxin Sign Language Recognition - Core Application
 * Handles WebSocket connection, camera, and prediction logic
 */

const AppConfig = {
    WS_URL: `ws://${window.location.hostname}:5000/ws/detect`,
    API_URL: window.location.origin,
    SEND_FPS: 10,
    JPEG_QUALITY: 0.6,
    MAX_HISTORY: 50,
    RECONNECT_DELAY: 3000,
    RECONNECT_MAX: 10
};

const State = {
    ws: null,
    video: null,
    canvas: null,
    ctx: null,
    stream: null,
    isRunning: false,
    isPaused: false,
    sendTimer: null,
    frameCount: 0,
    lastFpsTime: Date.now(),
    currentFps: 0,
    latency: 0,
    reconnectAttempts: 0,
    wordHistory: [],
    wordFreq: {},
    currentPrediction: null,
    predictions: []
};

function initApp() {
    State.video = document.getElementById('webcam');
    State.canvas = document.getElementById('canvas');
    State.ctx = State.canvas.getContext('2d');
    loadTheme();
    checkApiHealth();
    setupKeyboardShortcuts();
    log('系统已就绪', 'info');
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            toggleCamera();
        }
        if (e.key === 'Escape') {
            if (State.isRunning) stopCamera();
        }
    });
}

async function checkApiHealth() {
    try {
        const r = await fetch(`${AppConfig.API_URL}/api/health`);
        const d = await r.json();
        updateConnectionStatus('online', `已连接 | ${d.model_type?.toUpperCase() || '--'}`);
        log('API服务连接成功', 'success');
    } catch (e) {
        updateConnectionStatus('offline', 'API未连接');
        log('API服务未连接，请先启动: python api/app.py', 'error');
    }
}

function updateConnectionStatus(status, text) {
    const dot = document.getElementById('statusDot');
    const label = document.getElementById('statusText');
    if (dot) {
        dot.className = 'status-dot ' + status;
    }
    if (label) {
        label.textContent = text;
    }
}

async function toggleCamera() {
    if (State.isRunning) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    const btn = document.getElementById('btnToggle');
    try {
        btn.disabled = true;
        btn.textContent = '启动中...';
        log('请求摄像头权限...', 'info');

        State.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        State.video.srcObject = State.stream;
        State.video.style.display = 'block';
        await State.video.play();

        State.canvas.width = State.video.videoWidth;
        State.canvas.height = State.video.videoHeight;

        State.isRunning = true;
        State.isPaused = false;
        document.getElementById('videoPlaceholder').style.display = 'none';
        btn.textContent = '停止识别';
        btn.disabled = false;
        btn.className = 'control-btn control-btn-danger';

        log(`摄像头已开启 (${State.video.videoWidth}×${State.video.videoHeight})`, 'success');
        connectWebSocket();
        startSendLoop();
    } catch (e) {
        log('启动失败: ' + e.message, 'error');
        btn.textContent = '开始识别';
        btn.disabled = false;
    }
}

function stopCamera() {
    State.isRunning = false;
    if (State.sendTimer) {
        clearTimeout(State.sendTimer);
        State.sendTimer = null;
    }
    if (State.ws) {
        State.ws.close();
        State.ws = null;
    }
    if (State.stream) {
        State.stream.getTracks().forEach(t => t.stop());
        State.stream = null;
    }
    State.video.style.display = 'none';
    document.getElementById('videoPlaceholder').style.display = 'flex';
    const btn = document.getElementById('btnToggle');
    btn.textContent = '开始识别';
    btn.className = 'control-btn control-btn-primary';
    State.ctx.clearRect(0, 0, State.canvas.width, State.canvas.height);
    updateConnectionStatus('offline', '已停止');
    log('摄像头已关闭', 'info');
}

function connectWebSocket() {
    if (State.ws && State.ws.readyState === WebSocket.OPEN) return;

    log('正在连接WebSocket...', 'info');
    updateConnectionStatus('connecting', '连接中...');

    State.ws = new WebSocket(AppConfig.WS_URL);

    State.ws.onopen = () => {
        State.reconnectAttempts = 0;
        updateConnectionStatus('online', '已连接');
        log('WebSocket连接成功', 'success');
    };

    State.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handlePrediction(data);
    };

    State.ws.onclose = () => {
        if (State.isRunning && State.reconnectAttempts < AppConfig.RECONNECT_MAX) {
            State.reconnectAttempts++;
            log(`连接断开，${AppConfig.RECONNECT_DELAY/1000}秒后重试 (${State.reconnectAttempts}/${AppConfig.RECONNECT_MAX})`, 'warning');
            updateConnectionStatus('connecting', '重连中...');
            setTimeout(connectWebSocket, AppConfig.RECONNECT_DELAY);
        } else {
            updateConnectionStatus('offline', '连接断开');
        }
    };

    State.ws.onerror = (e) => {
        log('WebSocket连接错误', 'error');
    };
}

function startSendLoop() {
    const interval = 1000 / AppConfig.SEND_FPS;
    function loop() {
        if (!State.isRunning || State.isPaused) return;
        sendFrame();
        State.sendTimer = setTimeout(loop, interval);
    }
    loop();
}

function sendFrame() {
    if (!State.isRunning || !State.ws || State.ws.readyState !== WebSocket.OPEN) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = State.video.videoWidth;
    tempCanvas.height = State.video.videoHeight;
    tempCanvas.getContext('2d').drawImage(State.video, 0, 0);
    const base64 = tempCanvas.toDataURL('image/jpeg', AppConfig.JPEG_QUALITY).split(',')[1];

    State.frameCount++;
    const now = Date.now();
    if (now - State.lastFpsTime >= 1000) {
        State.currentFps = State.frameCount;
        State.frameCount = 0;
        State.lastFpsTime = now;
        updateFpsDisplay();
    }

    const t0 = performance.now();
    State.ws.send(JSON.stringify({ image: base64 }));
    State._lastSendTime = t0;
}

function handlePrediction(data) {
    State.latency = Math.round(performance.now() - (State._lastSendTime || performance.now()));
    updateLatencyDisplay();

    if (data.error) {
        log('预测错误: ' + data.error, 'error');
        return;
    }

    if (data.has_hand && data.predictions && data.predictions.length > 0) {
        State.predictions = data.predictions;
        const top = data.predictions[0];
        State.currentPrediction = top;
        updatePredictionDisplay(top);
        updateTopPredictions(data.predictions.slice(0, 5));
        addHistory(top.word, top.confidence);
        updateWordFreq(top.word);
        if (window.updateCharts) {
            window.updateCharts(data.predictions);
        }
    } else {
        clearPredictionDisplay();
    }

    if (data.buffer_size !== undefined) {
        updateBufferDisplay(data.buffer_size, data.buffer_target);
    }
}

function updatePredictionDisplay(pred) {
    const wordEl = document.getElementById('currentWord');
    const confEl = document.getElementById('currentConf');
    const confBar = document.getElementById('confidenceBarFill');
    const overlay = document.getElementById('predictionOverlay');
    const overlayWord = document.getElementById('overlayWord');
    const overlayConf = document.getElementById('overlayConf');

    if (wordEl) {
        wordEl.textContent = pred.word;
        wordEl.style.color = '#6366f1';
    }
    if (confEl) {
        confEl.textContent = (pred.confidence * 100).toFixed(1) + '%';
    }
    if (confBar) {
        const pct = Math.round(pred.confidence * 100);
        confBar.style.width = pct + '%';
        confBar.className = 'confidence-bar-fill ' + (pct >= 70 ? 'high' : pct >= 40 ? 'medium' : 'low');
    }
    if (overlay) {
        overlay.style.display = 'block';
    }
    if (overlayWord) {
        overlayWord.textContent = pred.word;
        overlayWord.classList.add('show');
    }
    if (overlayConf) {
        overlayConf.textContent = (pred.confidence * 100).toFixed(0) + '%';
        overlayConf.classList.add('show');
    }
}

function clearPredictionDisplay() {
    const wordEl = document.getElementById('currentWord');
    const confEl = document.getElementById('currentConf');
    const confBar = document.getElementById('confidenceBarFill');
    const overlayWord = document.getElementById('overlayWord');
    const overlayConf = document.getElementById('overlayConf');

    if (wordEl) {
        wordEl.textContent = '--';
        wordEl.style.color = '#64748b';
    }
    if (confEl) confEl.textContent = '0%';
    if (confBar) confBar.style.width = '0%';
    if (overlayWord) overlayWord.classList.remove('show');
    if (overlayConf) overlayConf.classList.remove('show');
}

function updateTopPredictions(preds) {
    const container = document.getElementById('topPredictions');
    if (!container) return;

    container.innerHTML = preds.map((p, i) => `
        <div class="top-prediction ${i === 0 ? 'active' : ''}">
            <div class="top-prediction-rank">${i + 1}</div>
            <div class="top-prediction-info">
                <div class="top-prediction-word">${p.word}</div>
                <div class="top-prediction-category">${p.category || ''}</div>
            </div>
            <div class="top-prediction-confidence">${(p.confidence * 100).toFixed(1)}%</div>
        </div>
    `).join('');
}

function addHistory(word, confidence) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('zh-CN');
    const last = State.wordHistory[0];
    if (last && last.word === word && now - last.timestamp < 1500) return;

    State.wordHistory.unshift({ word, confidence, timestamp: now, timeStr });
    if (State.wordHistory.length > AppConfig.MAX_HISTORY) {
        State.wordHistory.pop();
    }
    renderHistory();
}

function renderHistory() {
    const container = document.getElementById('historyList');
    if (!container) return;

    if (State.wordHistory.length === 0) {
        container.innerHTML = '<div style="text-align:center;padding:24px;color:#64748b;">暂无识别记录</div>';
        return;
    }

    container.innerHTML = State.wordHistory.slice(0, 15).map(h => `
        <div class="history-item">
            <span class="history-time">${h.timeStr}</span>
            <span class="history-word">${h.word}</span>
            <span class="history-confidence">${(h.confidence * 100).toFixed(0)}%</span>
        </div>
    `).join('');
}

function updateWordFreq(word) {
    State.wordFreq[word] = (State.wordFreq[word] || 0) + 1;
    if (window.updateWordFreqChart) {
        window.updateWordFreqChart(State.wordFreq);
    }
}

function updateFpsDisplay() {
    const el = document.getElementById('fpsValue');
    if (el) el.textContent = State.currentFps;
}

function updateLatencyDisplay() {
    const el = document.getElementById('latencyValue');
    if (el) el.textContent = State.latency;
}

function updateBufferDisplay(size, target) {
    const bar = document.getElementById('bufferBar');
    const text = document.getElementById('bufferText');
    if (bar) bar.style.width = (size / target * 100) + '%';
    if (text) text.textContent = `${size} / ${target}`;
}

function clearHistory() {
    State.wordHistory = [];
    State.wordFreq = {};
    renderHistory();
    clearPredictionDisplay();
    if (window.clearCharts) window.clearCharts();
    log('历史记录已清空', 'info');
}

function log(message, type = 'info') {
    const container = document.getElementById('logConsole');
    if (!container) return;
    const time = new Date().toLocaleTimeString('zh-CN');
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.innerHTML = `<span class="time">[${time}]</span> <span class="msg">${message}</span>`;
    container.insertBefore(line, container.firstChild);
    if (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
}

function togglePause() {
    State.isPaused = !State.isPaused;
    const btn = document.getElementById('btnPause');
    if (btn) {
        btn.textContent = State.isPaused ? '继续识别' : '暂停识别';
        btn.className = State.isPaused ? 'control-btn control-btn-primary' : 'control-btn control-btn-outline';
    }
    log(State.isPaused ? '识别已暂停' : '识别已继续', 'info');
}

// ===== Dark Mode Toggle =====
function toggleDarkMode() {
    const body = document.body;
    const btn = document.getElementById('themeToggle');
    if (body.classList.contains('dark')) {
        body.classList.remove('dark');
        body.classList.add('light');
        if (btn) btn.textContent = '☀️';
        localStorage.setItem('theme', 'light');
    } else {
        body.classList.remove('light');
        body.classList.add('dark');
        if (btn) btn.textContent = '🌙';
        localStorage.setItem('theme', 'dark');
    }
}

function loadTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'light') {
        document.body.classList.remove('dark');
        document.body.classList.add('light');
        const btn = document.getElementById('themeToggle');
        if (btn) btn.textContent = '☀️';
    }
}

// ===== Export History =====
function exportHistory() {
    if (State.wordHistory.length === 0) {
        log('没有可导出的记录', 'warning');
        return;
    }

    const csv = ['时间,词汇,置信度'];
    State.wordHistory.forEach(h => {
        csv.push(`${h.timeStr},${h.word},${(h.confidence * 100).toFixed(1)}%`);
    });

    const blob = new Blob(['\uFEFF' + csv.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `lingxin_history_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    log(`已导出 ${State.wordHistory.length} 条记录`, 'success');
}

// Expose to global scope for other modules
window.State = State;
window.App = {
    init: initApp,
    toggleCamera,
    stopCamera,
    clearHistory,
    togglePause,
    toggleDarkMode,
    exportHistory,
    log
};
