/* ══════════════════════════════════════════════════════════
   Vision-Assisted Grasping System  –  Frontend JS
   - Camera: getUserMedia (front/rear, mirror toggle)
   - Frame capture → binary WebSocket → server YOLO+MediaPipe
   - Canvas overlay: bbox, hand skeleton, direction arrow
   - Web Speech API for ASR (mic) and TTS (speaker)
   - WebSocket for guidance push + state updates
   ══════════════════════════════════════════════════════════ */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
let ws            = null;
let ttsEnabled    = true;
let asrActive     = false;
let recognition   = null;
let historyCount  = 0;

let cameraStream  = null;
let captureTimer  = null;
let isMirrored    = false;
let facingMode    = 'user';   // 'user' | 'environment'

// Hidden canvas used only for frame capture (never shown)
const _captureCanvas = document.createElement('canvas');
const _captureCtx    = _captureCanvas.getContext('2d');

// Guidance state → banner CSS class & icon
const STATE_MAP = {
  idle:        { cls: 'idle',        icon: '⌛', label: '空闲' },
  no_target:   { cls: 'searching',   icon: '🔍', label: '搜索中' },
  no_hand:     { cls: 'aligning',    icon: '🖐️', label: '等待手部' },
  aligning:    { cls: 'aligning',    icon: '➡️', label: '对齐中' },
  approaching: { cls: 'approaching', icon: '👉', label: '靠近中' },
  grasping:    { cls: 'grasping',    icon: '✊', label: '抓取中' },
  success:     { cls: 'success',     icon: '✅', label: '成功' },
};

// MediaPipe hand landmark connections (21 keypoints)
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// ─── Camera ───────────────────────────────────────────────────────────────────

async function startCamera() {
  const errEl = document.getElementById('camera-error');
  errEl.textContent = '';

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    errEl.textContent = '您的浏览器不支持摄像头API，请使用 Chrome 或 Safari';
    return;
  }

  // Stop any existing stream
  _stopStream();

  const constraints = {
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30 },
      facingMode: facingMode,
    },
    audio: false,
  };

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    const videoEl = document.getElementById('camera-view');
    videoEl.srcObject = cameraStream;
    await videoEl.play();

    // Hide prompt, show controls
    document.getElementById('camera-prompt').classList.add('hidden');
    document.getElementById('camera-controls').classList.remove('hidden');

    // Update camera status badge
    const camBadge = document.getElementById('s-camera');
    if (camBadge) { camBadge.className = 'badge active'; camBadge.textContent = '已启动'; }

    // Auto-mirror display for front camera (matches physical left/right for user)
    _applyMirror(facingMode === 'user');

    // Unlock Web Speech API on mobile (requires a user-gesture call to speak())
    _unlockTTS();

    // Start sending frames to server
    startFrameCapture();

  } catch (err) {
    let msg = '摄像头启动失败';
    if (err.name === 'NotAllowedError')  msg = '摄像头权限被拒绝，请在浏览器设置中允许';
    if (err.name === 'NotFoundError')    msg = '未找到摄像头设备';
    if (err.name === 'NotReadableError') msg = '摄像头被其他程序占用';
    errEl.textContent = msg;
  }
}

function _stopStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  stopFrameCapture();
  clearOverlayCanvas();
}

async function switchCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  await startCamera();
}

function _applyMirror(enable) {
  isMirrored = enable;
  const videoEl = document.getElementById('camera-view');
  const btn     = document.getElementById('mirror-btn');
  // Only mirror the video display; canvas overlay stays un-transformed because
  // server coordinates are already in flipped-frame space and must be drawn
  // straight onto the canvas to align with the CSS-mirrored video.
  videoEl.classList.toggle('mirrored', isMirrored);
  if (btn) btn.classList.toggle('active', isMirrored);
}

function toggleMirror() {
  _applyMirror(!isMirrored);
}

// ─── Frame capture & send ─────────────────────────────────────────────────────

const CAPTURE_FPS = 15;

function startFrameCapture() {
  stopFrameCapture();
  captureTimer = setInterval(_captureAndSend, Math.round(1000 / CAPTURE_FPS));
}

function stopFrameCapture() {
  if (captureTimer !== null) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
}

function _captureAndSend() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const videoEl = document.getElementById('camera-view');
  if (!videoEl || videoEl.readyState < 2 || videoEl.videoWidth === 0) return;

  _captureCanvas.width  = videoEl.videoWidth;
  _captureCanvas.height = videoEl.videoHeight;

  // Front camera: flip frame before sending so server guidance matches physical reality
  // (physical right = image left for front camera; flip restores correct orientation)
  if (facingMode === 'user') {
    _captureCtx.save();
    _captureCtx.scale(-1, 1);
    _captureCtx.drawImage(videoEl, -_captureCanvas.width, 0);
    _captureCtx.restore();
  } else {
    _captureCtx.drawImage(videoEl, 0, 0);
  }

  _captureCanvas.toBlob(blob => {
    if (!blob) return;
    blob.arrayBuffer().then(buf => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf);
    });
  }, 'image/jpeg', 0.7);
}

// ─── WebSocket ────────────────────────────────────────────────────────────────

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen  = () => setWSStatus(true);
  ws.onclose = () => { setWSStatus(false); setTimeout(connectWS, 2000); };
  ws.onerror = () => ws.close();

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if      (msg.type === 'guidance')          handleGuidance(msg.text);
    else if (msg.type === 'state')             handleState(msg.data);
    else if (msg.type === 'target_confirmed')  onTargetConfirmed(msg);
    else if (msg.type === 'target_cleared')    onTargetCleared();
  };
}

function sendWS(obj) {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify(obj));
}

function setWSStatus(ok) {
  const el  = document.getElementById('ws-status');
  const txt = document.getElementById('ws-text');
  el.className = 'ws-status ' + (ok ? 'connected' : 'disconnected');
  txt.textContent = ok ? '已连接' : '断开 — 重连中…';
}

// ─── Guidance handling ────────────────────────────────────────────────────────

function handleGuidance(text) {
  if (!text) return;
  document.getElementById('guidance-text').textContent = text;
  clearCompass();
  if (text.includes('向右')) activateArrow('arr-right');
  else if (text.includes('向左')) activateArrow('arr-left');
  else if (text.includes('向上')) activateArrow('arr-up');
  else if (text.includes('向下')) activateArrow('arr-down');
  if (ttsEnabled) speak(text);
  addHistory(text);
}

function handleState(data) {
  // Hand badge
  const handBadge = document.getElementById('s-hand');
  if (data.hand_detected) {
    handBadge.className = 'badge active';
    handBadge.textContent = '已检测';
  } else {
    handBadge.className = 'badge inactive';
    handBadge.textContent = '未检测';
  }

  // Target badge
  const targetBadge = document.getElementById('s-target');
  if (data.target_detected) {
    targetBadge.className = 'badge active';
    targetBadge.textContent = '已检测';
  } else {
    targetBadge.className = 'badge inactive';
    targetBadge.textContent = data.target_class ? '搜索中' : '未设置';
  }

  // Hand pose badge
  const poseBadge = document.getElementById('s-pose');
  if (data.hand_detected) {
    poseBadge.className = 'badge ' + (data.hand_open ? 'active' : 'warn');
    poseBadge.textContent = data.hand_open ? '张开' : '握拳';
  } else {
    poseBadge.className = 'badge neutral';
    poseBadge.textContent = '---';
  }

  // Guidance state badge + banner class
  const stateBadge = document.getElementById('s-state');
  const sm = STATE_MAP[data.guidance_state] || STATE_MAP.idle;
  stateBadge.className = 'badge neutral';
  stateBadge.textContent = sm.label;
  document.getElementById('guidance-banner').className = 'guidance-banner ' + sm.cls;

  // Compass center dot
  const xyAligned = ['approaching', 'grasping', 'success'].includes(data.guidance_state);
  document.getElementById('c-center').classList.toggle('aligned', xyAligned);
  if (data.guidance_state !== 'aligning') clearCompass();

  // Draw detection overlays on canvas
  drawOverlay(data);
}

// ─── Canvas overlay ───────────────────────────────────────────────────────────

/**
 * Returns the actual pixel rect of the video content within the video element,
 * accounting for letterboxing from object-fit: contain.
 */
function _getVideoDisplayRect() {
  const videoEl = document.getElementById('camera-view');
  const vw = videoEl.videoWidth  || 640;
  const vh = videoEl.videoHeight || 480;
  const cw = videoEl.offsetWidth;
  const ch = videoEl.offsetHeight;

  const videoAspect     = vw / vh;
  const containerAspect = cw / ch;

  let drawW, drawH, drawX, drawY;
  if (videoAspect > containerAspect) {
    drawW = cw;
    drawH = cw / videoAspect;
    drawX = 0;
    drawY = (ch - drawH) / 2;
  } else {
    drawH = ch;
    drawW = ch * videoAspect;
    drawX = (cw - drawW) / 2;
    drawY = 0;
  }
  return { x: drawX, y: drawY, w: drawW, h: drawH };
}

function clearOverlayCanvas() {
  const canvas = document.getElementById('overlay-canvas');
  if (!canvas) return;
  canvas.width  = canvas.offsetWidth  || 640;
  canvas.height = canvas.offsetHeight || 480;
}

function drawOverlay(state) {
  const canvas  = document.getElementById('overlay-canvas');
  if (!canvas) return;

  // Sync canvas pixel size to its CSS display size
  const dispW = canvas.offsetWidth;
  const dispH = canvas.offsetHeight;
  if (canvas.width !== dispW)  canvas.width  = dispW;
  if (canvas.height !== dispH) canvas.height = dispH;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, dispW, dispH);

  const rect = _getVideoDisplayRect();

  // Helper: convert normalised [0-1] coords → canvas pixel coords
  const px = nx => rect.x + nx * rect.w;
  const py = ny => rect.y + ny * rect.h;

  // ── Target bounding box ─────────────────────────────────────────────────
  if (state.target_bbox_norm) {
    const [x1n, y1n, x2n, y2n] = state.target_bbox_norm;
    const bx = px(x1n), by = py(y1n);
    const bw = (x2n - x1n) * rect.w;
    const bh = (y2n - y1n) * rect.h;

    // Box
    ctx.strokeStyle = '#00e640';
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(bx, by, bw, bh);

    // Corner highlights
    const cs = Math.min(bw, bh) * 0.15;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth   = 3;
    [[bx,by,1,1],[bx+bw,by,-1,1],[bx,by+bh,1,-1],[bx+bw,by+bh,-1,-1]].forEach(([cx2,cy2,sx,sy]) => {
      ctx.beginPath(); ctx.moveTo(cx2+sx*cs, cy2); ctx.lineTo(cx2, cy2); ctx.lineTo(cx2, cy2+sy*cs); ctx.stroke();
    });

    // Center dot
    const cxn = (x1n + x2n) / 2, cyn = (y1n + y2n) / 2;
    ctx.fillStyle = '#00e640';
    ctx.beginPath(); ctx.arc(px(cxn), py(cyn), 5, 0, Math.PI * 2); ctx.fill();

    // Label
    if (state.target_display) {
      ctx.font      = 'bold 13px monospace';
      ctx.fillStyle = '#00e640';
      ctx.fillText(state.target_display, bx + 4, by - 6);
    }
  }

  // ── Hand skeleton ───────────────────────────────────────────────────────
  if (state.hand_landmarks_norm) {
    const lm    = state.hand_landmarks_norm;
    const color = state.hand_open ? 'rgba(100,255,120,0.85)' : 'rgba(120,120,255,0.85)';

    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.8;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(px(lm[a][0]), py(lm[a][1]));
      ctx.lineTo(px(lm[b][0]), py(lm[b][1]));
      ctx.stroke();
    }
    ctx.fillStyle = color;
    for (const [x, y] of lm) {
      ctx.beginPath();
      ctx.arc(px(x), py(y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── Hand center indicator ───────────────────────────────────────────────
  if (state.hand_center_norm) {
    const [hxn, hyn] = state.hand_center_norm;
    const hpx = px(hxn), hpy = py(hyn);
    const color = state.hand_open ? '#64ff78' : '#6478ff';

    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.beginPath(); ctx.arc(hpx, hpy, 14, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(hpx, hpy, 4, 0, Math.PI * 2);  ctx.fill();
  }

  // ── Direction arrow ─────────────────────────────────────────────────────
  if (state.hand_center_norm && state.direction_norm) {
    const [hxn, hyn]   = state.hand_center_norm;
    const [dxr, dyr]   = state.direction_norm;
    const arrowLen = Math.min(rect.w, rect.h) * 0.12;

    const sx = px(hxn), sy = py(hyn);
    const ex = sx + dxr * arrowLen;
    const ey = sy + dyr * arrowLen;

    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth   = 3.5;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();

    // Arrowhead
    const angle  = Math.atan2(ey - sy, ex - sx);
    const tipLen = arrowLen * 0.28;
    ctx.fillStyle = '#00d4ff';
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - tipLen * Math.cos(angle - 0.45), ey - tipLen * Math.sin(angle - 0.45));
    ctx.lineTo(ex - tipLen * Math.cos(angle + 0.45), ey - tipLen * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
  }
}

// ─── Target control ───────────────────────────────────────────────────────────

function setTarget(text) {
  sendWS({ type: 'set_target', target: text });
}

function clearTarget() {
  sendWS({ type: 'clear_target' });
}

function onTargetConfirmed(msg) {
  const chip = document.getElementById('target-chip');
  chip.className = 'target-chip';
  chip.textContent = msg.display + (msg.coco_class !== msg.display ? ` (${msg.coco_class})` : '');
  document.getElementById('guidance-text').textContent = `目标已设置：${msg.display}`;
}

function onTargetCleared() {
  const chip = document.getElementById('target-chip');
  chip.className = 'target-chip empty';
  chip.textContent = '未设置';
  document.getElementById('guidance-text').textContent = '请先设置目标物体';
  document.getElementById('guidance-banner').className = 'guidance-banner idle';
  clearCompass();
  clearOverlayCanvas();
}

// ─── Web Speech API – ASR ─────────────────────────────────────────────────────

function toggleASR() {
  if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
    setASRHint('您的浏览器不支持语音识别，请使用 Chrome');
    return;
  }
  if (asrActive) { recognition && recognition.stop(); return; }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = 'zh-CN';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart  = () => { asrActive = true; setMicState(true); setASRHint('正在聆听，请说出物体名称…'); };
  recognition.onresult = (evt) => {
    const text = evt.results[0][0].transcript.trim();
    setASRHint(`识别结果："${text}"`);
    setTarget(text);
  };
  recognition.onerror  = (evt) => { setASRHint(`识别失败：${evt.error}`); stopASR(); };
  recognition.onend    = () => stopASR();
  recognition.start();
}

function stopASR() {
  asrActive = false;
  setMicState(false);
}

function setMicState(active) {
  const btn   = document.getElementById('mic-btn');
  const icon  = document.getElementById('mic-icon');
  const label = document.getElementById('mic-label');
  if (active) {
    btn.className   = 'mic-btn listening';
    icon.textContent  = '🔴';
    label.textContent = '聆听中… (点击停止)';
  } else {
    btn.className   = 'mic-btn';
    icon.textContent  = '🎤';
    label.textContent = '语音识别';
  }
}

function setASRHint(text) {
  document.getElementById('asr-hint').textContent = text;
}

// ─── Web Speech API – TTS ─────────────────────────────────────────────────────

let ttsQueue    = [];
let ttsSpeaking = false;
let ttsUnlocked = false;

/**
 * Must be called from within a user-gesture handler (e.g. button click).
 * Speaks a silent utterance to unlock Web Speech API on iOS / Android,
 * where speechSynthesis.speak() is blocked until the first gesture.
 */
function _unlockTTS() {
  if (ttsUnlocked || !('speechSynthesis' in window)) return;
  ttsUnlocked = true;
  const utt = new SpeechSynthesisUtterance('');
  utt.volume = 0;
  speechSynthesis.speak(utt);
}

function speak(text) {
  if (!ttsEnabled || !text) return;
  if (!('speechSynthesis' in window)) return;
  // Skip if this exact text is already the pending/current item (no duplicate)
  if (ttsQueue.length > 0 && ttsQueue[ttsQueue.length - 1] === text) return;
  // New different instruction: cancel stale speech so text and audio stay in sync
  speechSynthesis.cancel();
  ttsQueue = [text];
  ttsSpeaking = false;
  drainTTS();
}

function drainTTS() {
  if (ttsSpeaking || ttsQueue.length === 0) return;
  const text = ttsQueue.shift();
  const utt  = new SpeechSynthesisUtterance(text);
  utt.lang   = 'zh-CN';
  utt.rate   = 1.1;
  utt.pitch  = 1.0;
  utt.volume = 1.0;
  // Prefer a Chinese voice; fall back to any available voice (mobile may
  // only expose voices after a delay, so accept undefined = browser default)
  const voices  = speechSynthesis.getVoices();
  const zhVoice = voices.find(v => v.lang.startsWith('zh'));
  if (zhVoice) utt.voice = zhVoice;
  utt.onstart = () => { ttsSpeaking = true; };
  utt.onend = utt.onerror = () => { ttsSpeaking = false; drainTTS(); };
  speechSynthesis.speak(utt);
}

function onTTSToggle(cb) {
  ttsEnabled = cb.checked;
  document.getElementById('tts-label').textContent = ttsEnabled ? '已开启' : '已关闭';
  if (!ttsEnabled) { speechSynthesis.cancel(); ttsQueue = []; ttsSpeaking = false; }
}

// ─── Compass helpers ──────────────────────────────────────────────────────────

function clearCompass() {
  ['arr-up','arr-down','arr-left','arr-right'].forEach(id =>
    document.getElementById(id)?.classList.remove('active')
  );
}

function activateArrow(id) {
  document.getElementById(id)?.classList.add('active');
}

// ─── History ──────────────────────────────────────────────────────────────────

function addHistory(text) {
  const list  = document.getElementById('history-list');
  const empty = list.querySelector('.history-empty');
  if (empty) empty.remove();

  const timeStr = new Date().toTimeString().slice(0, 8);
  const item    = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `<span class="h-time">${timeStr}</span><span>${text}</span>`;
  list.insertBefore(item, list.firstChild);
  historyCount++;
  while (list.children.length > 30) list.removeChild(list.lastChild);
}

function clearHistory() {
  document.getElementById('history-list').innerHTML = '<div class="history-empty">暂无记录</div>';
  historyCount = 0;
}

// ─── Init ─────────────────────────────────────────────────────────────────────

window.addEventListener('load', () => {
  connectWS();

  // Pre-load TTS voices (Chrome needs this)
  if ('speechSynthesis' in window) {
    speechSynthesis.getVoices();
    speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
  }

  // Keep overlay canvas in sync with element size on resize
  window.addEventListener('resize', () => {
    clearOverlayCanvas();
  });
});
