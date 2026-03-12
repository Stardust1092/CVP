# Development Guide

> **Vision-Assisted Object Grasping System** · COMP5523 · PolyU
> Internal reference for contributors and maintainers. Last updated: 2026-03-12.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Layout](#2-repository-layout)
3. [Data Flow](#3-data-flow)
4. [Module Reference](#4-module-reference)
5. [WebSocket Protocol](#5-websocket-protocol)
6. [REST API](#6-rest-api)
7. [Guidance State Machine](#7-guidance-state-machine)
8. [Configuration Reference](#8-configuration-reference)
9. [Frontend Internals](#9-frontend-internals)
10. [Running the Server](#10-running-the-server)
11. [Performance & Tuning](#11-performance--tuning)
12. [Troubleshooting](#12-troubleshooting)
13. [Extension Points](#13-extension-points)

---

## 1. Architecture Overview

The system is split into a **browser-owned capture layer** and a **server-side inference layer**, connected by a single persistent WebSocket.

```
┌─────────────────────────────────────────────────────────┐
│  Browser                                                │
│                                                         │
│  getUserMedia ──► <video> ──► captureCanvas (15 fps)    │
│       │                             │                   │
│       │              JPEG 0.7 quality (binary WS frame) │
│       │                             │                   │
│  Web Speech ASR ──► JSON text ──────┤                   │
│                                     │                   │
│  ◄── JSON state ◄── WebSocket /ws ◄─┘                   │
│        │                                                │
│  Canvas overlay                                         │
│  (bbox · hand skeleton · direction arrow)               │
│                                                         │
│  Web Speech TTS ◄── guidance text                       │
└──────────────────────────┬──────────────────────────────┘
                           │  ws://localhost:8000/ws
                           │  wss://<LAN-IP>:8443/ws
┌──────────────────────────▼──────────────────────────────┐
│  FastAPI Server (app/main.py)                           │
│                                                         │
│  asyncio event loop                                     │
│    ws.receive() ──► binary ──► ThreadPoolExecutor(1)    │
│                                   │                     │
│                           FrameProcessor                │
│                           ├─ ObjectDetector (YOLOv8n)   │
│                           ├─ HandTracker (MediaPipe)    │
│                           └─ GuidancePolicy             │
│                                   │                     │
│    ws.send_json() ◄───────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Browser-owned camera | Works over LAN without server USB/V4L access; enables mobile use |
| Binary WebSocket (not HTTP multipart) | Lower overhead, no per-frame HTTP handshake |
| `ThreadPoolExecutor(max_workers=1)` | Serialises YOLO + MediaPipe calls; prevents GPU memory contention |
| `asyncio.Lock` back-pressure | Drops new frame if the previous one is still being processed, keeping the UI responsive |
| Normalised coordinates (0–1) | Frontend overlay is resolution-independent; works at any camera resolution |
| Self-signed TLS cert (auto-generated) | Enables `getUserMedia` on mobile over LAN without a CA-signed cert |

---

## 2. Repository Layout

```
CVP/
├── app/                         # Backend Python package
│   ├── __init__.py
│   ├── config.py                # All tuneable constants and mappings
│   ├── detector.py              # YOLOv8 inference wrapper
│   ├── hand_tracker.py          # MediaPipe GestureRecognizer wrapper
│   ├── guidance.py              # Guidance state machine
│   ├── camera_processor.py      # Per-frame pipeline (no camera capture)
│   └── main.py                  # FastAPI app, WebSocket + REST routes
├── static/                      # Frontend (served as static files)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/                      # Model weights (auto-downloaded, git-ignored)
│   └── gesture_recognizer.task  # ~25 MB, MediaPipe float16
├── run.py                       # CLI entry point (HTTP / HTTPS)
├── requirements.txt
├── cert.pem                     # Auto-generated TLS cert (git-ignored)
├── key.pem                      # Auto-generated TLS key (git-ignored)
└── DEVELOPMENT.md               # This file
```

---

## 3. Data Flow

### Per-frame pipeline (backend)

```
JPEG bytes (WebSocket binary)
        │
        ▼
cv2.imdecode()  →  BGR ndarray (H×W×3)
        │
        ├──► ObjectDetector.detect(frame, target_class)
        │         YOLOv8n inference (GPU)
        │         Filter by target COCO class
        │         Return highest-confidence bbox (x1,y1,x2,y2) or None
        │
        ├──► HandTracker.track(frame)
        │         BGR → RGB conversion
        │         MediaPipe GestureRecognizer.recognize_for_video()
        │         Return list[NormalizedLandmark] (21 pts) or None
        │         Side-effect: store gesture label (Open_Palm / Closed_Fist …)
        │
        └──► GuidancePolicy.update(hand_center, obj_bbox, hand_open, shape, target)
                  Priority-ordered state transitions (see §7)
                  Cooldown check  →  emit zh_text or None
                  Return (zh_text, cv_text)

State dict (normalised coords)  →  WebSocket JSON  →  Browser
```

### Frontend capture & send

```
setInterval(_captureAndSend, 1000/15)   ← 15 fps timer
        │
        ▼
drawImage(videoEl)  →  _captureCanvas
        │
        ├── facingMode === 'user':
        │       ctx.scale(-1, 1)  →  flip horizontally
        │       (server receives mirrored frame → guidance L/R matches physical)
        │
        └── facingMode === 'environment':
                draw as-is
        │
toBlob('image/jpeg', 0.7)  →  arrayBuffer  →  ws.send(buf)
```

> **Mirror invariant:** The `<video>` element is CSS-mirrored (`scaleX(-1)`) for front camera display. The canvas overlay is **not** CSS-mirrored; its server-returned coordinates (based on the flipped frame) map directly onto the mirrored video display.

---

## 4. Module Reference

### `app/config.py`

Single source of truth for all constants. No logic.

```python
YOLO_MODEL        = "yolov8n.pt"
DETECTION_CONF    = 0.40          # YOLO confidence threshold
DETECTION_IOU     = 0.45          # NMS IoU threshold
MAX_HANDS         = 1
HAND_DETECT_CONF  = 0.65
HAND_TRACK_CONF   = 0.50
XY_ALIGN_THRESHOLD   = 0.12      # normalised frame fraction
CLOSE_DISTANCE_RATIO = 0.65      # relative to object longest side
GUIDANCE_COOLDOWN_SEC = 2.5
ZH_TO_COCO: dict[str, str]       # Chinese → COCO-80 class name
GUIDANCE_TEXTS: dict[str, str]   # key → Chinese TTS string
CV_TEXTS: dict[str, str]         # key → English overlay string
```

---

### `app/detector.py` — `ObjectDetector`

| Method | Signature | Description |
|---|---|---|
| `detect` | `(frame, target_class?) → list[dict]` | Run YOLOv8n; filter by class if given; sort by confidence desc |
| `draw_detections` | `(frame, detections, target_class?) → frame` | Draw bboxes in-place (not used in WebSocket mode) |

Detection dict schema:
```python
{ "class": "bottle", "conf": 0.87, "bbox": (x1, y1, x2, y2) }
```

---

### `app/hand_tracker.py` — `HandTracker`

Uses `GestureRecognizer` (MediaPipe Tasks) in `VIDEO` mode. Timestamps are forced strictly increasing via `max(perf_counter_ms, last_ts + 1)`.

| Method | Returns | Description |
|---|---|---|
| `track(frame)` | `list[NormalizedLandmark] \| None` | Infer on one BGR frame; update `_last_gesture` |
| `get_palm_center(lm, shape)` | `(x, y)` px | Mean of wrist + 4 MCP joints |
| `is_hand_open(lm)` | `bool` | Uses gesture label; falls back to fingertip-spread heuristic |
| `get_gesture_label()` | `str` | Last detected gesture name |

Open gestures: `Open_Palm`, `Victory`, `ILoveYou`, `Pointing_Up`
Close gestures: `Closed_Fist`, `Thumb_Down`, `Thumb_Up`

---

### `app/guidance.py` — `GuidancePolicy`

Stateful class. One instance per server lifetime (shared across WS connections).

| Method | Description |
|---|---|
| `update(hand_center, obj_bbox, hand_open, frame_shape, target_class)` | Compute next instruction; return `(zh_text, cv_text)` or `(None, None)` |
| `get_direction_vector(hand_center, obj_bbox, frame_shape)` | Unit vector hand→object; `None` if already aligned (mag < 0.02) |
| `reset()` | Call when target changes |

**`_emit(key)`** — respects `GUIDANCE_COOLDOWN_SEC`: identical instruction repeats only after cooldown; a different instruction fires immediately.

---

### `app/camera_processor.py` — `FrameProcessor`

Singleton created at app startup. Thread-safe via `threading.Lock` on target state.

| Method | Thread-safe | Description |
|---|---|---|
| `set_target(text)` | ✓ | Map Chinese→COCO; reset guidance |
| `clear_target()` | ✓ | Reset state |
| `process_frame(jpeg_bytes)` | Executor thread | Full pipeline; returns state dict |
| `pop_guidance()` | ✓ | Dequeue one guidance string (or `None`) |
| `get_state()` | ✓ | Snapshot of last processed frame state |

State dict keys: `hand_detected`, `target_detected`, `hand_open`, `guidance_state`, `last_guidance`, `target_class`, `target_display`, `target_bbox_norm`, `hand_center_norm`, `hand_landmarks_norm`, `direction_norm`

---

### `app/main.py` — FastAPI Application

**WebSocket lifecycle:**

```python
await ws.accept()
loop = asyncio.get_running_loop()
proc_lock = asyncio.Lock()

while True:
    raw = await asyncio.wait_for(ws.receive(), timeout=1.0)

    if raw["type"] == "websocket.disconnect":
        break                         # clean exit

    if raw.get("bytes"):
        if proc_lock.locked():
            continue                  # back-pressure: drop frame
        async with proc_lock:
            state = await loop.run_in_executor(_exec, processor.process_frame, raw["bytes"])
        ...

except (WebSocketDisconnect, RuntimeError):
    pass
```

**Why `asyncio.get_running_loop()` not `get_event_loop()`:** `get_event_loop()` is deprecated in Python 3.12+ and may raise in some executor contexts.

---

## 5. WebSocket Protocol

**Endpoint:** `ws://host:8000/ws` (HTTP) · `wss://host:8443/ws` (HTTPS)

### Client → Server

| Message type | Format | Description |
|---|---|---|
| Video frame | `bytes` (JPEG) | Raw JPEG frame from browser camera |
| Set target | `text` (JSON) | `{"type": "set_target", "target": "瓶子"}` |
| Clear target | `text` (JSON) | `{"type": "clear_target"}` |

### Server → Client

**`guidance`** — emitted when a new instruction is ready:
```json
{
  "type": "guidance",
  "text": "向右移动"
}
```

**`state`** — emitted after every processed frame:
```json
{
  "type": "state",
  "data": {
    "hand_detected": true,
    "target_detected": true,
    "hand_open": false,
    "guidance_state": "aligning",
    "last_guidance": "向右移动",
    "target_class": "bottle",
    "target_display": "瓶子",
    "target_bbox_norm": [0.31, 0.18, 0.62, 0.74],
    "hand_center_norm": [0.50, 0.50],
    "hand_landmarks_norm": [[0.48, 0.52], "...21 total"],
    "direction_norm": [0.71, 0.00]
  }
}
```

> All spatial values are normalised to `[0.0, 1.0]` relative to frame width/height.
> `direction_norm` is a unit vector `[dx, dy]` from hand center to object center; `null` when already aligned (magnitude < 0.02).

**`target_confirmed`**:
```json
{"type": "target_confirmed", "display": "瓶子", "coco_class": "bottle"}
```

**`target_cleared`**:
```json
{"type": "target_cleared"}
```

---

## 6. REST API

Intended for testing and external integration. The WebSocket path is preferred for real-time use.

### `POST /api/set_target`

```http
POST /api/set_target
Content-Type: application/json

{"target": "苹果"}
```

```json
{"coco_class": "apple", "display": "苹果"}
```

### `POST /api/clear_target`

```http
POST /api/clear_target
```

```json
{"status": "cleared"}
```

### `GET /api/state`

Returns the same dict as the WebSocket `state.data` payload.

---

## 7. Guidance State Machine

### States

| State | Description |
|---|---|
| `idle` | No target set |
| `no_target` | Target set but not detected in frame |
| `no_hand` | Target detected, no hand in frame |
| `aligning` | Hand detected; guiding left/right/up/down |
| `approaching` | XY aligned; guiding hand forward (depth) |
| `grasping` | Hand close enough; instruct open/close |
| `success` | Hand closed at object — task complete |

### Transition Logic (priority-ordered)

```
target_class is None
  └──► IDLE

obj_bbox is None
  └──► NO_TARGET  →  emit "no_target"

hand_center is None
  └──► NO_HAND    →  emit "no_hand"

|Δx_norm| > XY_ALIGN_THRESHOLD (0.12)
  └──► ALIGNING   →  emit "move_right" or "move_left"

|Δy_norm| > XY_ALIGN_THRESHOLD (0.12)
  └──► ALIGNING   →  emit "move_down" or "move_up"

dist_px ≥ obj_size × CLOSE_DISTANCE_RATIO (0.65)
  └──► APPROACHING →  emit "move_forward"

hand_open == True
  └──► GRASPING   →  emit "grasp"   ("握住目标")

hand_open == False (first time)
  └──► SUCCESS    →  emit "success"
```

Where:
- `Δx_norm = (obj_cx - hand_cx) / frame_W`
- `dist_px = euclidean(hand_center, obj_center)`
- `obj_size = max(obj_w, obj_h)`

---

## 8. Configuration Reference

All in `app/config.py`. No environment variables or external config files.

### Detection

| Key | Default | Range | Notes |
|---|---|---|---|
| `YOLO_MODEL` | `"yolov8n.pt"` | any YOLO weights | `n`=fastest, `s/m/l/x`=more accurate |
| `DETECTION_CONF` | `0.40` | 0.1–0.9 | Lower → more recall, more noise |
| `DETECTION_IOU` | `0.45` | 0.1–0.9 | NMS threshold |

### Hand Tracking

| Key | Default | Notes |
|---|---|---|
| `MAX_HANDS` | `1` | Increase only with multi-hand guidance logic |
| `HAND_DETECT_CONF` | `0.65` | Detection confidence (higher = fewer false positives) |
| `HAND_TRACK_CONF` | `0.50` | Tracking confidence per frame |

### Guidance

| Key | Default | Tuning guide |
|---|---|---|
| `XY_ALIGN_THRESHOLD` | `0.12` | Increase → guidance fires only when hand is further off-center. At 640 px: 0.12 → ±77 px |
| `CLOSE_DISTANCE_RATIO` | `0.65` | Decrease → grasp phase triggers earlier (hand farther from object) |
| `GUIDANCE_COOLDOWN_SEC` | `2.5` | Decrease → more frequent TTS; increase → quieter but slower feedback |

### Adding Object Mappings

Add entries to `ZH_TO_COCO` to support new Chinese names:
```python
ZH_TO_COCO: dict[str, str] = {
    "水壶": "bottle",
    "保温杯": "cup",
    # COCO-80 class names: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
}
```

---

## 9. Frontend Internals

### Camera & Capture (`app.js`)

```
startCamera()
  ├── navigator.mediaDevices.getUserMedia({ video: {facingMode} })
  ├── videoEl.srcObject = stream
  ├── _applyMirror(facingMode === 'user')   ← CSS scaleX(-1) on video only
  ├── _unlockTTS()                          ← iOS/Android audio unlock (user gesture required)
  └── startFrameCapture()                   ← setInterval @ 15 fps

_captureAndSend()
  ├── drawImage(videoEl)  [raw frame, ignores CSS transform]
  ├── if front camera: ctx.scale(-1,1) flip  ← server sees corrected orientation
  └── toBlob(jpeg, 0.7) → ws.send(buf)
```

### Canvas Overlay

`drawOverlay(state)` renders onto `#overlay-canvas` (positioned over `<video>`):
- **Bounding box** — green rect + corner accents + center dot + label
- **Hand skeleton** — 21 keypoints + HAND_CONNECTIONS edges (green=open, blue=closed)
- **Hand center** — circle at palm center
- **Direction arrow** — line + arrowhead from hand center toward object center

> Canvas is **not** CSS-transformed. Server coordinates (from flipped frame) align directly with the CSS-mirrored video display. Applying CSS mirror to the canvas would double-flip all overlays.

### TTS (Web Speech API)

```
speak(text)
  ├── deduplicate: skip if same as last queued
  ├── speechSynthesis.cancel()   ← abort stale speech immediately
  └── drainTTS()
        └── SpeechSynthesisUtterance(text)
              lang='zh-CN', rate=1.1
              prefer zhVoice → fallback to browser default
              speechSynthesis.speak(utt)

_unlockTTS()    ← called once from startCamera() (user gesture context)
  └── speak silent utterance (volume=0) to unlock mobile audio
```

**iOS/Android note:** `speechSynthesis.speak()` is blocked until the first call within a user-gesture handler. `_unlockTTS()` fires a silent utterance during the camera-start click to satisfy this requirement.

### ASR (Web Speech API)

```
toggleASR()
  └── SpeechRecognition({ lang: 'zh-CN', interimResults: false })
        onresult → setTarget(transcript)   ← sends WS set_target
```

---

## 10. Running the Server

### `run.py` CLI

```
python run.py [--https] [--port PORT]
```

| Flag | Default | Description |
|---|---|---|
| `--https` | off | Start HTTPS server on port 8443 in addition to HTTP on 8000 |
| `--port` | `8000` | HTTP port (HTTPS always uses 8443) |

### Dual-server mode (`--https`)

```python
# HTTP on port 8000 (background thread, own event loop)
threading.Thread(target=_run_server, kwargs={host, port=8000}).start()

# HTTPS on port 8443 (main thread, blocks)
_run_server(host, port=8443, ssl_certfile, ssl_keyfile)
```

Both servers share the same `app.main` module import → same `FrameProcessor` singleton.

### TLS Certificate

Auto-generated via `cryptography` if `cert.pem` / `key.pem` are absent:
- RSA 2048, valid 365 days
- SAN: `localhost`, `127.0.0.1`, detected LAN IP

---

## 11. Performance & Tuning

### Baseline (RTX 5070 Laptop, 640×480 @ 15 fps)

| Stage | Typical latency |
|---|---|
| JPEG decode | ~1 ms |
| YOLOv8n (GPU) | ~8–12 ms |
| MediaPipe Gesture (CPU XNNPACK) | ~15–25 ms |
| Guidance + serialise | < 1 ms |
| **End-to-end server** | ~25–40 ms |

### Reducing latency on low-end hardware

1. Lower capture FPS: `const CAPTURE_FPS = 10` in `app.js`
2. Lower resolution: set `width: { ideal: 320 }, height: { ideal: 240 }` in `getUserMedia` constraints
3. Switch to lighter model: `YOLO_MODEL = "yolov8n.pt"` (already the lightest)
4. JPEG quality: `toBlob('image/jpeg', 0.5)` — reduces bandwidth, slightly lower accuracy

### Back-pressure

If `proc_lock` is held when a new frame arrives, the frame is dropped (not queued). This prevents unbounded memory growth and ensures the UI always reflects the latest camera state.

---

## 12. Troubleshooting

### WebSocket shows "连接中" (connecting loop)

**Symptom:** Status badge never turns green; recognition doesn't work.
**Cause:** WebSocket handler crashed with `RuntimeError: Cannot call "receive" once a disconnect message has been received`.
**Fix:** Already patched in `main.py` — ensure `raw.get("type") == "websocket.disconnect"` check is present and `except (WebSocketDisconnect, RuntimeError)` catches both.

### Camera not starting on mobile

**Cause:** `getUserMedia` requires HTTPS on non-localhost origins.
**Fix:** Run with `python run.py --https` and access via `https://<LAN-IP>:8443`.

### No TTS audio on mobile

**Cause:** iOS/Android block `speechSynthesis.speak()` until a user-gesture call.
**Fix:** `_unlockTTS()` is called inside `startCamera()` (button click context). Ensure it fires before any guidance message arrives.

### Hand not detected

- Ensure adequate lighting (MediaPipe struggles in very low light)
- Keep hand within the center 80% of frame
- If `HAND_DETECT_CONF` is too high, lower it to `0.50`

### MediaPipe timestamp error

**Symptom:** `Timestamp must be monotonically increasing`.
**Fix:** Already patched in `hand_tracker.py` — `ts_ms = max(perf_counter_ms - t0, last_ts + 1)`.

### Chinese characters garbled in terminal (Windows)

```bash
set PYTHONIOENCODING=utf-8   # before running python run.py
```

### Model download fails

Manually place models in `models/`:
- `gesture_recognizer.task` — [download from Google](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)
- `yolov8n.pt` — [download from Ultralytics](https://github.com/ultralytics/assets/releases)

---

## 13. Extension Points

### Adding new supported objects

Edit `ZH_TO_COCO` in `config.py` and add a quick-pick button in `index.html`.

### Supporting additional languages

1. Add ASR language: change `recognition.lang` in `app.js`
2. Add TTS voice: add a new `GUIDANCE_TEXTS_XX` dict in `config.py`
3. Send language preference via WebSocket and switch in `GuidancePolicy._emit()`

### Multi-hand guidance

1. Increase `MAX_HANDS` in `config.py`
2. Modify `HandTracker.track()` to return all landmark sets
3. Update `FrameProcessor` to select the hand closest to target

### Depth estimation (replacing "move forward")

Replace the `CLOSE_DISTANCE_RATIO` heuristic with a depth model (e.g. MiDaS) to provide metric distance guidance. Hook into `FrameProcessor.process_frame()` after the existing detection steps.

### Recording / logging

`FrameProcessor.process_frame()` returns a full state dict on every call. Add a logging hook there to record sessions for evaluation or replay.

---

*For questions about the project, open an issue in the repository.*
