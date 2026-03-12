# Vision-Assisted Object Grasping System

**Real-time computer vision guidance for visually impaired users** — COMP5523 Group Project, The Hong Kong Polytechnic University

A browser-based system that combines YOLOv8 object detection and MediaPipe hand tracking to deliver step-by-step voice guidance ("move right", "move closer", "grasp it") as the user reaches for a target object.

---

## Features

- **Zero-setup camera** — browser owns the capture via `getUserMedia`; no server-side OpenCV capture
- **GPU-accelerated detection** — YOLOv8n on CUDA 12.x; MediaPipe runs on CPU (TFLite XNNPACK)
- **Chinese voice I/O** — Web Speech API for both ASR (target selection) and TTS (guidance)
- **Mobile support** — HTTPS dual-server mode unlocks `getUserMedia` on iOS Safari / Android Chrome
- **Real-time canvas overlay** — bounding box, 21-point hand skeleton, direction arrow
- **State-machine policy** — priority-ordered guidance: align → approach → grasp

---

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| NVIDIA GPU + CUDA | 12.x (optional, CPU fallback supported) |
| Browser | Chrome ≥ 107 or Safari ≥ 15 |

### Install

```bash
git clone <repo-url> && cd CVP

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# GPU (RTX 50-series / CUDA 12.8) — skip on CPU-only machines
pip install torch==2.10.0+cu128 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Model weights are downloaded automatically on first run (~31 MB total).

### Run

```bash
# Desktop
python run.py
# → http://localhost:8000

# Desktop + Mobile (HTTPS required for mobile camera)
python run.py --https
# → Desktop : http://localhost:8000
# → Mobile  : https://<LAN-IP>:8443
```

On first `--https` launch a self-signed TLS certificate is auto-generated.
**Mobile cert bypass:** Chrome → *Advanced → Proceed* · Safari → *Show Details → Visit this website → Reload*

---

## How It Works

```
Browser                                     Server
──────                                      ──────
getUserMedia (15 fps)
  │ JPEG binary frame (WebSocket)
  └──────────────────────────────────────► /ws
                                            ├─ YOLOv8n  ──► target bbox
                                            └─ MediaPipe ──► 21 landmarks + gesture
                                                 │
                                            GuidancePolicy (state machine)
                                                 │ zh text + normalised coords
  ◄──────────────────────────────────────────────┘
Canvas overlay + Web Speech TTS
```

**Guidance state machine:** `IDLE → NO_TARGET → NO_HAND → ALIGNING → APPROACHING → GRASPING → SUCCESS`

---

## Usage

1. Click **启动摄像头** and grant camera permission
2. Set a target — tap a quick-pick button or use **语音识别** to say the object name in Chinese
3. Hold your hand in frame and follow spoken instructions
4. Use **⟳** to switch cameras · **⇔** to toggle mirror mode

Supported objects include: 瓶子 · 杯子 · 手机 · 苹果 · 书 · 碗 · 香蕉 · 鼠标 (and 30+ more via config)

---

## Project Structure

```
CVP/
├── app/
│   ├── config.py            # Thresholds, ZH→COCO mapping, TTS texts
│   ├── detector.py          # YOLOv8 wrapper
│   ├── hand_tracker.py      # MediaPipe GestureRecognizer wrapper
│   ├── guidance.py          # State-machine guidance policy
│   ├── camera_processor.py  # Per-frame pipeline orchestrator
│   └── main.py              # FastAPI app, WebSocket + REST endpoints
├── static/
│   ├── index.html
│   ├── style.css            # Dark glassmorphism, responsive
│   └── app.js               # Camera, WebSocket, canvas overlay, ASR/TTS
├── models/                  # Auto-downloaded weights (git-ignored)
├── run.py                   # Entry point (HTTP / HTTPS dual-server)
└── requirements.txt
```

---

## Configuration

Key parameters in `app/config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `DETECTION_CONF` | `0.40` | Lower = more detections, more false positives |
| `XY_ALIGN_THRESHOLD` | `0.12` | Normalised offset before emitting directional guidance (±77 px @ 640 px) |
| `CLOSE_DISTANCE_RATIO` | `0.65` | Fraction of object size at which grasp phase triggers |
| `GUIDANCE_COOLDOWN_SEC` | `2.5` | Minimum interval between repeated TTS phrases |

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full API reference, architecture deep-dive, and tuning guide.

---

## Acknowledgements

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · [Google MediaPipe](https://ai.google.dev/edge/mediapipe) · [FastAPI](https://fastapi.tiangolo.com/) · COMP5523, PolyU
