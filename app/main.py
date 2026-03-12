"""
FastAPI application entry point.
Provides:
  WS   /ws              → bidirectional: receives JPEG frames (binary) +
                          target commands (JSON text); pushes state + guidance
  POST /api/set_target  → set target object (REST alternative)
  GET  /api/state       → current system state
  Static files from /static/ (index.html served at /)
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .camera_processor import FrameProcessor

processor = FrameProcessor()
_exec = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _exec.shutdown(wait=False)


app = FastAPI(
    title="Vision-Assisted Grasping System",
    description="COMP5523 Group Project – PolyU",
    lifespan=lifespan,
)


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    proc_lock = asyncio.Lock()

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # WebSocket disconnect delivered as a message (Starlette low-level API)
            if raw.get("type") == "websocket.disconnect":
                break

            # ── Binary message: JPEG frame from browser camera ───────────────
            if raw.get("bytes"):
                # Drop frame if previous is still processing (back-pressure)
                if proc_lock.locked():
                    continue
                async with proc_lock:
                    state = await loop.run_in_executor(
                        _exec, processor.process_frame, raw["bytes"]
                    )
                guidance = processor.pop_guidance()
                if guidance:
                    await ws.send_json({"type": "guidance", "text": guidance})
                await ws.send_json({"type": "state", "data": state})

            # ── Text message: JSON command from browser ───────────────────────
            elif raw.get("text"):
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "set_target":
                    text = msg.get("target", "").strip()
                    if text:
                        coco = processor.set_target(text)
                        await ws.send_json({
                            "type": "target_confirmed",
                            "display": text,
                            "coco_class": coco,
                        })
                    else:
                        processor.clear_target()
                        await ws.send_json({"type": "target_cleared"})

                elif msg.get("type") == "clear_target":
                    processor.clear_target()
                    await ws.send_json({"type": "target_cleared"})

    except (WebSocketDisconnect, RuntimeError):
        pass


# ─── REST API ─────────────────────────────────────────────────────────────────

class SetTargetBody(BaseModel):
    target: str = ""


@app.post("/api/set_target")
async def api_set_target(body: SetTargetBody):
    target = body.target.strip()
    if not target:
        return JSONResponse({"error": "empty target"}, status_code=400)
    coco = processor.set_target(target)
    return JSONResponse({"coco_class": coco, "display": target})


@app.post("/api/clear_target")
async def api_clear_target():
    processor.clear_target()
    return JSONResponse({"status": "cleared"})


@app.get("/api/state")
async def api_state():
    return JSONResponse(processor.get_state())


# ─── Static Files (must be last) ──────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
