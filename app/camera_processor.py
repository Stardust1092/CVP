"""
Frame processing module.
Processes JPEG frames received from the browser frontend:
  1. Decodes the JPEG frame
  2. Detects target objects with YOLO
  3. Tracks the user's hand with MediaPipe
  4. Generates guidance instructions
  5. Returns detection data with normalized coordinates for frontend overlay

No camera capture here — the browser provides frames via WebSocket.
"""
import queue
import threading

import cv2
import numpy as np

from .detector import ObjectDetector
from .hand_tracker import HandTracker
from .guidance import GuidancePolicy, GuidanceState
from .config import ZH_TO_COCO


class FrameProcessor:
    def __init__(self):
        self._detector = ObjectDetector()
        self._hand_tracker = HandTracker()
        self._guidance = GuidancePolicy()

        self._guidance_queue: queue.Queue[str] = queue.Queue()

        self._target_lock = threading.Lock()
        self._target_class: str | None = None
        self._target_display: str = ""

        self._last_state: dict = self._empty_state()

    # ── Public API ───────────────────────────────────────────────────────────

    def set_target(self, text: str) -> str:
        text = text.strip()
        coco_class = ZH_TO_COCO.get(text) or text.lower()
        with self._target_lock:
            self._target_class = coco_class
            self._target_display = text
        self._guidance.reset()
        return coco_class

    def clear_target(self):
        with self._target_lock:
            self._target_class = None
            self._target_display = ""
        self._guidance.reset()

    def process_frame(self, jpeg_bytes: bytes) -> dict:
        """
        Decode and process one JPEG frame from the browser.
        Returns a state dict with normalized detection coordinates (0–1).
        This method is thread-safe and called from a single-thread executor.
        """
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return self._last_state

        H, W = frame.shape[:2]

        with self._target_lock:
            target_class = self._target_class
            target_display = self._target_display

        # ── Object detection ─────────────────────────────────────────────────
        target_bbox: tuple | None = None
        target_bbox_norm: list | None = None
        if target_class:
            detections = self._detector.detect(frame, target_class)
            if detections:
                x1, y1, x2, y2 = detections[0]["bbox"]
                target_bbox = (x1, y1, x2, y2)
                target_bbox_norm = [x1 / W, y1 / H, x2 / W, y2 / H]

        # ── Hand tracking ─────────────────────────────────────────────────────
        hand_lm = self._hand_tracker.track(frame)
        hand_center: tuple | None = None
        hand_center_norm: list | None = None
        hand_open = False
        hand_landmarks_norm: list | None = None
        if hand_lm:
            hc = self._hand_tracker.get_palm_center(hand_lm, frame.shape)
            hand_center = hc
            hand_center_norm = [hc[0] / W, hc[1] / H]
            hand_open = self._hand_tracker.is_hand_open(hand_lm)
            hand_landmarks_norm = [[lm.x, lm.y] for lm in hand_lm]

        # ── Guidance ──────────────────────────────────────────────────────────
        zh_text, _ = self._guidance.update(
            hand_center, target_bbox, hand_open, frame.shape, target_class
        )
        if zh_text:
            self._guidance_queue.put(zh_text)

        # ── Direction vector ──────────────────────────────────────────────────
        direction_norm: list | None = None
        if hand_center and target_bbox:
            dv = self._guidance.get_direction_vector(
                hand_center, target_bbox, frame.shape
            )
            if dv:
                direction_norm = list(dv)

        state = {
            "hand_detected": hand_center is not None,
            "target_detected": target_bbox is not None,
            "hand_open": hand_open,
            "guidance_state": self._guidance.state,
            "last_guidance": zh_text or "",
            "target_class": target_class,
            "target_display": target_display,
            "target_bbox_norm": target_bbox_norm,
            "hand_center_norm": hand_center_norm,
            "hand_landmarks_norm": hand_landmarks_norm,
            "direction_norm": direction_norm,
        }
        self._last_state = state
        return state

    def pop_guidance(self) -> str | None:
        try:
            return self._guidance_queue.get_nowait()
        except queue.Empty:
            return None

    def get_state(self) -> dict:
        return self._last_state.copy()

    def _empty_state(self) -> dict:
        return {
            "hand_detected": False,
            "target_detected": False,
            "hand_open": False,
            "guidance_state": GuidanceState.IDLE,
            "last_guidance": "",
            "target_class": None,
            "target_display": "",
            "target_bbox_norm": None,
            "hand_center_norm": None,
            "hand_landmarks_norm": None,
            "direction_norm": None,
        }
