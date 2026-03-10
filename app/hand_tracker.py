"""
Hand tracking module — MediaPipe Tasks API (v0.10+).

Uses GestureRecognizer which provides BOTH:
  - 21 hand landmarks (for position/direction computation)
  - Canned gesture label ("Open_Palm", "Closed_Fist", etc.)

This is more reliable than manual y-coordinate finger analysis,
especially when the hand is held at an angle (e.g., reaching sideways).

Model (~25 MB float16) is auto-downloaded to models/ on first run.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
    drawing_styles,
)

from .config import MAX_HANDS, HAND_DETECT_CONF, HAND_TRACK_CONF

# ─── Model auto-download ──────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
_MODEL_PATH = Path(__file__).parent.parent / "models" / "gesture_recognizer.task"

# Gestures classified as "open hand"
_OPEN_GESTURES = {"Open_Palm", "Victory", "ILoveYou", "Pointing_Up"}
# Gestures classified as "closed / grasping"
_CLOSE_GESTURES = {"Closed_Fist", "Thumb_Down", "Thumb_Up"}


def _ensure_model() -> str:
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        print("[HandTracker] Downloading gesture_recognizer model (~25 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[HandTracker] Model saved to {_MODEL_PATH}")
    return str(_MODEL_PATH)


# ─── HandTracker ──────────────────────────────────────────────────────────────

class HandTracker:
    # Palm center landmarks: wrist + 4 MCP joints
    PALM_INDICES = [0, 5, 9, 13, 17]
    TIP_INDICES  = [4, 8, 12, 16, 20]

    def __init__(self):
        model_path = _ensure_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = GestureRecognizerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=HAND_DETECT_CONF,
            min_hand_presence_confidence=HAND_DETECT_CONF,
            min_tracking_confidence=HAND_TRACK_CONF,
        )
        self._recognizer = GestureRecognizer.create_from_options(options)
        self._t0_ms = int(time.perf_counter() * 1000)
        self._last_gesture: str = "None"

    def track(self, frame: np.ndarray) -> list | None:
        """
        Process one BGR frame.

        Returns list of 21 NormalizedLandmark, or None if no hand detected.
        Side-effect: updates self._last_gesture.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.perf_counter() * 1000) - self._t0_ms
        result = self._recognizer.recognize_for_video(mp_image, ts_ms)

        if result.hand_landmarks:
            # Update gesture label
            if result.gestures:
                self._last_gesture = result.gestures[0][0].category_name
            else:
                self._last_gesture = "None"
            return result.hand_landmarks[0]

        self._last_gesture = "None"
        return None

    # ── Spatial helpers ───────────────────────────────────────────────────────

    def get_palm_center(
        self, landmarks: list, frame_shape: tuple
    ) -> tuple[int, int]:
        """Return (x, y) pixel coords of palm center."""
        H, W = frame_shape[:2]
        xs = [landmarks[i].x for i in self.PALM_INDICES]
        ys = [landmarks[i].y for i in self.PALM_INDICES]
        return (int(np.mean(xs) * W), int(np.mean(ys) * H))

    def get_fingertip_center(
        self, landmarks: list, frame_shape: tuple
    ) -> tuple[int, int]:
        """Return average pixel position of all 5 fingertips."""
        H, W = frame_shape[:2]
        xs = [landmarks[i].x for i in self.TIP_INDICES]
        ys = [landmarks[i].y for i in self.TIP_INDICES]
        return (int(np.mean(xs) * W), int(np.mean(ys) * H))

    def is_hand_open(self, landmarks: list) -> bool:
        """
        True when the gesture is an open-hand pose.
        Uses GestureRecognizer result (much more robust than manual analysis).
        Falls back to fingertip-spread heuristic if gesture is "None".
        """
        if self._last_gesture in _OPEN_GESTURES:
            return True
        if self._last_gesture in _CLOSE_GESTURES:
            return False
        # Fallback: measure fingertip spread as fraction of hand size
        return self._fingertip_spread_open(landmarks)

    def _fingertip_spread_open(self, landmarks: list) -> bool:
        """
        Heuristic: open hand has high spread between index and pinky tips.
        Uses 3D normalised coordinates so it's orientation-independent.
        """
        idx_tip  = landmarks[8]   # index fingertip
        pinky_tip = landmarks[20]  # pinky fingertip
        wrist = landmarks[0]

        # Distance from index to pinky tip
        spread = (
            (idx_tip.x - pinky_tip.x) ** 2 +
            (idx_tip.y - pinky_tip.y) ** 2 +
            (idx_tip.z - pinky_tip.z) ** 2
        ) ** 0.5

        # Distance from wrist to middle fingertip (hand size proxy)
        mid_tip = landmarks[12]
        hand_size = (
            (wrist.x - mid_tip.x) ** 2 +
            (wrist.y - mid_tip.y) ** 2 +
            (wrist.z - mid_tip.z) ** 2
        ) ** 0.5

        # Open if spread > 50% of hand size
        return hand_size > 0 and (spread / hand_size) > 0.50

    def get_gesture_label(self) -> str:
        """Return the last detected gesture label."""
        return self._last_gesture

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_hand(self, frame: np.ndarray, landmarks: list) -> np.ndarray:
        """Draw hand skeleton on frame (in-place)."""
        lm_style   = drawing_styles.get_default_hand_landmarks_style()
        conn_style = drawing_utils.DrawingSpec(color=(200, 200, 200), thickness=2)
        drawing_utils.draw_landmarks(
            frame, landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=lm_style,
            connection_drawing_spec=conn_style,
        )
        return frame
