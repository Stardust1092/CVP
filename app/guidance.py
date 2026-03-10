"""
Guidance policy module.
Implements a state machine that converts spatial estimates into
step-by-step movement instructions for visually impaired users.

States:
  IDLE        → waiting for target to be set
  NO_TARGET   → target set but not found in frame
  NO_HAND     → target found but no hand detected
  ALIGNING    → guiding hand left/right/up/down
  APPROACHING → XY aligned, guiding hand forward
  GRASPING    → hand at object, instruct to open/close
  SUCCESS     → grasp completed
"""
import time
from .config import (
    GUIDANCE_COOLDOWN_SEC,
    XY_ALIGN_THRESHOLD,
    CLOSE_DISTANCE_RATIO,
    GUIDANCE_TEXTS,
    CV_TEXTS,
)


class GuidanceState:
    IDLE = "idle"
    NO_TARGET = "no_target"
    NO_HAND = "no_hand"
    ALIGNING = "aligning"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    SUCCESS = "success"


class GuidancePolicy:
    def __init__(self):
        self.state = GuidanceState.IDLE
        self.last_key: str | None = None
        self.last_time: float = 0.0
        self.success_announced: bool = False

    def reset(self):
        self.state = GuidanceState.IDLE
        self.last_key = None
        self.last_time = 0.0
        self.success_announced = False

    # ── public interface ────────────────────────────────────────────────────

    def update(
        self,
        hand_center: tuple[int, int] | None,
        obj_bbox: tuple[int, int, int, int] | None,
        hand_open: bool,
        frame_shape: tuple,
        target_class: str | None,
    ) -> tuple[str | None, str | None]:
        """
        Compute next guidance instruction.

        Returns:
            (zh_text, cv_text) – Chinese string for TTS/UI, English string for
            OpenCV overlay.  Both are None when no instruction should be issued.
        """
        W, H = frame_shape[1], frame_shape[0]

        if target_class is None:
            self.state = GuidanceState.IDLE
            return None, None

        if obj_bbox is None:
            self.state = GuidanceState.NO_TARGET
            return self._emit("no_target")

        if hand_center is None:
            self.state = GuidanceState.NO_HAND
            return self._emit("no_hand")

        hx, hy = hand_center
        x1, y1, x2, y2 = obj_bbox
        obj_cx = (x1 + x2) / 2
        obj_cy = (y1 + y2) / 2
        obj_w = x2 - x1
        obj_h = y2 - y1
        obj_size = max(obj_w, obj_h, 1)

        # Normalised offset (hand → object)
        ndx = (obj_cx - hx) / W   # positive = object is to the right
        ndy = (obj_cy - hy) / H   # positive = object is below

        # Euclidean distance (pixels) from palm center to object center
        dist_px = ((hx - obj_cx) ** 2 + (hy - obj_cy) ** 2) ** 0.5
        close = dist_px < obj_size * CLOSE_DISTANCE_RATIO

        # ── Priority-ordered guidance ──────────────────────────────────────
        # 1. Left/Right alignment
        if abs(ndx) > XY_ALIGN_THRESHOLD:
            self.state = GuidanceState.ALIGNING
            key = "move_right" if ndx > 0 else "move_left"
            return self._emit(key)

        # 2. Up/Down alignment
        if abs(ndy) > XY_ALIGN_THRESHOLD:
            self.state = GuidanceState.ALIGNING
            key = "move_down" if ndy > 0 else "move_up"
            return self._emit(key)

        # 3. Depth (forward) guidance
        if not close:
            self.state = GuidanceState.APPROACHING
            return self._emit("move_forward")

        # 4. Grasp phase
        self.state = GuidanceState.GRASPING
        if hand_open:
            return self._emit("grasp")
        else:
            if not self.success_announced:
                self.success_announced = True
                self.state = GuidanceState.SUCCESS
                return self._emit("success")

        return None, None

    def get_direction_vector(
        self,
        hand_center: tuple[int, int],
        obj_bbox: tuple[int, int, int, int],
        frame_shape: tuple,
    ) -> tuple[float, float] | None:
        """
        Returns unit vector (dx, dy) pointing from hand to object,
        or None when already aligned.
        """
        W, H = frame_shape[1], frame_shape[0]
        hx, hy = hand_center
        x1, y1, x2, y2 = obj_bbox
        obj_cx = (x1 + x2) / 2
        obj_cy = (y1 + y2) / 2

        dx = (obj_cx - hx) / W
        dy = (obj_cy - hy) / H
        mag = (dx ** 2 + dy ** 2) ** 0.5

        if mag < 0.02:
            return None
        return (dx / mag, dy / mag)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _emit(self, key: str) -> tuple[str | None, str | None]:
        """
        Emit instruction for key, respecting cooldown.
        Same instruction repeats after GUIDANCE_COOLDOWN_SEC seconds.
        Different instruction emits immediately.
        """
        now = time.time()
        cooldown_ok = (now - self.last_time) >= GUIDANCE_COOLDOWN_SEC
        new_instruction = key != self.last_key

        if new_instruction or cooldown_ok:
            self.last_key = key
            self.last_time = now
            return GUIDANCE_TEXTS.get(key), CV_TEXTS.get(key)

        return None, None
