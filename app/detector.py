"""
Object detection module using YOLOv8 (Ultralytics).
Detects common household objects in real-time.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from .config import YOLO_MODEL, DETECTION_CONF, DETECTION_IOU


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.conf = DETECTION_CONF
        self.iou = DETECTION_IOU
        self.names: dict[int, str] = self.model.names

    def detect(self, frame: np.ndarray, target_class: str | None = None) -> list[dict]:
        """
        Run YOLO detection on a frame.

        Args:
            frame: BGR numpy array
            target_class: if set, only return detections of this COCO class name

        Returns:
            List of dicts with keys: class, conf, bbox (x1,y1,x2,y2)
            Sorted by confidence descending.
        """
        results = self.model(
            frame, verbose=False, conf=self.conf, iou=self.iou
        )[0]

        detections: list[dict] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if target_class and cls_name.lower() != target_class.lower():
                continue

            detections.append({
                "class": cls_name,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
            })

        detections.sort(key=lambda d: d["conf"], reverse=True)
        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict],
        target_class: str | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes with labels on frame (in-place)."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            is_target = (
                target_class is not None
                and det["class"].lower() == target_class.lower()
            )
            color = (0, 230, 0) if is_target else (160, 160, 160)
            thickness = 3 if is_target else 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{det['class']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

            # Draw center dot for target
            if is_target:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        return frame
