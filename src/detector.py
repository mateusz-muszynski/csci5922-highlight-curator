"""
Stage 1 — Player Detection
Uses YOLOv8m to detect every player bounding box in each frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


# Bounding box type alias: (x1, y1, x2, y2, confidence, class_id)
Detection = Tuple[float, float, float, float, float, int]


class PlayerDetector:
    """
    Wraps YOLOv8m for per-frame player detection.

    Usage
    -----
    detector = PlayerDetector(model_path="models/yolov8m_players.pt")
    detections = detector.detect(frame)          # List[Detection]
    crop = detector.crop_player(frame, detections[0])
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        player_class_id: int = 0,
        input_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.player_class_id = player_class_id
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load weights — falls back to ultralytics hub download if path missing
        model_file = Path(model_path)
        weight_source = str(model_file) if model_file.exists() else "yolov8m.pt"
        self.model = YOLO(weight_source)
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (as returned by cv2.VideoCapture).

        Returns
        -------
        List of (x1, y1, x2, y2, confidence, class_id) tuples,
        one entry per detected player bounding box.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.player_class_id],
            imgsz=self.input_size,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())
                detections.append((x1, y1, x2, y2, conf, cls))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a list of frames, returning one list per frame."""
        return [self.detect(f) for f in frames]

    def crop_player(
        self,
        frame: np.ndarray,
        detection: Detection,
        pad_factor: float = 0.05,
    ) -> np.ndarray:
        """
        Crop the player region from *frame* using the bounding box in *detection*.
        Optional *pad_factor* adds a small margin around the box.
        """
        x1, y1, x2, y2 = detection[:4]
        h, w = frame.shape[:2]

        pad_w = (x2 - x1) * pad_factor
        pad_h = (y2 - y1) * pad_factor

        cx1 = max(0, int(x1 - pad_w))
        cy1 = max(0, int(y1 - pad_h))
        cx2 = min(w, int(x2 + pad_w))
        cy2 = min(h, int(y2 + pad_h))

        return frame[cy1:cy2, cx1:cx2]

    def crop_upper_body(
        self,
        frame: np.ndarray,
        detection: Detection,
        upper_fraction: float = 0.55,
    ) -> np.ndarray:
        """
        Return only the upper portion of a player crop (where the jersey number lives).
        *upper_fraction* controls what fraction of the box height to keep.
        """
        x1, y1, x2, y2 = detection[:4]
        h, w = frame.shape[:2]

        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(w, int(x2))
        cy2 = min(h, int(y1 + (y2 - y1) * upper_fraction))

        return frame[cy1:cy2, cx1:cx2]
