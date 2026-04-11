"""
Shared utilities — config loading, device selection, video I/O, visualisation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load the YAML config and return as a plain dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_nested(cfg: Dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value: get_nested(cfg, 'detector', 'conf_threshold')."""
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(override: Optional[str] = None) -> str:
    """
    Return the best available device string.
    Priority: override > CUDA > MPS (Apple Silicon) > CPU.
    """
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

class VideoReader:
    """
    Context-manager wrapper around cv2.VideoCapture.

    Usage
    -----
    with VideoReader("match.mp4") as reader:
        for frame_idx, frame in reader:
            process(frame)
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None

    # Properties filled after open
    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._cap:
            self._cap.release()

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1


class VideoWriter:
    """
    Context-manager wrapper around cv2.VideoWriter.

    Usage
    -----
    with VideoWriter("out.mp4", fps=30, width=1920, height=1080) as writer:
        for frame in frames:
            writer.write(frame)
    """

    def __init__(
        self,
        path: str,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        fourcc: str = "mp4v",
    ) -> None:
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        code = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            self.path, code, self.fps, (self.width, self.height)
        )
        return self

    def __exit__(self, *_: Any) -> None:
        if self._writer:
            self._writer.release()

    def write(self, frame: np.ndarray) -> None:
        if self._writer:
            self._writer.write(frame)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    (255, 56,  56),  (255, 157,  56), (255, 255,  56),
    ( 56, 255,  56), ( 56, 255, 255), ( 56, 157, 255),
    (157,  56, 255), (255,  56, 255),
]


def draw_tracks(
    frame: np.ndarray,
    players: List,
    target_jersey: Optional[int] = None,
) -> np.ndarray:
    """
    Overlay bounding boxes and track labels on *frame*.
    Target player is highlighted in a different colour.
    """
    vis = frame.copy()
    for player in players:
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        is_target = (
            target_jersey is not None and player.jersey_number == target_jersey
        )
        color = (0, 255, 0) if is_target else _PALETTE[player.track_id % len(_PALETTE)]
        thickness = 3 if is_target else 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label = f"#{player.jersey_number}" if player.jersey_number is not None else f"T{player.track_id}"
        cv2.putText(
            vis, label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            cv2.LINE_AA,
        )

    return vis


# ---------------------------------------------------------------------------
# Metrics helpers (used in Experiment 2)
# ---------------------------------------------------------------------------

def compute_clip_f1(
    predicted_frames: List[Tuple[int, int]],
    ground_truth_frames: List[Tuple[int, int]],
    tolerance_frames: int = 90,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 at the clip level with a *tolerance_frames*
    window (default = 3 s at 30 fps as per Experiment 2 spec).

    Parameters
    ----------
    predicted_frames   : list of (start, end) tuples for predicted clips
    ground_truth_frames: list of (start, end) tuples for ground-truth clips

    Returns
    -------
    (precision, recall, f1)
    """
    def overlaps(a: Tuple[int, int], b: Tuple[int, int], tol: int) -> bool:
        return a[0] <= b[1] + tol and b[0] <= a[1] + tol

    tp = sum(
        1 for p in predicted_frames
        if any(overlaps(p, g, tolerance_frames) for g in ground_truth_frames)
    )

    precision = tp / len(predicted_frames) if predicted_frames else 0.0
    recall    = tp / len(ground_truth_frames) if ground_truth_frames else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for profiling pipeline stages."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")
