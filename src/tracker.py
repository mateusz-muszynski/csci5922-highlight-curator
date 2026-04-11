"""
Stage 3 — Player Tracking
ByteTrack (via supervision) augmented with a jersey-number identity anchor.

Key idea
--------
ByteTrack assigns numeric track IDs based on appearance / IoU. We augment this
with a *jersey lookup table* that maps track_id → jersey_number. When a tracklet
re-enters after occlusion the jersey number read from earlier frames acts as a
hard identity constraint that can override the default re-ID.
"""

from __future__ import annotations

from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class TrackedPlayer:
    """Snapshot of one player in one frame."""

    __slots__ = ("track_id", "bbox", "jersey_number", "jersey_conf", "frame_idx")

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        jersey_number: Optional[int] = None,
        jersey_conf: float = 0.0,
        frame_idx: int = 0,
    ) -> None:
        self.track_id = track_id
        self.bbox = bbox                    # (x1, y1, x2, y2)
        self.jersey_number = jersey_number
        self.jersey_conf = jersey_conf
        self.frame_idx = frame_idx

    def __repr__(self) -> str:
        return (
            f"TrackedPlayer(id={self.track_id}, "
            f"jersey={self.jersey_number}, "
            f"conf={self.jersey_conf:.2f})"
        )


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class PlayerTracker:
    """
    Wraps ``supervision.ByteTrack`` and maintains a jersey-number lookup table
    that persists across frames.

    Usage
    -----
    tracker = PlayerTracker(target_jersey=10)
    for frame_idx, (frame, detections) in enumerate(stream):
        players = tracker.update(detections, jersey_reads, frame_idx)
        target_players = [p for p in players if p.jersey_number == tracker.target_jersey]
    """

    def __init__(
        self,
        target_jersey: Optional[int] = None,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.80,
        frame_rate: int = 30,
        jersey_reid_weight: float = 0.90,
        jersey_vote_window: int = 10,
        jersey_min_votes: int = 3,
    ) -> None:
        self.target_jersey = target_jersey
        self.jersey_reid_weight = jersey_reid_weight
        self.jersey_vote_window = jersey_vote_window
        self.jersey_min_votes = jersey_min_votes

        self._byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

        # track_id → confirmed jersey number (anchored after enough votes)
        self._jersey_anchor: Dict[int, int] = {}
        # track_id → sliding window of recent (jersey_number, conf) readings
        self._jersey_votes: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # jersey_number → track_id  (for re-ID after occlusion)
        self._jersey_to_track: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        raw_detections: List[Tuple],   # List[(x1,y1,x2,y2,conf,cls)]
        jersey_reads: List[Tuple[Optional[int], float]],  # parallel to raw_detections
        frame_idx: int = 0,
    ) -> List[TrackedPlayer]:
        """
        Feed one frame's detections into ByteTrack and apply jersey anchoring.

        Parameters
        ----------
        raw_detections : list of (x1, y1, x2, y2, conf, cls)
        jersey_reads   : parallel list of (jersey_number_or_None, confidence)
        frame_idx      : current frame index (for bookkeeping)

        Returns
        -------
        List of TrackedPlayer objects for the current frame.
        """
        if not raw_detections:
            return []

        sv_detections = self._to_sv_detections(raw_detections)
        tracked = self._byte_tracker.update_with_detections(sv_detections)

        players: List[TrackedPlayer] = []
        for i, (xyxy, track_id) in enumerate(
            zip(tracked.xyxy, tracked.tracker_id)
        ):
            # -----------------------------------------------------------
            # Map tracked detection back to the closest raw detection
            # so we can retrieve the jersey read for this track.
            # -----------------------------------------------------------
            raw_idx = self._match_tracked_to_raw(xyxy, raw_detections)
            number, conf = jersey_reads[raw_idx] if raw_idx is not None else (None, 0.0)

            # Accumulate votes for this track
            if number is not None and conf > 0.0:
                self._accumulate_vote(track_id, number, conf)

            # Attempt jersey re-ID if anchor is already set
            resolved_number = self._resolve_jersey(track_id, number, conf)

            players.append(
                TrackedPlayer(
                    track_id=int(track_id),
                    bbox=(float(xyxy[0]), float(xyxy[1]),
                          float(xyxy[2]), float(xyxy[3])),
                    jersey_number=resolved_number,
                    jersey_conf=conf if resolved_number == number else 1.0,
                    frame_idx=frame_idx,
                )
            )

        return players

    def reset(self) -> None:
        """Clear all tracking state (e.g. between videos)."""
        self._byte_tracker.reset()
        self._jersey_anchor.clear()
        self._jersey_votes.clear()
        self._jersey_to_track.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_sv_detections(
        self, raw: List[Tuple]
    ) -> sv.Detections:
        """Convert raw detection list to a supervision.Detections object."""
        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in raw], dtype=np.float32)
        conf = np.array([d[4] for d in raw], dtype=np.float32)
        cls  = np.array([d[5] for d in raw], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)

    def _accumulate_vote(
        self, track_id: int, number: int, conf: float
    ) -> None:
        """Add a jersey read to the sliding vote window; anchor if quorum reached."""
        votes = self._jersey_votes[track_id]
        votes.append((number, conf))

        # Keep only the most recent *jersey_vote_window* reads
        if len(votes) > self.jersey_vote_window:
            votes.pop(0)

        # Check for quorum
        if track_id in self._jersey_anchor:
            return  # already anchored

        numbers = [n for n, _ in votes]
        most_common, count = Counter(numbers).most_common(1)[0]
        if count >= self.jersey_min_votes:
            self._jersey_anchor[track_id] = most_common
            self._jersey_to_track[most_common] = track_id

    def _resolve_jersey(
        self,
        track_id: int,
        raw_number: Optional[int],
        raw_conf: float,
    ) -> Optional[int]:
        """
        Return the best-known jersey number for *track_id*.

        Priority order:
        1. Anchored jersey number (high confidence after N votes).
        2. Raw CNN read if confidence exceeds reid_weight threshold.
        3. None.
        """
        if track_id in self._jersey_anchor:
            return self._jersey_anchor[track_id]
        if raw_number is not None and raw_conf >= self.jersey_reid_weight:
            return raw_number
        return None

    @staticmethod
    def _match_tracked_to_raw(
        tracked_xyxy: np.ndarray,
        raw_detections: List[Tuple],
    ) -> Optional[int]:
        """
        Find the raw detection index whose box has the highest IoU with
        *tracked_xyxy*. Returns None if raw_detections is empty.
        """
        if not raw_detections:
            return None

        best_idx, best_iou = 0, -1.0
        tx1, ty1, tx2, ty2 = tracked_xyxy

        for i, det in enumerate(raw_detections):
            rx1, ry1, rx2, ry2 = det[:4]
            ix1 = max(tx1, rx1)
            iy1 = max(ty1, ry1)
            ix2 = min(tx2, rx2)
            iy2 = min(ty2, ry2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = (
                (tx2 - tx1) * (ty2 - ty1)
                + (rx2 - rx1) * (ry2 - ry1)
                - inter
            )
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_idx = iou, i

        return best_idx
