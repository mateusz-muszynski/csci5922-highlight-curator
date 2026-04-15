"""
Stage 4 — Highlight Scoring & Assembly
ResNet-50 feature extractor + LSTM to produce a per-clip relevance score.
Clips above *highlight_threshold* that contain the target player are assembled
into a final highlight reel using FFmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import cv2

from .tracker import TrackedPlayer


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class ScorerLSTM(nn.Module):
    """
    ResNet-50 (frozen or fine-tuned) feature extractor followed by a
    bidirectional LSTM that outputs a highlight score in [0, 1].

    Input  : (batch, seq_len, C, H, W)  — a temporal clip of frames
    Output : (batch,)                   — scalar score per clip
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.30,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        # --- Backbone (ResNet-50, global-average-pool output = 2048-d) ---
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = tv_models.resnet50(weights=weights)
        # Remove the classification head; keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = feature_dim   # 2048 for ResNet-50

        # --- Temporal LSTM ---
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B,) scores

        Frames are processed through the backbone in mini-chunks of
        BACKBONE_CHUNK_SIZE to cap peak GPU memory regardless of clip length.
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # Process backbone in chunks to avoid OOM on long clips
        CHUNK = 16  # frames per chunk through ResNet-50
        feat_chunks = []
        for start in range(0, B * T, CHUNK):
            chunk = x_flat[start: start + CHUNK]
            feat_chunks.append(self.backbone(chunk))  # (chunk, 2048, 1, 1)
        feats = torch.cat(feat_chunks, dim=0)          # (B*T, 2048, 1, 1)
        feats = feats.view(B, T, self.feature_dim)

        out, _ = self.lstm(feats)          # (B, T, hidden*2)
        score = self.head(out[:, -1, :])   # use last timestep → (B, 1)
        return score.squeeze(1)            # (B,)


# ---------------------------------------------------------------------------
# Highlight clip container
# ---------------------------------------------------------------------------

class HighlightClip:
    """Metadata for one candidate highlight clip."""

    __slots__ = ("start_frame", "end_frame", "score", "contains_target")

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        score: float,
        contains_target: bool = False,
    ) -> None:
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.score = score
        self.contains_target = contains_target

    def __repr__(self) -> str:
        return (
            f"HighlightClip(frames={self.start_frame}–{self.end_frame}, "
            f"score={self.score:.3f}, target={self.contains_target})"
        )


# ---------------------------------------------------------------------------
# Main scorer / assembler
# ---------------------------------------------------------------------------

class HighlightScorer:
    """
    Score sliding-window clips from a video and assemble selected clips.

    Usage
    -----
    scorer = HighlightScorer(model_path="models/scorer_lstm.pt", target_jersey=10)
    clips  = scorer.score_video(video_path, track_history)
    scorer.assemble(video_path, clips, output_path="outputs/highlights.mp4")
    """

    # ImageNet normalisation
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)
    # Resize each frame before feeding the backbone
    _FRAME_SIZE = (224, 224)

    def __init__(
        self,
        model_path: str = "models/scorer_lstm.pt",
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.30,
        clip_length_frames: int = 90,
        clip_stride_frames: int = 15,
        highlight_threshold: float = 0.60,
        ball_proximity_boost: float = 0.10,
        attacking_third_boost: float = 0.15,
        pre_roll_frames: int = 30,
        post_roll_frames: int = 60,
        min_clip_gap_frames: int = 30,
        target_jersey: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.clip_length = clip_length_frames
        self.clip_stride = clip_stride_frames
        self.highlight_threshold = highlight_threshold
        self.ball_proximity_boost = ball_proximity_boost
        self.attacking_third_boost = attacking_third_boost
        self.pre_roll = pre_roll_frames
        self.post_roll = post_roll_frames
        self.min_gap = min_clip_gap_frames
        self.target_jersey = target_jersey
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ScorerLSTM(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_backbone=False,
        )
        weights_path = Path(model_path)
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            print(
                f"[HighlightScorer] WARNING: weights not found at '{model_path}'. "
                "Running with random weights — train first!"
            )
        self.model.to(self.device)
        self.model.eval()

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self._FRAME_SIZE),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_video(
        self,
        video_path: str,
        track_history: Dict[int, List[TrackedPlayer]],
    ) -> List[HighlightClip]:
        """
        Score all sliding-window clips in *video_path*.

        Streams frames directly from disk — never loads the full video into RAM.
        Only windows that contain the target jersey are scored by the LSTM;
        all others are skipped immediately, saving both time and memory.

        Parameters
        ----------
        video_path     : path to the source video
        track_history  : frame_idx → list of TrackedPlayer objects

        Returns
        -------
        List of HighlightClip objects above threshold, sorted by start_frame.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        clips: List[HighlightClip] = []
        windows = list(range(0, total_frames - self.clip_length + 1, self.clip_stride))
        print(f"[Scorer] {total_frames} frames · {len(windows)} windows to evaluate")

        for win_idx, start in enumerate(windows):
            end = start + self.clip_length

            # Skip window immediately if target jersey not present — saves LSTM time
            contains_target = self._clip_contains_target(start, end, track_history)
            if not contains_target:
                continue

            # Stream just this window's frames from disk
            clip_frames = self._read_clip_frames(video_path, start, end)
            if not clip_frames:
                continue

            score = self._score_clip(clip_frames)
            score = self._apply_context_boosts(score, start, end, track_history)

            if score >= self.highlight_threshold:
                clips.append(HighlightClip(start, end, score, contains_target=True))

            if win_idx % 50 == 0:
                print(f"[Scorer]   window {win_idx}/{len(windows)}  clips so far: {len(clips)}")

        merged = self._merge_clips(clips, total_frames)
        return sorted(merged, key=lambda c: c.start_frame)

    def _read_clip_frames(
        self,
        video_path: str,
        start: int,
        end: int,
    ) -> List[np.ndarray]:
        """Seek to *start* and read exactly (end - start) frames. No full-video load."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: List[np.ndarray] = []
        for _ in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def assemble(
        self,
        video_path: str,
        clips: List[HighlightClip],
        output_path: str,
        ffmpeg_crf: int = 23,
        ffmpeg_preset: str = "fast",
    ) -> str:
        """
        Cut and concatenate *clips* from *video_path* into *output_path*.
        Requires FFmpeg to be installed and on PATH.
        """
        if not clips:
            print("[HighlightScorer] No clips to assemble.")
            return ""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        temp_dir = output.parent / "temp_clips"
        temp_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        segment_files: List[str] = []

        for idx, clip in enumerate(clips):
            seg_path = str(temp_dir / f"seg_{idx:04d}.mp4")
            start_sec = max(0.0, (clip.start_frame - self.pre_roll) / fps)
            end_sec = (clip.end_frame + self.post_roll) / fps
            duration = end_sec - start_sec

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_sec:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-crf", str(ffmpeg_crf),
                "-preset", ffmpeg_preset,
                "-an",
                seg_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            segment_files.append(seg_path)

        # Write concat list and merge
        list_file = str(temp_dir / "concat_list.txt")
        with open(list_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            str(output),
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)
        print(f"[HighlightScorer] Highlight reel saved to {output}")
        return str(output)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_clip(self, frames: List[np.ndarray]) -> float:
        """Run the LSTM scorer on a list of BGR frames."""
        tensors = [self._transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        clip_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        with torch.no_grad():
            score = self.model(clip_tensor)
        return float(score.item())

    def _apply_context_boosts(
        self,
        base_score: float,
        start: int,
        end: int,
        track_history: Dict[int, List[TrackedPlayer]],
    ) -> float:
        """Add context-aware score boosts for ball proximity and field position."""
        # Ball proximity boost: if target player has multiple appearances → boost
        target_appearances = sum(
            1
            for fi in range(start, end)
            for p in track_history.get(fi, [])
            if p.jersey_number == self.target_jersey
        )
        if target_appearances > (end - start) * 0.3:
            base_score += self.ball_proximity_boost

        # Attacking-third heuristic: if the target player's average x position
        # is in the right 1/3 of the frame → attacking-third boost.
        xs = [
            (p.bbox[0] + p.bbox[2]) / 2.0
            for fi in range(start, end)
            for p in track_history.get(fi, [])
            if p.jersey_number == self.target_jersey
        ]
        if xs:
            # Normalise to [0, 1] assuming frame width is unknown here;
            # use raw pixel ratio against a typical 1920-wide frame.
            avg_x_norm = np.mean(xs) / 1920.0
            if avg_x_norm > 0.67:
                base_score += self.attacking_third_boost

        return min(base_score, 1.0)

    def _clip_contains_target(
        self,
        start: int,
        end: int,
        track_history: Dict[int, List[TrackedPlayer]],
    ) -> bool:
        """Return True if the target jersey number appears in at least one frame."""
        if self.target_jersey is None:
            return True
        return any(
            p.jersey_number == self.target_jersey
            for fi in range(start, end)
            for p in track_history.get(fi, [])
        )

    def _merge_clips(
        self, clips: List[HighlightClip], total_frames: int
    ) -> List[HighlightClip]:
        """Merge clips that are separated by fewer than min_gap frames."""
        if not clips:
            return []

        clips = sorted(clips, key=lambda c: c.start_frame)
        merged: List[HighlightClip] = [clips[0]]

        for clip in clips[1:]:
            last = merged[-1]
            if clip.start_frame - last.end_frame < self.min_gap:
                merged[-1] = HighlightClip(
                    start_frame=last.start_frame,
                    end_frame=max(last.end_frame, clip.end_frame),
                    score=max(last.score, clip.score),
                    contains_target=last.contains_target or clip.contains_target,
                )
            else:
                merged.append(clip)

        return merged
