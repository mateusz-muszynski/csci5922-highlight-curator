"""
Player Number-Guided AI Highlight Video Curator — Full Pipeline Inference.

Usage
-----
python main.py --video match.mp4 --jersey 10
python main.py --video match.mp4 --jersey 10 --output outputs/player10.mp4
python main.py --video match.mp4 --jersey 10 --device cuda --conf 0.35 --debug
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm

from src.detector import PlayerDetector
from src.jersey_reader import JerseyReader
from src.tracker import PlayerTracker, TrackedPlayer
from src.scorer import HighlightScorer
from src.utils import (
    VideoReader,
    VideoWriter,
    draw_tracks,
    get_device,
    load_config,
    Timer,
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    jersey_number: int,
    output_path: Optional[str] = None,
    config_path: str = "config.yaml",
    device: Optional[str] = None,
    conf_override: Optional[float] = None,
    highlight_thresh_override: Optional[float] = None,
    debug_video: Optional[str] = None,
    frame_skip: int = 1,
    clip_length_override: Optional[int] = None,
    clip_stride_override: Optional[int] = None,
    jersey_color: Optional[str] = None,
) -> str:
    """
    Execute the four-stage pipeline end-to-end.

    Parameters
    ----------
    frame_skip : int
        Process every Nth frame (1 = every frame, 3 = every 3rd frame).
        Useful for long videos on CPU — set to 2 or 3 to run ~3x faster.

    Returns
    -------
    Path to the assembled highlight reel.
    """
    cfg    = load_config(config_path)
    device = device or get_device()

    # Resolve all model paths relative to the config file's directory so the
    # pipeline works regardless of the caller's working directory.
    cfg_dir = Path(config_path).resolve().parent
    def _abs(rel: str) -> str:
        p = Path(rel)
        return str(p) if p.is_absolute() else str(cfg_dir / p)

    # ── Build models ─────────────────────────────────────────────────────
    det_cfg = cfg["detector"]
    detector = PlayerDetector(
        model_path=_abs(det_cfg["model_path"]),
        conf_threshold=conf_override or det_cfg["conf_threshold"],
        iou_threshold=det_cfg["iou_threshold"],
        player_class_id=det_cfg["player_class_id"],
        input_size=det_cfg["input_size"],
        device=device,
    )

    jr_cfg = cfg["jersey_reader"]
    # color_hint: CLI/notebook arg takes priority, then config, then None
    color_hint = jersey_color or jr_cfg.get("color_hint") or None
    reader = JerseyReader(
        model_path=_abs(jr_cfg["model_path"]),
        num_classes=jr_cfg["num_classes"],
        input_size=tuple(jr_cfg["input_size"]),
        conf_threshold=jr_cfg["conf_threshold"],
        orientation_enabled=jr_cfg["orientation_filter"]["enabled"],
        facing_threshold=jr_cfg["orientation_filter"]["facing_threshold"],
        color_hint=color_hint,
        device=device,
    )
    if color_hint:
        print(f"[Pipeline] Jersey colour hint: {color_hint} (V threshold={JerseyReader._DARK_BRIGHTNESS_THRESHOLD})")

    tr_cfg = cfg["tracker"]
    tracker = PlayerTracker(
        target_jersey=jersey_number,
        track_activation_threshold=tr_cfg["track_activation_threshold"],
        lost_track_buffer=tr_cfg["lost_track_buffer"],
        minimum_matching_threshold=tr_cfg["minimum_matching_threshold"],
        frame_rate=tr_cfg["frame_rate"],
        jersey_reid_weight=tr_cfg["jersey_reid_weight"],
        jersey_vote_window=tr_cfg["jersey_vote_window"],
        jersey_min_votes=tr_cfg["jersey_min_votes"],
    )

    sc_cfg = cfg["scorer"]
    vc_cfg = cfg["video"]
    scorer = HighlightScorer(
        model_path=_abs(sc_cfg["model_path"]),
        feature_dim=sc_cfg["feature_dim"],
        hidden_dim=sc_cfg["hidden_dim"],
        num_layers=sc_cfg["num_layers"],
        dropout=sc_cfg["dropout"],
        clip_length_frames=clip_length_override or sc_cfg["clip_length_frames"],
        clip_stride_frames=clip_stride_override or sc_cfg["clip_stride_frames"],
        highlight_threshold=highlight_thresh_override or sc_cfg["highlight_threshold"],
        ball_proximity_boost=sc_cfg["ball_proximity_boost"],
        attacking_third_boost=sc_cfg["attacking_third_boost"],
        pre_roll_frames=vc_cfg["pre_roll_frames"],
        post_roll_frames=vc_cfg["post_roll_frames"],
        min_clip_gap_frames=vc_cfg["min_clip_gap_frames"],
        target_jersey=jersey_number,
        device=device,
    )

    # ── Pass 1: Detect → Read → Track ────────────────────────────────────
    if frame_skip > 1:
        print(f"\n[Pipeline] Pass 1 — Detect / Read / Track  (jersey #{jersey_number}, every {frame_skip}th frame)")
    else:
        print(f"\n[Pipeline] Pass 1 — Detect / Read / Track  (jersey #{jersey_number})")

    track_history: Dict[int, List[TrackedPlayer]] = defaultdict(list)
    jersey_counter: Dict[int, int] = defaultdict(int)
    debug_writer: Optional[VideoWriter] = None

    with VideoReader(video_path) as vr:
        total = vr.total_frames
        if debug_video:
            debug_writer = VideoWriter(debug_video, fps=vr.fps, width=vr.width, height=vr.height).__enter__()

        with Timer("Pass 1 (detect+track)") as t1:
            for frame_idx, frame in tqdm(vr, total=total, desc="  frames"):
                # Skip frames for speed on long videos
                if frame_skip > 1 and frame_idx % frame_skip != 0:
                    continue

                # Stage 1
                detections = detector.detect(frame)

                # Stage 2 — read jersey from upper-body crop of each detection
                jersey_reads = []
                for det in detections:
                    crop = detector.crop_upper_body(frame, det)
                    number, conf = reader.read(crop)
                    jersey_reads.append((number, conf))
                    if number is not None:
                        jersey_counter[number] += 1

                # Stage 3
                players = tracker.update(detections, jersey_reads, frame_idx)
                track_history[frame_idx] = players

                # Optional debug overlay
                if debug_writer is not None:
                    vis = draw_tracks(frame, players, target_jersey=jersey_number)
                    debug_writer.write(vis)

    if debug_writer is not None:
        debug_writer.__exit__(None, None, None)

    # Free detector + reader GPU memory before loading the scorer
    del detector, reader
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print jersey number sighting summary
    if jersey_counter:
        top = sorted(jersey_counter.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Jersey sightings (top 10): " +
              "  ".join(f"#{n}={c}" for n, c in top))

    target_frames = sum(
        1
        for fi, players in track_history.items()
        for p in players
        if p.jersey_number == jersey_number
    )
    print(
        f"  Pass 1 complete — jersey #{jersey_number} visible in "
        f"{target_frames} / {total} frames ({t1.elapsed:.1f}s)"
    )

    # ── Pass 2: Score → Assemble ─────────────────────────────────────────
    print("\n[Pipeline] Pass 2 — Score & Assemble highlights")

    if output_path is None:
        vid_stem = Path(video_path).stem
        output_path = str(Path(vc_cfg["output_dir"]) / f"{vid_stem}_jersey{jersey_number}.mp4")

    with Timer("Pass 2 (score+assemble)") as t2:
        clips = scorer.score_video(video_path, track_history)

    print(f"  {len(clips)} highlight clips selected ({t2.elapsed:.1f}s)")
    for c in clips:
        print(f"    frames {c.start_frame}–{c.end_frame}  score={c.score:.3f}")

    if clips:
        result = scorer.assemble(
            video_path,
            clips,
            output_path,
            ffmpeg_crf=vc_cfg.get("ffmpeg_crf", 23),
            ffmpeg_preset=vc_cfg.get("ffmpeg_preset", "fast"),
        )
        print(f"\n[Pipeline] Done!  Highlight reel → {result}")
        return result
    else:
        print(
            f"\n[Pipeline] No highlights found for jersey #{jersey_number}. "
            "Try lowering --highlight-thresh."
        )
        return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a player highlight reel from a match video."
    )
    p.add_argument("--video",   required=True, help="Path to input match video")
    p.add_argument("--jersey",  required=True, type=int, help="Target jersey number (0–99)")
    p.add_argument("--output",  default=None, help="Output video path")
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--device",  default=None, help="cuda / mps / cpu")
    p.add_argument("--conf",    type=float, default=None, dest="conf",
                   help="Override YOLO confidence threshold")
    p.add_argument("--highlight-thresh", type=float, default=None,
                   dest="highlight_thresh",
                   help="Override LSTM highlight score threshold")
    p.add_argument("--debug",   default=None, metavar="PATH",
                   help="Write a debug video with bounding-box overlays")
    p.add_argument("--frame-skip", type=int, default=1, dest="frame_skip",
                   help="Process every Nth frame (default 1=all). Use 3 for ~3x speedup on CPU.")
    p.add_argument("--clip-length", type=int, default=None, dest="clip_length",
                   help="Override LSTM clip length in frames (default from config)")
    p.add_argument("--clip-stride", type=int, default=None, dest="clip_stride",
                   help="Override LSTM clip stride in frames (default from config)")
    p.add_argument("--jersey-color", default=None, dest="jersey_color",
                   choices=["dark", "light"],
                   help="Filter crops by jersey brightness: 'dark' or 'light'")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(
        video_path=args.video,
        jersey_number=args.jersey,
        output_path=args.output,
        config_path=args.config,
        device=args.device,
        conf_override=args.conf,
        highlight_thresh_override=args.highlight_thresh,
        debug_video=args.debug,
        frame_skip=args.frame_skip,
        clip_length_override=args.clip_length,
        clip_stride_override=args.clip_stride,
        jersey_color=args.jersey_color,
    )
