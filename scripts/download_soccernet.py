"""
SoccerNet Dataset Downloader.

Downloads the three SoccerNet subsets needed for this project:
  - Player detection      (Stage 1 training)
  - Jersey number recognition (Stage 2 training)
  - Action spotting       (Stage 4 training)

Registration at https://www.soccer-net.org/ is required.
Set your credentials via environment variables or pass them as CLI args:

    export SOCCERNET_USER="your@email.com"
    export SOCCERNET_PASS="your_password"
    python scripts/download_soccernet.py --task all

Usage
-----
python scripts/download_soccernet.py --task all
python scripts/download_soccernet.py --task detection
python scripts/download_soccernet.py --task jersey
python scripts/download_soccernet.py --task actions
python scripts/download_soccernet.py --task all --data-dir data/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# SoccerNet download wrappers
# ---------------------------------------------------------------------------

def download_detection(data_dir: Path, username: str, password: str) -> None:
    """Download SoccerNet tracking / player-detection dataset."""
    from SoccerNet.Downloader import SoccerNetDownloader

    dest = data_dir / "soccernet_detection"
    dest.mkdir(parents=True, exist_ok=True)

    mySN = SoccerNetDownloader(LocalDirectory=str(dest))
    mySN.username = username
    mySN.password = password

    print("[download] Downloading SoccerNet Tracking dataset (player detection)...")
    mySN.downloadDataTask(task="tracking", split=["train", "valid", "test"],
                          verbose=True)
    print(f"[download] Tracking data saved to {dest}")


def download_jersey(data_dir: Path, username: str, password: str) -> None:
    """Download SoccerNet Jersey Number Recognition dataset."""
    from SoccerNet.Downloader import SoccerNetDownloader

    dest = data_dir / "soccernet_jersey"
    dest.mkdir(parents=True, exist_ok=True)

    mySN = SoccerNetDownloader(LocalDirectory=str(dest))
    mySN.username = username
    mySN.password = password

    print("[download] Downloading SoccerNet Jersey Number Recognition dataset...")
    mySN.downloadDataTask(task="jersey-2023", split=["train", "valid", "test"],
                          verbose=True)
    print(f"[download] Jersey data saved to {dest}")


def download_actions(data_dir: Path, username: str, password: str) -> None:
    """Download SoccerNet Action Spotting dataset (labels only by default)."""
    from SoccerNet.Downloader import SoccerNetDownloader

    dest = data_dir / "soccernet_actions"
    dest.mkdir(parents=True, exist_ok=True)

    mySN = SoccerNetDownloader(LocalDirectory=str(dest))
    mySN.username = username
    mySN.password = password

    print("[download] Downloading SoccerNet Action Spotting labels...")
    # Download labels first (small); add features="ResNET_PCA512" for pre-extracted feats
    mySN.downloadDataTask(task="action-spotting-2023",
                          split=["train", "valid", "test"],
                          verbose=True)
    print(f"[download] Action spotting data saved to {dest}")


# ---------------------------------------------------------------------------
# Synthetic stub creator (for quick-test with no credentials)
# ---------------------------------------------------------------------------

def create_synthetic_stubs(data_dir: Path) -> None:
    """
    Create tiny synthetic stub datasets so that training scripts can run in
    --quick-test mode without real SoccerNet data.

    This lets you verify the training code end-to-end on any machine.
    """
    import numpy as np
    from PIL import Image

    print("[stubs] Creating synthetic stub datasets for quick-test mode...")

    # ── Jersey CNN stubs ─────────────────────────────────────────────────
    jersey_root = data_dir / "soccernet_jersey"
    for jersey_num in range(10):  # create 10 jersey classes
        label_str = f"{jersey_num:02d}"
        label_dir = jersey_root / label_str
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(20):  # 20 images per class
            img = Image.fromarray(
                np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
            )
            img.save(label_dir / f"stub_{i:04d}.jpg")
    print(f"  Jersey stubs: {jersey_root} (10 classes × 20 images)")

    # ── Action spotting (LSTM scorer) stubs ─────────────────────────────
    clips_root = data_dir / "soccernet_actions" / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    for clip_idx in range(20):
        clip_dir = clips_root / f"clip_{clip_idx:04d}"
        clip_dir.mkdir(exist_ok=True)
        label = clip_idx % 2  # alternating pos/neg
        (clip_dir / "label.json").write_text(json.dumps({"label": label}))
        for frame_idx in range(30):  # 30 frames per clip
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(clip_dir / f"frame_{frame_idx:04d}.jpg")
    print(f"  Action spotting stubs: {clips_root} (20 clips × 30 frames)")

    # ── YOLO detection stubs — minimal dataset.yaml only ────────────────
    det_root = data_dir / "soccernet_detection"
    det_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (det_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (det_root / "labels"  / split).mkdir(parents=True, exist_ok=True)
        # Create one tiny image + label so YOLO doesn't crash
        img = Image.fromarray(
            np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        )
        img.save(det_root / "images" / split / "stub_0000.jpg")
        # YOLO label format: class cx cy w h (normalised)
        (det_root / "labels" / split / "stub_0000.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    yaml_content = (
        f"path: {det_root.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: player\n"
    )
    (det_root / "dataset.yaml").write_text(yaml_content)
    print(f"  YOLO stubs: {det_root} (1 image per split)")

    print("[stubs] Done. Run:  python run_training.py --quick-test")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download SoccerNet datasets for all three training stages."
    )
    p.add_argument(
        "--task",
        choices=["all", "detection", "jersey", "actions", "stubs"],
        default="all",
        help=(
            "Which dataset(s) to download. "
            "Use 'stubs' to create tiny synthetic data for quick-test mode "
            "(no credentials required)."
        ),
    )
    p.add_argument("--data-dir",  default="data",  help="Root data directory")
    p.add_argument("--username",  default=None,     help="SoccerNet username (or env SOCCERNET_USER)")
    p.add_argument("--password",  default=None,     help="SoccerNet password (or env SOCCERNET_PASS)")
    return p


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "stubs":
        create_synthetic_stubs(data_dir)
        return

    username = args.username or os.environ.get("SOCCERNET_USER", "")
    password = args.password or os.environ.get("SOCCERNET_PASS", "")

    if not username or not password:
        print(
            "[download] ERROR: SoccerNet credentials required.\n"
            "  Set SOCCERNET_USER and SOCCERNET_PASS environment variables, or\n"
            "  pass --username and --password.\n"
            "  Register at https://www.soccer-net.org/\n\n"
            "  TIP: To run without credentials, use:\n"
            "    python scripts/download_soccernet.py --task stubs"
        )
        sys.exit(1)

    if args.task in ("all", "detection"):
        download_detection(data_dir, username, password)

    if args.task in ("all", "jersey"):
        download_jersey(data_dir, username, password)

    if args.task in ("all", "actions"):
        download_actions(data_dir, username, password)

    print("\n[download] All requested datasets downloaded successfully.")


if __name__ == "__main__":
    main(build_parser().parse_args())
