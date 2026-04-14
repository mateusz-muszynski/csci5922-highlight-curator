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

def _render_jersey_crop(
    number: int,
    width: int = 64,
    height: int = 128,
    dark_bg: bool = True,
    rng: "np.random.Generator | None" = None,
) -> "Image.Image":
    """
    Render a synthetic jersey-number crop using PIL.

    The image mimics a player torso crop: a solid-colour jersey background
    with a two-digit number drawn in a contrasting colour, plus small amounts
    of noise, blur, and random rotation to create variety.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    if rng is None:
        rng = np.random.default_rng()

    # Jersey background colour
    if dark_bg:
        r = int(rng.integers(0, 70))
        g = int(rng.integers(0, 70))
        b = int(rng.integers(30, 120))
    else:
        r = int(rng.integers(180, 255))
        g = int(rng.integers(180, 255))
        b = int(rng.integers(180, 255))
    bg_color = (r, g, b)

    # Number colour (contrasting)
    if dark_bg:
        fg_color = (
            int(rng.integers(200, 255)),
            int(rng.integers(200, 255)),
            int(rng.integers(200, 255)),
        )
    else:
        fg_color = (
            int(rng.integers(0, 60)),
            int(rng.integers(0, 60)),
            int(rng.integers(0, 60)),
        )

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw the two-digit number centred in the torso band (y 30–80% of height)
    text = f"{number:02d}"
    font_size = max(20, int(height * 0.30) + int(rng.integers(-4, 5)))
    # PIL's built-in bitmap font is always available; use it for portability
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Get text bounding box for centring
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2 + int(rng.integers(-4, 5))
    y = int(height * 0.35) + int(rng.integers(-6, 7))
    draw.text((x, y), text, fill=fg_color, font=font)

    # Light Gaussian blur to simulate camera softness
    if rng.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))

    # Add pixel noise
    arr = np.array(img, dtype=np.int16)
    noise = rng.integers(-15, 16, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


def create_synthetic_stubs(
    data_dir: Path,
    images_per_class: int = 50,
    num_classes: int = 100,
) -> None:
    """
    Create synthetic stub datasets so training scripts can run without real
    SoccerNet data.

    Jersey CNN stubs use PIL-rendered digit images (actual numbers drawn on
    coloured backgrounds) so the CNN learns real number patterns rather than
    random noise.  All 100 classes (00–99) are covered.

    Parameters
    ----------
    images_per_class : int
        Number of synthetic crops to generate per jersey number.
        Default 50 gives a reasonable balance between training time and quality.
    num_classes : int
        Number of jersey classes to generate (up to 100).
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)

    print("[stubs] Creating synthetic stub datasets ...")
    print(f"  Jersey CNN: {num_classes} classes × {images_per_class} images each")

    # ── Jersey CNN stubs ─────────────────────────────────────────────────
    jersey_root = data_dir / "soccernet_jersey"
    for jersey_num in range(num_classes):
        label_str = f"{jersey_num:02d}"
        label_dir = jersey_root / label_str
        label_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(label_dir.glob("stub_*.jpg")))
        if existing >= images_per_class:
            continue  # idempotent — skip if already generated
        for i in range(existing, images_per_class):
            # Alternate dark/light backgrounds for variety
            dark = (i % 2 == 0)
            img = _render_jersey_crop(jersey_num, dark_bg=dark, rng=rng)
            img.save(label_dir / f"stub_{i:04d}.jpg")
    print(f"  Jersey stubs saved → {jersey_root}")

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
    p.add_argument("--images-per-class", type=int, default=50, dest="images_per_class",
                   help="Synthetic images per jersey class (default 50, use 20 for quick-test)")
    p.add_argument("--num-classes", type=int, default=100, dest="num_classes",
                   help="Number of jersey classes to generate 0–N (default 100)")
    return p


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "stubs":
        create_synthetic_stubs(
            data_dir,
            images_per_class=args.images_per_class,
            num_classes=args.num_classes,
        )
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
