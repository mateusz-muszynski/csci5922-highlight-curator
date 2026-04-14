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
    """
    Download SoccerNet Jersey Number Recognition dataset and normalise
    it to the folder structure expected by JerseyDataset:

        data/soccernet_jersey/
            <jersey_number>/   ← two-digit string, e.g. "06", "23"
                img_001.jpg
                ...

    SoccerNet jersey-2023 ships its images already in per-class folders,
    but the top-level layout after extraction can vary.  This function
    locates the actual class directories regardless of nesting depth.
    """
    import shutil
    from SoccerNet.Downloader import SoccerNetDownloader

    dest = data_dir / "soccernet_jersey"
    dest.mkdir(parents=True, exist_ok=True)

    mySN = SoccerNetDownloader(LocalDirectory=str(dest))
    mySN.username = username
    mySN.password = password

    print("[download] Downloading SoccerNet Jersey Number Recognition dataset...")
    # Download train split only — valid/test add GB without improving training
    mySN.downloadDataTask(task="jersey-2023", split=["train"], verbose=True)
    print(f"[download] Raw download complete → {dest}")

    # ── Normalise folder structure ────────────────────────────────────────
    # Walk the download tree; any directory whose name parses as 00–99
    # and contains images is a class dir.  Move images up to dest/<label>/.
    moved = 0
    for root, dirs, files in os.walk(dest):
        root_path = Path(root)
        if root_path == dest:
            continue
        try:
            label = int(root_path.name)
        except ValueError:
            continue
        if not (0 <= label <= 99):
            continue
        imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not imgs:
            continue
        target = dest / f"{label:02d}"
        target.mkdir(exist_ok=True)
        for img in imgs:
            src = root_path / img
            dst = target / img
            if not dst.exists():
                shutil.move(str(src), str(dst))
                moved += 1

    total = sum(
        len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        for p in dest.iterdir() if p.is_dir()
    )
    print(f"[download] Jersey data normalised → {dest}  ({total:,} images, {moved} moved)")


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

# Candidate system font paths searched in order (Kaggle/Ubuntu first, macOS fallback)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
]


def _load_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Return the first available TrueType font at *size*, else PIL default."""
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _render_jersey_crop(
    number: int,
    width: int = 64,
    height: int = 128,
    dark_bg: bool = True,
    rng: "np.random.Generator | None" = None,
) -> "Image.Image":
    """
    Render one synthetic jersey-number crop with heavy photometric and
    geometric augmentation so the CNN sees realistic training samples.

    Augmentations applied per image:
    - Varied jersey colours (team-kit palettes: navy, red, green, black,
      white, yellow, grey)
    - Font-size jitter ±20 %
    - Random translation within the torso band
    - Rotation ±15°
    - Perspective warp (simulates off-axis camera angle)
    - Gaussian blur (camera softness / motion blur)
    - Pixel noise
    - Brightness / contrast jitter
    - Optional occlusion patch (simulates arm in front of number)
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

    if rng is None:
        rng = np.random.default_rng()

    # ── Jersey background: realistic kit colours ──────────────────────────
    DARK_PALETTES = [
        (10, 20, 80),    # navy blue
        (120, 20, 20),   # red / maroon
        (10, 80, 20),    # dark green
        (20, 20, 20),    # black
        (80, 40, 10),    # dark brown / bronze
        (60, 0, 90),     # purple
    ]
    LIGHT_PALETTES = [
        (240, 240, 240), # white
        (240, 220, 30),  # yellow
        (200, 230, 255), # light blue
        (240, 200, 200), # light red / pink
        (180, 240, 180), # light green
    ]

    palette = DARK_PALETTES if dark_bg else LIGHT_PALETTES
    base_color = palette[int(rng.integers(len(palette)))]
    # Add per-image colour jitter
    bg_color = tuple(
        int(np.clip(c + int(rng.integers(-20, 21)), 0, 255)) for c in base_color
    )

    # ── Number (foreground) colour: high contrast with background ─────────
    if dark_bg:
        fg_base = (240, 240, 240)
        # Occasionally use gold/yellow numbers (common on dark kits)
        if rng.random() < 0.2:
            fg_base = (255, 215, 0)
    else:
        fg_base = (20, 20, 20)
        if rng.random() < 0.15:
            fg_base = (180, 0, 0)  # red numbers on white kits
    fg_color = tuple(
        int(np.clip(c + int(rng.integers(-15, 16)), 0, 255)) for c in fg_base
    )

    # ── Render on a larger canvas then crop, to allow rotation headroom ───
    canvas_w, canvas_h = width * 2, height * 2
    img = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    text = f"{number:02d}"
    # Font size: base ~30 % of height with ±20 % jitter
    base_font_size = max(18, int(height * 0.30))
    font_size = int(base_font_size * rng.uniform(0.80, 1.25))
    font = _load_font(font_size)

    # Centre text in canvas with translation jitter
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = (canvas_w - tw) // 2 + int(rng.integers(-8, 9))
    cy = int(canvas_h * 0.40) + int(rng.integers(-10, 11))
    draw.text((cx, cy), text, fill=fg_color, font=font)

    # ── Geometric augmentation ────────────────────────────────────────────
    # Rotation ±15°
    angle = float(rng.uniform(-15, 15))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg_color)

    # Perspective warp (simulate camera tilt): only with TrueType font renders
    if rng.random() < 0.5:
        w, h = img.size
        d = int(rng.integers(0, max(1, w // 8)))
        side = int(rng.integers(0, 2))  # 0 = left tilt, 1 = right tilt
        if side == 0:
            coeffs = (0, d, 0, 0, w, h - d, w, h)
        else:
            coeffs = (0, 0, 0, d, w, h, w, h - d)
        # PIL perspective uses an 8-coefficient flat transform
        img = img.transform(
            img.size, Image.PERSPECTIVE,
            _perspective_coeffs(
                [(0, 0), (w, 0), (w, h), (0, h)],
                [(d if side else 0, 0),
                 (w - (0 if side else d), 0),
                 (w - (d if side else 0), h),
                 (0 if side else d, h)],
            ),
            resample=Image.BILINEAR,
            fillcolor=bg_color,
        )

    # Crop back to target size from the centre of the canvas
    left = (canvas_w - width) // 2
    top  = (canvas_h - height) // 2
    img  = img.crop((left, top, left + width, top + height))

    # ── Photometric augmentation ──────────────────────────────────────────
    # Gaussian blur
    if rng.random() < 0.6:
        radius = float(rng.uniform(0.3, 1.8))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Brightness / contrast jitter
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.7, 1.3)))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.7, 1.4)))

    # Pixel noise
    arr = np.array(img, dtype=np.int16)
    noise = rng.integers(-20, 21, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Occlusion: small dark rectangle simulating an arm/shadow (10 % chance)
    if rng.random() < 0.10:
        draw2 = ImageDraw.Draw(img)
        ox = int(rng.integers(0, width - width // 4))
        oy = int(rng.integers(0, height - height // 4))
        ow = int(rng.integers(width // 6, width // 3))
        oh = int(rng.integers(height // 8, height // 4))
        draw2.rectangle([ox, oy, ox + ow, oy + oh], fill=(0, 0, 0))

    return img


def _perspective_coeffs(
    src: list,
    dst: list,
) -> tuple:
    """
    Compute the 8 coefficients for PIL's PERSPECTIVE transform.
    src / dst are lists of 4 (x, y) corner points (top-left, top-right,
    bottom-right, bottom-left).
    """
    import numpy as np

    matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array([X for (_, _), (X, _) in zip(src, dst)] +
                 [Y for (_, _), (_, Y) in zip(src, dst)], dtype=np.float64)
    # Interleave X / Y correctly
    B = []
    for (_, _), (X, Y) in zip(src, dst):
        B.extend([X, Y])
    B = np.array(B, dtype=np.float64)
    res = np.linalg.solve(A, B)
    return tuple(np.array(res).flatten())


def create_synthetic_stubs(
    data_dir: Path,
    images_per_class: int = 500,
    num_classes: int = 100,
) -> None:
    """
    Create synthetic stub datasets so training scripts can run without real
    SoccerNet data.

    Jersey CNN stubs use PIL-rendered digit images with heavy augmentation
    (colour, geometry, blur, noise) so the CNN learns real digit patterns.

    Parameters
    ----------
    images_per_class : int
        Renders per jersey class. 500 is the default (Kaggle GPU mode).
        Pass 100 for quick-test (< 5 min on GPU).
    num_classes : int
        Jersey classes to cover (0 … num_classes-1). Max 100.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    num_classes = min(num_classes, 100)

    print("[stubs] Creating synthetic stub datasets ...")
    print(f"  Jersey CNN: {num_classes} classes × {images_per_class} images each "
          f"(total {num_classes * images_per_class:,})")

    # ── Jersey CNN stubs ─────────────────────────────────────────────────
    jersey_root = data_dir / "soccernet_jersey"
    for jersey_num in range(num_classes):
        label_str = f"{jersey_num:02d}"
        label_dir = jersey_root / label_str
        label_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(label_dir.glob("stub_*.jpg")))
        if existing >= images_per_class:
            continue  # idempotent
        for i in range(existing, images_per_class):
            dark = bool(rng.integers(0, 2))   # random dark / light split
            img = _render_jersey_crop(jersey_num, dark_bg=dark, rng=rng)
            img.save(label_dir / f"stub_{i:04d}.jpg")
    total = sum(len(list((jersey_root / f"{n:02d}").glob("stub_*.jpg")))
                for n in range(num_classes) if (jersey_root / f"{n:02d}").exists())
    print(f"  Jersey stubs saved → {jersey_root}  ({total:,} images)")

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
