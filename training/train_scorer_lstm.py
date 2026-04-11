"""
Stage 4 Training — ResNet-50 + LSTM Highlight Scorer.

Trains a clip-level binary classifier using SoccerNet Action Spotting labels.
Positive clips contain labelled action events; negatives are random non-event
windows sampled from the same video.

Usage
-----
# Full training:
python training/train_scorer_lstm.py

# Quick test (2 epochs, 100 clips):
python training/train_scorer_lstm.py --quick-test

# Custom config:
python training/train_scorer_lstm.py --config config.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from src.scorer import ScorerLSTM
from src.utils import load_config, get_device


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClipDataset(Dataset):
    """
    Loads fixed-length temporal clips from pre-extracted frame directories.

    Expected directory structure (one folder per video clip):

        data/soccernet_actions/
            clips/
                <clip_id>/
                    label.json      ← {"label": 0 or 1}
                    frame_0000.jpg
                    frame_0001.jpg
                    ...
            OR
            annotations.json        ← see _load_from_annotations()

    Clips shorter than clip_length are zero-padded; longer clips are
    centre-cropped.
    """

    FRAME_SIZE = (224, 224)
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: str,
        clip_length: int = 90,
        augment: bool = False,
        max_clips: Optional[int] = None,
    ) -> None:
        self.data_dir    = Path(data_dir)
        self.clip_length = clip_length
        self.augment     = augment
        self.samples: List[Tuple[Path, int]] = []  # (clip_dir, label)

        clips_root = self.data_dir / "clips"
        if clips_root.exists():
            self._load_from_clips_dir(clips_root)
        else:
            print(
                f"[ClipDataset] No clips/ directory found under {data_dir}. "
                "Download SoccerNet Action Spotting data and run "
                "scripts/download_soccernet.py --task actions first."
            )

        if max_clips and len(self.samples) > max_clips:
            rng = random.Random(42)
            self.samples = rng.sample(self.samples, max_clips)

        self._transform = self._make_transform(augment)

    def _load_from_clips_dir(self, clips_root: Path) -> None:
        for clip_dir in sorted(clips_root.iterdir()):
            label_file = clip_dir / "label.json"
            if not label_file.exists():
                continue
            label = json.loads(label_file.read_text()).get("label", 0)
            self.samples.append((clip_dir, int(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_dir, label = self.samples[idx]
        frames = self._load_frames(clip_dir)
        clip_tensor = self._pad_or_crop(frames)
        return clip_tensor, torch.tensor(float(label))

    # ------------------------------------------------------------------

    def _load_frames(self, clip_dir: Path) -> List[torch.Tensor]:
        frame_paths = sorted(clip_dir.glob("frame_*.jpg")) + \
                      sorted(clip_dir.glob("frame_*.png"))
        tensors = []
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(img))
        return tensors

    def _pad_or_crop(self, frames: List[torch.Tensor]) -> torch.Tensor:
        T_target = self.clip_length
        if not frames:
            # Return a zero clip
            return torch.zeros(T_target, 3, *self.FRAME_SIZE)

        stacked = torch.stack(frames)               # (T_actual, C, H, W)
        T_actual = stacked.size(0)

        if T_actual >= T_target:
            start = (T_actual - T_target) // 2
            return stacked[start: start + T_target]

        # Pad with zeros
        pad = torch.zeros(T_target - T_actual, *stacked.shape[1:])
        return torch.cat([stacked, pad], dim=0)

    def _make_transform(self, augment: bool) -> T.Compose:
        if augment:
            return T.Compose([
                T.ToPILImage(),
                T.Resize((self.FRAME_SIZE[0] + 16, self.FRAME_SIZE[1] + 16)),
                T.RandomCrop(self.FRAME_SIZE),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.MEAN, self.STD),
            ])
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.FRAME_SIZE),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for clips, labels in tqdm(loader, desc="  train", leave=False):
        clips  = clips.to(device)      # (B, T, C, H, W)
        labels = labels.to(device)     # (B,)

        optimizer.zero_grad()
        scores = model(clips)           # (B,)
        loss   = criterion(scores, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds       = (scores >= 0.5).float()
        correct    += int((preds == labels).sum())
        total      += clips.size(0)

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct   / total if total else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for clips, labels in tqdm(loader, desc="  val  ", leave=False):
        clips  = clips.to(device)
        labels = labels.to(device)

        scores = model(clips)
        loss   = criterion(scores, labels)

        total_loss += loss.item() * clips.size(0)
        preds       = (scores >= 0.5).float()
        correct    += int((preds == labels).sum())
        total      += clips.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct   / total if total else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    cfg      = load_config(args.config)
    scfg     = cfg["training"]["scorer_lstm"]
    device   = args.device or get_device()
    log_dir  = Path(scfg["log_dir"])
    save_path = Path(scfg["save_path"])

    log_dir.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Quick-test overrides ─────────────────────────────────────────────
    if args.quick_test:
        print("[train_scorer_lstm] QUICK-TEST mode.")
        epochs      = scfg["quick_test_epochs"]
        batch_size  = scfg["quick_test_batch_size"]
        max_clips   = scfg["quick_test_clips"]
        clip_length = scfg.get("quick_test_clip_length", 16)
        # Force CPU on quick-test to avoid MPS/CUDA OOM with large ResNet-50 batches
        device = "cpu"
        print(f"[train_scorer_lstm] Quick-test: clip_length={clip_length}, device=cpu")
    else:
        epochs      = args.epochs or scfg["epochs"]
        batch_size  = scfg["batch_size"]
        max_clips   = None
        clip_length = scfg["clip_length_frames"]
    data_dir    = args.data_dir or scfg["data_dir"]

    # ── Dataset ──────────────────────────────────────────────────────────
    full_ds = ClipDataset(
        data_dir,
        clip_length=clip_length,
        augment=True,
        max_clips=max_clips,
    )

    if len(full_ds) == 0:
        print(
            "[train_scorer_lstm] No clip data found. "
            "Skipping training — run scripts/download_soccernet.py --task actions first."
        )
        return

    val_size   = max(1, int(len(full_ds) * 0.15))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds.dataset.augment = False  # type: ignore

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=scfg["num_workers"] if not args.quick_test else 0,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=scfg["num_workers"] if not args.quick_test else 0,
        pin_memory=(device != "cpu"),
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = ScorerLSTM(
        feature_dim=scfg["feature_dim"],
        hidden_dim=scfg["hidden_dim"],
        num_layers=scfg["num_layers"],
        dropout=scfg["dropout"],
        pretrained_backbone=True,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    writer = SummaryWriter(log_dir=str(log_dir))

    print(
        f"[train_scorer_lstm] Training LSTM scorer | "
        f"device={device} | epochs={epochs} | "
        f"train={train_size} val={val_size} | clip_len={clip_length}"
    )

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_acc)
        elapsed = time.time() - t0

        writer.add_scalars("Loss",     {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc,  "val": vl_acc},  epoch)

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ New best val_acc={best_val_acc:.4f}  saved → {save_path}")

    writer.close()
    print(f"[train_scorer_lstm] Done. Best val accuracy: {best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the ResNet-50 + LSTM highlight scorer.")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--device",   default=None)
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--data-dir", dest="data_dir", default=None)
    p.add_argument("--quick-test", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
