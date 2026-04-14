"""
Stage 2 Training — Two-digit Jersey Number CNN.

Trains a ResNet-18 backbone to classify player crops into 100 digit-pair classes
(00–99) using the SoccerNet Jersey Number Recognition dataset.

Usage
-----
# Full training:
python training/train_jersey_cnn.py

# Quick test (2 epochs, 500 samples):
python training/train_jersey_cnn.py --quick-test

# Custom config:
python training/train_jersey_cnn.py --config config.yaml --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import torchvision.models as tv_models
from PIL import Image
from tqdm import tqdm

from src.jersey_reader import JerseyCNN
from src.utils import load_config, get_device


def _generate_stubs(data_dir: str, num_classes: int, images_per_class: int) -> None:
    """Generate PIL-rendered jersey stubs if not already present."""
    from pathlib import Path
    from scripts.download_soccernet import create_synthetic_stubs
    create_synthetic_stubs(
        Path(data_dir),
        images_per_class=images_per_class,
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JerseyDataset(Dataset):
    """
    Expects the SoccerNet jersey-number dataset laid out as:

        data/soccernet_jersey/
            <jersey_number>/          ← e.g. "07", "10", "23"
                <crop_image>.jpg
                ...

    The folder name is the two-digit label.  Folders with names outside
    "00"–"99" are silently skipped.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[T.Compose] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for label_dir in sorted(self.root.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label = int(label_dir.name)
                if not (0 <= label <= 99):
                    continue
            except ValueError:
                continue

            for img_path in label_dir.glob("*.jpg"):
                self.samples.append((img_path, label))
            for img_path in label_dir.glob("*.png"):
                self.samples.append((img_path, label))

        if not self.samples:
            print(
                f"[JerseyDataset] WARNING: No images found under {root_dir}. "
                "Download SoccerNet Jersey Number Recognition data first."
            )

        if max_samples and len(self.samples) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def make_transforms(
    input_size: Tuple[int, int],   # (W, H)
    augment: bool = False,
) -> T.Compose:
    w, h = input_size
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if augment:
        return T.Compose([
            T.Resize((h + 16, w + 8)),
            T.RandomCrop((h, w)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Model factory — uses JerseyCNN so saved keys match JerseyReader.load
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 100, backbone_name: str = "resnet34") -> nn.Module:
    return JerseyCNN(num_classes=num_classes, pretrained=True, backbone_name=backbone_name)


# ---------------------------------------------------------------------------
# Training loop
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

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += int((preds == labels).sum())
        total      += imgs.size(0)

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

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += int((preds == labels).sum())
        total      += imgs.size(0)

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct   / total if total else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    cfg      = load_config(args.config)
    jcfg     = cfg["training"]["jersey_cnn"]
    device   = args.device or get_device()
    log_dir  = Path(jcfg["log_dir"])
    save_path = Path(jcfg["save_path"])

    log_dir.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Mode overrides ───────────────────────────────────────────────────
    kaggle_mode      = getattr(args, "kaggle",      False)
    kaggle_full_mode = getattr(args, "kaggle_full", False)

    if args.quick_test:
        print("[train_jersey_cnn] QUICK-TEST mode (CPU smoke-test).")
        epochs      = jcfg["quick_test_epochs"]
        batch_size  = jcfg["quick_test_batch_size"]
        max_samples = jcfg["quick_test_samples"]
        _generate_stubs(data_dir,
                        num_classes=jcfg.get("quick_test_num_classes", 10),
                        images_per_class=jcfg.get("quick_test_images_per_class", 40))
    elif kaggle_mode:
        print("[train_jersey_cnn] KAGGLE mode — 100 classes × 500 imgs, 20 epochs (GPU).")
        epochs      = jcfg.get("kaggle_epochs", 20)
        batch_size  = jcfg.get("kaggle_batch_size", 64)
        max_samples = None
        _generate_stubs(data_dir,
                        num_classes=jcfg.get("kaggle_num_classes", 100),
                        images_per_class=jcfg.get("kaggle_images_per_class", 500))
    elif kaggle_full_mode:
        print("[train_jersey_cnn] KAGGLE-FULL mode — real SoccerNet data.")
        epochs     = jcfg.get("full_epochs", 50)
        batch_size = jcfg.get("full_batch_size", 64)
        max_samples = None
    else:
        epochs      = args.epochs or jcfg["epochs"]
        batch_size  = jcfg["batch_size"]
        max_samples = None

    input_size  = tuple(jcfg["input_size"])   # (W, H)
    num_classes = jcfg["num_classes"]
    data_dir    = args.data_dir or jcfg["data_dir"]

    # ── Dataset ──────────────────────────────────────────────────────────
    train_tf = make_transforms(input_size, augment=True)
    val_tf   = make_transforms(input_size, augment=False)

    full_dataset = JerseyDataset(data_dir, transform=train_tf, max_samples=max_samples)

    if len(full_dataset) == 0:
        print(
            "[train_jersey_cnn] No data available. "
            "Skipping training — download SoccerNet jersey data first."
        )
        return

    val_size   = max(1, int(len(full_dataset) * jcfg["val_split"]))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Apply val transform to val split
    val_ds.dataset.transform = val_tf  # type: ignore

    # Kaggle has limited CPU RAM — cap workers to avoid system OOM
    num_workers = 0 if args.quick_test else (2 if kaggle_mode else jcfg["num_workers"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # ── Model / optimiser ────────────────────────────────────────────────
    backbone_name = jcfg.get("backbone", "resnet34")
    model     = build_model(num_classes=num_classes, backbone_name=backbone_name).to(device)
    label_smoothing = jcfg.get("label_smoothing", 0.10)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"[train_jersey_cnn] backbone={backbone_name} | label_smoothing={label_smoothing}")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=jcfg["lr"],
        weight_decay=jcfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer    = SummaryWriter(log_dir=str(log_dir))

    print(
        f"[train_jersey_cnn] Training {num_classes}-class CNN | "
        f"device={device} | epochs={epochs} | "
        f"train={train_size} val={val_size}"
    )

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ New best val_acc={best_val_acc:.4f}  saved → {save_path}")

    writer.close()
    print(f"[train_jersey_cnn] Done. Best val accuracy: {best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the two-digit jersey CNN.")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--device",   default=None)
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--data-dir", dest="data_dir", default=None)
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--kaggle", action="store_true",
                   help="GPU: 100 classes × 500 PIL-rendered images, 20 epochs (~15 min T4)")
    p.add_argument("--kaggle-full", dest="kaggle_full", action="store_true",
                   help="GPU: real SoccerNet jersey data, 50 epochs (hours, best accuracy)")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
