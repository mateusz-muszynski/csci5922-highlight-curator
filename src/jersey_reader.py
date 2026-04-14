"""
Stage 2 — Jersey Number Recognition
Two-step pipeline:
  1. Orientation filter  — discard rear-facing crops (pose heuristic).
  2. Two-digit CNN       — ResNet-18 backbone, 100 output classes (00–99).

Training data: SoccerNet Jersey Number Recognition dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import cv2


class JerseyCNN(nn.Module):
    """
    ResNet-18 backbone with a linear head that predicts one of 100 digit-pair
    classes (00, 01, … 99).
    """

    def __init__(self, num_classes: int = 100, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.40),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Main reader class
# ---------------------------------------------------------------------------

class JerseyReader:
    """
    Predict the jersey number visible in a player crop.

    Usage
    -----
    reader = JerseyReader(model_path="models/jersey_cnn.pt")
    number, confidence = reader.read(crop_bgr)
    # returns (None, 0.0) when orientation filter rejects the crop,
    # color hint filter rejects the crop, or confidence < threshold.

    color_hint : "dark" | "light" | None
        When set, crops whose average brightness does not match the hint
        are rejected before the CNN runs. Useful when the target player
        wears a distinctively dark (e.g. navy/black) or light (white)
        jersey and the opposing team wears the other colour.
        dark  → rejects crops with mean V (HSV) > dark_brightness_threshold
        light → rejects crops with mean V (HSV) < (255 - dark_brightness_threshold)
    """

    # ImageNet normalisation (backbone was pretrained on ImageNet)
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    # Mean HSV-V threshold below which a crop is considered "dark"
    _DARK_BRIGHTNESS_THRESHOLD: int = 85

    def __init__(
        self,
        model_path: str = "models/jersey_cnn.pt",
        num_classes: int = 100,
        input_size: Tuple[int, int] = (64, 128),   # (W, H)
        conf_threshold: float = 0.70,
        orientation_enabled: bool = True,
        facing_threshold: float = 0.35,
        color_hint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.num_classes = num_classes
        self.input_w, self.input_h = input_size
        self.conf_threshold = conf_threshold
        self.orientation_enabled = orientation_enabled
        self.facing_threshold = facing_threshold
        self.color_hint = color_hint  # "dark" | "light" | None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = JerseyCNN(num_classes=num_classes, pretrained=False)
        model_file = Path(model_path)
        if model_file.exists():
            state = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            print(
                f"[JerseyReader] WARNING: weights not found at '{model_path}'. "
                "Running with random weights — train first!"
            )
        self.model.to(self.device)
        self.model.eval()

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_h, self.input_w)),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self, crop_bgr: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """
        Predict jersey number from a BGR player crop.

        Returns
        -------
        (number, confidence)  where number is 0–99 or None if rejected.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0

        if self.orientation_enabled and self._is_rear_facing(crop_bgr):
            return None, 0.0

        if self.color_hint and not self._matches_color_hint(crop_bgr):
            return None, 0.0

        tensor = self._preprocess(crop_bgr)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        confidence = float(conf.item())
        number = int(pred.item())

        if confidence < self.conf_threshold:
            return None, confidence

        return number, confidence

    def build_transform(self, augment: bool = False) -> T.Compose:
        """Return a torchvision transform suitable for training or inference."""
        if not augment:
            return self._transform

        return T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_h + 16, self.input_w + 8)),
            T.RandomCrop((self.input_h, self.input_w)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomHorizontalFlip(p=0.3),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """BGR → RGB → transform → (1, C, H, W) tensor on self.device."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(crop_rgb).unsqueeze(0).to(self.device)
        return tensor

    def _matches_color_hint(self, crop_bgr: np.ndarray) -> bool:
        """
        Return True if the crop's dominant jersey colour matches self.color_hint.

        Measures the mean HSV Value channel over the middle torso band of the
        crop (rows 25%–65%), which is where the jersey number lives.

        dark  → mean V < _DARK_BRIGHTNESS_THRESHOLD
        light → mean V > (255 - _DARK_BRIGHTNESS_THRESHOLD)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return True  # can't tell — don't filter

        h, w = crop_bgr.shape[:2]
        band = crop_bgr[int(h * 0.25): int(h * 0.65), :, :]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mean_v = float(hsv[:, :, 2].mean())

        if self.color_hint == "dark":
            return mean_v < self._DARK_BRIGHTNESS_THRESHOLD
        if self.color_hint == "light":
            return mean_v > (255 - self._DARK_BRIGHTNESS_THRESHOLD)
        return True

    def _is_rear_facing(self, crop_bgr: np.ndarray) -> bool:
        """
        Lightweight heuristic: estimate whether the player is back-facing by
        measuring the horizontal brightness gradient in the upper-body region.

        A front-facing player typically has a visible number patch (high-contrast
        region in the centre of the torso). A rear-facing player has a smooth
        gradient or the number patch sits off-centre.

        This avoids a full pose estimation dependency at inference time.
        Returns True when the crop is likely rear-facing → skip jersey read.
        """
        if crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 8:
            return False

        # Focus on middle horizontal band (likely jersey area)
        h, w = crop_bgr.shape[:2]
        band = crop_bgr[int(h * 0.25): int(h * 0.65), :, :]

        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Horizontal gradient magnitude (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        abs_grad = np.abs(sobelx)

        # High gradient → number / seam patterns likely visible (front-facing)
        # Low gradient everywhere → plain back (rear-facing)
        mean_grad = float(abs_grad.mean())
        # Threshold tuned empirically; adjust via facing_threshold in config
        rear_threshold = 5.0 * (1.0 - self.facing_threshold)
        return mean_grad < rear_threshold
