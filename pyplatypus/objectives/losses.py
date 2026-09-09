"""Losses as modules, and the factory that turns a spec into one.

All of them take raw logits. A model that had already applied softmax would be silently
wrong here in a way that is very hard to see, which is why `build_model` returns logits
and there is a test asserting it.
"""

from __future__ import annotations

import torch
from torch import nn

from pyplatypus.objectives import functional as f
from pyplatypus.spec.components import (
    CceDiceLoss,
    CceLoss,
    ComboLoss,
    DiceLoss,
    FocalLoss,
    FocalTverskyLoss,
    IouLoss,
    LossSpec,
    LovaszLoss,
    TverskyLoss,
)


class SegmentationLoss(nn.Module):
    """Logits in, one scalar out."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Dice(SegmentationLoss):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        return 1 - f.dice_coefficient(f.probabilities(logits), target, self.smooth).mean()


class Iou(SegmentationLoss):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        return 1 - f.iou_coefficient(f.probabilities(logits), target, self.smooth).mean()


class Cce(SegmentationLoss):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        return f.cross_entropy(logits, target, self.label_smoothing)


class CceDice(SegmentationLoss):
    """The workhorse blend: cross-entropy keeps the gradients healthy early, Dice pulls
    the overlap up once the easy pixels are done."""

    def __init__(self, cce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.cce_weight = cce_weight
        self.smooth = smooth

    def forward(self, logits, target):
        cce = f.cross_entropy(logits, target)
        dice = 1 - f.dice_coefficient(f.probabilities(logits), target, self.smooth).mean()
        return self.cce_weight * cce + (1 - self.cce_weight) * dice


class Focal(SegmentationLoss):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        return f.focal(logits, target, self.gamma, self.alpha)


class Tversky(SegmentationLoss):
    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, target):
        coefficient = f.tversky_coefficient(
            f.probabilities(logits), target, self.alpha, self.smooth
        )
        return 1 - coefficient.mean()


class FocalTversky(SegmentationLoss):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        coefficient = f.tversky_coefficient(
            f.probabilities(logits), target, self.alpha, self.smooth
        )
        return (1 - coefficient).clamp_min(0).pow(self.gamma).mean()


class Combo(SegmentationLoss):
    def __init__(self, alpha: float = 0.5, ce_ratio: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio

    def forward(self, logits, target):
        return f.combo(logits, target, self.alpha, self.ce_ratio)


class Lovasz(SegmentationLoss):
    def __init__(self, per_image: bool = False):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, target):
        return f.lovasz_softmax(logits, target, self.per_image)


_BUILDERS = {
    IouLoss: lambda s: Iou(s.smooth),
    DiceLoss: lambda s: Dice(s.smooth),
    CceLoss: lambda s: Cce(s.label_smoothing),
    CceDiceLoss: lambda s: CceDice(s.cce_weight, s.smooth),
    FocalLoss: lambda s: Focal(s.gamma, s.alpha),
    TverskyLoss: lambda s: Tversky(s.alpha, s.smooth),
    FocalTverskyLoss: lambda s: FocalTversky(s.alpha, s.gamma, s.smooth),
    ComboLoss: lambda s: Combo(s.alpha, s.ce_ratio),
    LovaszLoss: lambda s: Lovasz(s.per_image),
}


def build_loss(spec: LossSpec) -> SegmentationLoss:
    """Spec to module. The spec already guaranteed the name and the ranges are valid."""
    builder = _BUILDERS.get(type(spec))
    if builder is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise KeyError(f"no implementation for loss {type(spec).__name__}")
    return builder(spec)
