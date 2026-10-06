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
    BoundaryLoss,
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
    """Logits in, one scalar out.

    `needs_distance` is how a loss asks the data path for the signed distance map of the
    target. It is false for every loss but `boundary`, and the flag exists rather than an
    `isinstance` check so that the engine, the loader and the trainer each ask the loss
    itself rather than three places knowing a name.
    """

    needs_distance: bool = False

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Two arguments here, three in a loss that sets `needs_distance`.

        Not a third parameter on every loss: eight of the nine would take one they never
        read, and the trainer asks the flag rather than passing something nobody wants.
        """
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


class Boundary(SegmentationLoss):
    """A region loss plus Kervadec's surface term, weighted by `alpha`.

    `mean(phi * p)` where `phi` is the signed distance to the truth's boundary - negative
    inside the object, positive outside - and `p` is the predicted probability. A voxel
    predicted far outside the truth costs in proportion to how far, which is exactly what
    an overlap score does not measure: Dice counts a voxel the same wherever it sits.

    **Never alone.** The surface term has no notion of how much of the object was found,
    only of where the probability was put, so by itself it is minimised by a confident
    prediction deep inside a shrunken object. The region term is what keeps it honest, and
    `alpha` of 1 is refused rather than silently allowed.

    Measured on a sphere of known volume: given two predictions of near-equal Dice, one
    31.9% too large and one 25.2% too small, Dice *prefers the larger* by 0.0067 while this
    term prefers the smaller by 0.0025 - 0.8% of Dice's scale against 3.1% of its own. That
    is the asymmetry it exists to correct, and it is modest; whether it moves a trained
    model's volumes is a question for a training run, not for this docstring.
    """

    needs_distance = True

    def __init__(self, region: SegmentationLoss, alpha: float = 0.5):
        super().__init__()
        self.region = region
        self.alpha = alpha

    def forward(self, logits, target, distance=None):
        if distance is None:
            raise ValueError(
                "the boundary loss needs the signed distance map of the target, which the "
                "loader computes when the loss asks for it. Reaching here without one "
                "means the loader was built without `with_distance=True`."
            )
        surface = (f.probabilities(logits) * distance).mean()
        return self.alpha * self.region(logits, target) + (1 - self.alpha) * surface


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
    BoundaryLoss: lambda s: Boundary(build_loss(s.region), s.alpha),
}


def loss_needs_distance(spec: LossSpec) -> bool:
    """Whether this loss wants the target's signed distance map, asked of the spec.

    The engine builds the loader before the trainer builds the loss, so the question has
    to be answerable from the specification alone. One place knows the answer, and it is
    beside the losses rather than in the engine.
    """
    return isinstance(spec, BoundaryLoss)


def build_loss(spec: LossSpec) -> SegmentationLoss:
    """Spec to module. The spec already guaranteed the name and the ranges are valid."""
    builder = _BUILDERS.get(type(spec))
    if builder is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise KeyError(f"no implementation for loss {type(spec).__name__}")
    return builder(spec)
