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
    L1Loss,
    LossSpec,
    LovaszLoss,
    MseLoss,
    SmoothL1Loss,
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

    #: Whether this loss asks the data path for the mask of the voxels its target is
    #: defined on. The same mechanism as `needs_distance` and for the same reason: the
    #: engine builds the loader before the trainer builds the loss, so the question has to
    #: be answerable before anything is trained. Only the regression losses set it, and
    #: only when they were told to score the foreground.
    needs_mask: bool = False

    #: Whether the channels are independent questions rather than alternatives. Set by
    #: `build_loss` from the data's `regions`, never guessed from the target: a channel of
    #: all zeros is ordinary in both worlds, so there is nothing in the tensors to read it
    #: off, and a loss that inferred it would be right most of the time.
    multilabel: bool = False

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits to probabilities, the way this loss was told to.

        Every overlap-based loss here goes through this rather than calling the functional
        form directly, so the softmax-or-sigmoid decision has exactly one home. A method for
        the reason `needs_distance` is a flag: the loss is asked, rather than several places
        knowing which losses care.
        """
        return f.probabilities(logits, multilabel=self.multilabel)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Two arguments here, three in a loss that sets `needs_distance` or `needs_mask`.

        Not a third parameter on every loss: most of them would take one they never read,
        and the trainer asks the flag rather than passing something nobody wants. The two
        flags are mutually exclusive by construction - no loss here asks for both - and a
        test says so, because the loader has one slot to put the answer in.
        """
        raise NotImplementedError


class Dice(SegmentationLoss):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        return 1 - f.dice_coefficient(self.probabilities(logits), target, self.smooth).mean()


class Iou(SegmentationLoss):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        return 1 - f.iou_coefficient(self.probabilities(logits), target, self.smooth).mean()


class Cce(SegmentationLoss):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        if self.multilabel:
            return f.binary_cross_entropy(logits, target, self.label_smoothing)
        return f.cross_entropy(logits, target, self.label_smoothing)


class CceDice(SegmentationLoss):
    """The workhorse blend: cross-entropy keeps the gradients healthy early, Dice pulls
    the overlap up once the easy pixels are done."""

    def __init__(self, cce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.cce_weight = cce_weight
        self.smooth = smooth

    def forward(self, logits, target):
        cce = (
            f.binary_cross_entropy(logits, target)
            if self.multilabel
            else f.cross_entropy(logits, target)
        )
        dice = 1 - f.dice_coefficient(self.probabilities(logits), target, self.smooth).mean()
        return self.cce_weight * cce + (1 - self.cce_weight) * dice


class Focal(SegmentationLoss):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        if self.multilabel:
            return f.binary_focal(logits, target, self.gamma, self.alpha)
        return f.focal(logits, target, self.gamma, self.alpha)


class Tversky(SegmentationLoss):
    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, target):
        coefficient = f.tversky_coefficient(
            self.probabilities(logits), target, self.alpha, self.smooth
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
            self.probabilities(logits), target, self.alpha, self.smooth
        )
        return (1 - coefficient).clamp_min(0).pow(self.gamma).mean()


class Combo(SegmentationLoss):
    def __init__(self, alpha: float = 0.5, ce_ratio: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio

    def forward(self, logits, target):
        return f.combo(logits, target, self.alpha, self.ce_ratio, multilabel=self.multilabel)


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
        surface = (self.probabilities(logits) * distance).mean()
        return self.alpha * self.region(logits, target) + (1 - self.alpha) * surface


class Lovasz(SegmentationLoss):
    def __init__(self, per_image: bool = False):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, target):
        return f.lovasz_softmax(logits, target, self.per_image)


class Regression(SegmentationLoss):
    """Score a continuous target, and say where it is defined.

    **Nothing is activated.** `probabilities()` is never called here: a flow field or a
    distance in pixels is unbounded, and a sigmoid in front of one would squash it while
    still returning a number and a gradient. So the first argument is the model's raw
    output used as the prediction, and `logits` is the wrong word for it - kept only
    because the signature is shared.

    **`where` is the whole content of this family**, and the reason is measured. A
    continuous instance target - flows, or horizontal and vertical distances - is defined
    on the objects and undefined off them: there is no "direction towards the centre" for
    a pixel belonging to no object, so the builder writes zero there. On the 536 Data
    Science Bowl cases on this disk the foreground is **13.97% of the pixels at 256x256**
    (13.48% at native size, 0.03% to 59.2% per case, 167 cases under 5%), so an unmasked
    mean spends 86% of itself on a target that a single output bias satisfies exactly.

    `foreground` therefore divides by the mask's own total, which makes the number mean
    "average error per object voxel". `everywhere` is the other defensible answer and is
    what CellPose and HoVer-Net's plain regression terms do - their background zero is a
    real value, not a missing one, and HoVer-Net masks only its gradient term. Both are
    offered; neither is inferred.

    **The mask is never read off the target**, which is the one shape this family refuses.
    A flow field is exactly zero at the centre of every object and the horizontal and
    vertical maps are each zero along a line through it, so "the voxels where the target is
    nonzero" excludes precisely the seed pixels - and the watershed oracle on #211 measured
    that the seeds, not the seams, are what limits this approach. A mask inferred that way
    would be wrong in the one place the method depends on, and right everywhere a test
    would look.
    """

    def __init__(self, where: str = "foreground"):
        super().__init__()
        self.where = where
        self.needs_mask = where == "foreground"

    def error(self, prediction: torch.Tensor, target: torch.Tensor, mask) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, logits, target, mask=None):
        if self.needs_mask and mask is None:
            raise ValueError(
                f"`{type(self).__name__.lower()}` with `where: foreground` needs the mask "
                "of the voxels its target is defined on, which the loader computes when the "
                "loss asks for it. Reaching here without one means the loader was built "
                "without `with_foreground=True`. `where: everywhere` scores the whole "
                "picture instead, which is a different objective rather than a workaround."
            )
        return self.error(logits, target, mask if self.needs_mask else None)


class Mse(Regression):
    """Squared error. The gradient grows with the error, so one badly wrong voxel counts
    for a lot - which is what you want on a target whose large values are the informative
    ones, and what you do not want on noisy labels."""

    def error(self, prediction, target, mask):
        return f.mse(prediction, target, mask=mask)


class L1(Regression):
    """Absolute error, whose gradient is the same size however wrong the voxel is. Robust
    to a few bad labels and indifferent to being nearly right."""

    def error(self, prediction, target, mask):
        return f.l1(prediction, target, mask=mask)


class SmoothL1(Regression):
    """Squared within `beta` of the truth and absolute outside it: `mse`'s gradients where
    the prediction is close and `l1`'s tolerance where it is not."""

    def __init__(self, where: str = "foreground", beta: float = 1.0):
        super().__init__(where)
        self.beta = beta

    def error(self, prediction, target, mask):
        return f.smooth_l1(prediction, target, self.beta, mask=mask)


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
    BoundaryLoss: lambda s: Boundary(build_loss(s.region), s.alpha),  # flag set below
    MseLoss: lambda s: Mse(s.where),
    L1Loss: lambda s: L1(s.where),
    SmoothL1Loss: lambda s: SmoothL1(s.where, s.beta),
}


def loss_needs_distance(spec: LossSpec) -> bool:
    """Whether this loss wants the target's signed distance map, asked of the spec.

    The engine builds the loader before the trainer builds the loss, so the question has
    to be answerable from the specification alone. One place knows the answer, and it is
    beside the losses rather than in the engine.
    """
    return isinstance(spec, BoundaryLoss)


def loss_needs_mask(spec: LossSpec) -> bool:
    """Whether this loss wants the mask of the voxels its target is defined on.

    Asked of the specification for the reason `loss_needs_distance` is: the loader is built
    before the loss, so the answer has to come from the spec alone. The two never both say
    yes - a regression loss is not a region loss and `boundary` takes only region losses -
    which is what lets the loader carry one extra tensor rather than a pair of optional
    ones. `test_the_two_data_flags_are_never_both_set` is where that is held.
    """
    return isinstance(spec, MseLoss | L1Loss | SmoothL1Loss) and spec.where == "foreground"


def loss_supports_multilabel(spec: LossSpec) -> bool:
    """Whether this loss means anything when a voxel may be in several classes at once.

    Answered by the loss specification itself, which carries `supports_multilabel`. Kept as
    a function because this is the question callers ask, and because an `isinstance` list
    here would be a second place to update when a loss is added - the kind of pair that
    stays agreed right up until it does not.
    """
    return spec.supports_multilabel


def build_loss(spec: LossSpec, *, multilabel: bool = False) -> SegmentationLoss:
    """Spec to module. The spec already guaranteed the name and the ranges are valid.

    `multilabel` travels into nested losses too - `boundary` wraps a region loss, and a
    surface term computed from a softmax inside a sigmoid model would be wrong in a way
    that still trains.
    """
    if multilabel and not loss_supports_multilabel(spec):
        raise ValueError(
            f"{type(spec).__name__} cannot be used with overlapping classes; see "
            "loss_supports_multilabel"
        )
    builder = _BUILDERS.get(type(spec))
    if builder is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise KeyError(f"no implementation for loss {type(spec).__name__}")
    loss = builder(spec)
    loss.multilabel = multilabel
    if isinstance(loss, Boundary):
        loss.region.multilabel = multilabel
    return loss
