"""Metrics, reported on hard predictions.

Losses work on probabilities because they need a gradient. A metric does not, so it is
computed on the argmax - the mask the user will actually be handed. Soft metrics read
higher than the model deserves, and the gap grows with how unsure it is.
"""

from __future__ import annotations

import torch
from torch import nn

from pyplatypus.objectives import functional as f
from pyplatypus.spec.components import (
    ClDiceMetric,
    DiceMetric,
    IouMetric,
    MetricSpec,
    SurfaceDiceMetric,
    TverskyMetric,
)


class SegmentationMetric(nn.Module):
    """Higher is better, always in 0..1."""

    name = "metric"

    #: Whether a case's score can be rebuilt from its pieces' overlap counts. True for
    #: anything written as a formula on TP, FP and FN, which is how a tiled image is scored
    #: as one image. False for a metric that reads the shape of a whole mask.
    accumulates = True

    #: Whether the channels are independent questions. Set by `build_metrics` from the
    #: data's `regions`. It changes two things here: how a hard prediction is taken, and
    #: whether `include_background` has anything to exclude.
    multilabel = False

    #: Millimetres per voxel, or None when the samples are not on a common grid. Set by
    #: `build_metrics` from the data's `target_spacing`, the same way `multilabel` is set
    #: from `regions`: both are properties of the data rather than of the metric. Only
    #: `surface_dice` reads it - a tolerance in millimetres is not a tolerance in voxels -
    #: and it is declared here rather than there because the builder sets it on every
    #: metric, and an attribute that exists on one class only is a surprise to the reader.
    spacing: tuple[float, ...] | None = None

    def __init__(self, smooth: float = 1.0, include_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def combine(self, tp, fp, fn):
        """The metric itself, as a formula on overlap statistics.

        Everything else here is plumbing on top of this. Written at this level because a
        score is not always computed from one array: a tiled image arrives in pieces, and
        summing its TP, FP and FN and applying the formula once is the score for the whole
        image, where averaging the pieces' scores is not - a ratio of sums is not the mean
        of ratios.
        """
        raise NotImplementedError

    def coefficient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.combine(*f.overlaps(prediction, target))

    def reduce(self, per_class: torch.Tensor) -> torch.Tensor:
        """One number from per-class scores, honouring include_background.

        With nested regions there is nothing to honour: `regions` has no background entry,
        so channel 0 is the first region a user asked for and dropping it would drop a class
        rather than a convention. The flag is ignored here rather than refused, because
        `cldice` defaults it to False and a default is not a request - the spec refuses the
        combination only when it was written down.
        """
        if self.multilabel:
            return per_class.mean()
        if not self.include_background:
            if per_class.shape[-1] < 2:
                raise ValueError(
                    "include_background=False needs at least two classes to leave one out"
                )
            per_class = per_class[..., 1:]
        return per_class.mean()

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # `logits > 0` is `sigmoid(logits) > 0.5` exactly, without computing the sigmoid -
        # and the point is that each channel decides for itself. An argmax here would hand
        # back one class per voxel, which is the answer the model was not asked for: a voxel
        # in both the core and the whole tumour would be reported in whichever scored higher.
        hard = (
            (logits > 0).to(target.dtype)
            if self.multilabel
            else f.as_onehot(logits.argmax(dim=1), logits.shape[1])
        )
        return self.reduce(self.coefficient(hard, target))  # (batch, class) -> scalar


class Dice(SegmentationMetric):
    name = "dice"

    def combine(self, tp, fp, fn):
        return f.dice_from_overlaps(tp, fp, fn, self.smooth)


class Iou(SegmentationMetric):
    name = "iou"

    def combine(self, tp, fp, fn):
        return f.iou_from_overlaps(tp, fp, fn, self.smooth)


class Tversky(SegmentationMetric):
    name = "tversky"

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0, include_background: bool = True):
        super().__init__(smooth, include_background)
        self.alpha = alpha

    def combine(self, tp, fp, fn):
        return f.tversky_from_overlaps(tp, fp, fn, self.alpha, self.smooth)


class ClDice(SegmentationMetric):
    """Centerline Dice: does the *structure* coincide, rather than the pixels.

    A retinal vessel is one to five pixels wide. Move it by one pixel and Dice can halve,
    while the thing that matters - is it there, is it continuous - has not changed. This
    scores each mask against the other's skeleton, so a displaced vessel scores well and a
    broken one does not.

    Not retina-specific: airways, neurons, cracks, roads and catheters have the same shape
    of problem.
    """

    name = "cldice"
    accumulates = False

    def __init__(self, iterations: int = 5, smooth: float = 1.0, include_background: bool = False):
        super().__init__(smooth, include_background)
        self.iterations = iterations

    def reduce(self, per_class: torch.Tensor) -> torch.Tensor:
        """Refuse a class whose structure was too thick to peel, rather than score it 1.0.

        Checked after the background is dropped, so a background too solid to skeletonise -
        which is ordinary - does not stop a run that never asked about it.
        """
        chosen = per_class if self.include_background or self.multilabel else per_class[..., 1:]
        if torch.isnan(chosen).any():
            raise ValueError(
                f"clDice at iterations={self.iterations} found no centreline in a mask that "
                "is not empty: the structure is thicker than that many peels can reduce. "
                "Raise `iterations` past its half-width. Reported rather than scored, "
                "because an empty skeleton otherwise reads as a perfect 1.0."
            )
        return super().reduce(per_class)

    def combine(self, tp, fp, fn):
        raise TypeError(
            "clDice cannot be computed from overlap counts: a skeleton is a property of a "
            "whole mask, not of how many pixels hit and missed. `accumulates` is False for "
            "exactly this reason, and a caller that reaches here has not checked it."
        )

    def coefficient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return f.cldice_from_masks(prediction, target, self.iterations, self.smooth)


class SurfaceDice(SegmentationMetric):
    """How much of each boundary is within `tau` of the other.

    The third question this package can ask of a mask, after volume (`dice`) and
    connectedness (`cldice`): **how far wrong** is the edge. A model can hold 0.88 of Dice
    with volumes a fifth too large, because the overshoot is all at the boundary and on a
    small lesion the boundary is most of the object.

    `spacing` is set by `build_metrics` from the data's `target_spacing`, so `tau` is in
    millimetres when the samples are on a common grid and in voxels when they are not. It
    is the only metric here that reads it, and the only one whose number is therefore not
    comparable between two runs that resized differently.
    """

    name = "surface_dice"
    accumulates = False

    def __init__(self, tau: float, smooth: float = 1.0, include_background: bool = False):
        super().__init__(smooth, include_background)
        self.tau = tau

    def combine(self, tp, fp, fn):
        raise TypeError(
            "surface Dice cannot be computed from overlap counts: a boundary is a property "
            "of a whole mask, not of how many voxels hit and missed. `accumulates` is False "
            "for exactly that reason, and a caller that reaches here has not checked it."
        )

    def coefficient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return f.surface_dice_from_masks(prediction, target, self.tau, self.spacing, self.smooth)


_BUILDERS = {
    IouMetric: lambda s: Iou(s.smooth, s.include_background),
    DiceMetric: lambda s: Dice(s.smooth, s.include_background),
    TverskyMetric: lambda s: Tversky(s.alpha, s.smooth, s.include_background),
    ClDiceMetric: lambda s: ClDice(s.iterations, s.smooth, s.include_background),
    SurfaceDiceMetric: lambda s: SurfaceDice(s.tau, s.smooth, s.include_background),
}


def build_metric(
    spec: MetricSpec,
    *,
    multilabel: bool = False,
    spacing: tuple[float, ...] | None = None,
) -> SegmentationMetric:
    """`multilabel` and `spacing` are both properties of the data rather than of the metric.

    They arrive here rather than in a constructor for the same reason: the specification
    says what to measure and the data says what the channels and the voxels are, and a
    metric that had to find `regions` or `target_spacing` for itself would be a second
    place that knows the shape of a spec.
    """
    builder = _BUILDERS.get(type(spec))
    if builder is None:  # pragma: no cover
        raise KeyError(f"no implementation for metric {type(spec).__name__}")
    metric = builder(spec)
    metric.multilabel = multilabel
    metric.spacing = spacing
    return metric


def build_metrics(
    specs: list[MetricSpec],
    *,
    multilabel: bool = False,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, SegmentationMetric]:
    """Keyed by name, because that is how they end up in the history table."""
    return {spec.name: build_metric(spec, multilabel=multilabel, spacing=spacing) for spec in specs}
