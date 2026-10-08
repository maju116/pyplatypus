"""One model to train.

`input_shape` carries the spatial size and nothing else; `rank` is read off its length.
That is PLAN.md rule 1, and it is why there is no `net_h`/`net_w` pair here - a pair
cannot become a triple without an API break.

`ModelSpec` is what every task's model has: a name, a size, how long to train, what to
train with, and whether to train at all. What it deliberately does *not* have is `loss`
and `metrics`. Segmentation chooses both from a menu of nine and three; YOLOv3 has one
composite objective that is part of the architecture, and mean average precision is not
one option among several. Offering a choice that does not exist is worse than offering
none - see the architecture comparison in the README for the same argument about weights.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, model_validator

from pyplatypus.spec.common import Activation, Architecture, SpatialShape, SpecModel
from pyplatypus.spec.components import (
    Adam,
    AugmentationStep,
    CallbackSpec,
    CceLoss,
    IouMetric,
    LossSpec,
    MetricSpec,
    OptimizerSpec,
)


class Initialiser(str, Enum):
    HE_NORMAL = "he_normal"
    HE_UNIFORM = "he_uniform"
    GLOROT_NORMAL = "glorot_normal"
    GLOROT_UNIFORM = "glorot_uniform"


class ModelSpec(SpecModel):
    """What one model needs whatever it is predicting."""

    name: str = Field(min_length=1, description="Unique within a spec; names the outputs.")

    input_shape: SpatialShape
    channels: int = Field(
        3,
        ge=1,
        description=(
            "How many channels the network takes: 3 for colour, 1 for greyscale, and one per "
            "sequence for multi-modal data such as the four MRI sequences in BraTS. Derived "
            "rather than declared when `channels_from` names the files, because then the data "
            "decides. Not adopted from a weights file - the data pipeline needs it before a "
            "network exists, and a specification has to be checkable offline."
        ),
    )

    optimizer: OptimizerSpec = Field(
        default_factory=Adam,
        description=(
            "How the weights are updated. `name` selects which one and decides which other "
            "keys this block accepts."
        ),
    )
    callbacks: list[CallbackSpec] = Field(
        default_factory=list,
        description=(
            "Things that happen between epochs: stopping early, writing checkpoints, moving "
            "the learning rate, averaging the weights. A callback watching a number no metric "
            "will produce is refused when the specification is built rather than waited on "
            "forever, and so is one watching `val_*` in a run with no validation set."
        ),
    )
    augmentation: list[AugmentationStep] | None = Field(
        None,
        description=(
            "albumentations transforms, applied to the training data only - never to "
            "validation, where the point is to measure the same thing every epoch. Each step "
            "is probed at the model's own input size while the pipeline is built, because "
            "support for volumes is uneven and a transform that cannot take one raises from "
            "inside the library."
        ),
    )

    epochs: int = Field(
        10,
        ge=1,
        description=(
            "How many passes over the training data. The default is deliberately small: it "
            "is enough to see that a run works and not enough to mistake for a result. With "
            "`cosine_annealing` this is also the length the schedule is spread over."
        ),
    )
    batch_size: int = Field(
        8,
        ge=1,
        description=(
            "How many samples go through the network at once. Limited by memory rather than "
            "by the problem, and it is the first thing to lower when a 3D run will not fit. "
            "It interacts with `batch_normalization`: statistics taken over very few samples "
            "are noisy."
        ),
    )

    weights: str | None = Field(
        None,
        description="Registry name (e.g. 'dsbowl-unet') or a path to a local checkpoint.",
    )
    fit: bool = Field(True, description="Set false to load weights and skip training.")

    @property
    def rank(self) -> int:
        """2 for images, 3 for volumes. Derived, never declared."""
        return len(self.input_shape)

    @property
    def monitorable(self) -> set[str]:
        """Quantities a callback may watch, given what this model reports."""
        return {"train_loss", "val_loss"}

    def weights_fingerprint(self) -> dict:
        """What must match for a file of weights to belong to this model.

        Asked of the spec rather than listed in `pyplatypus.weights`, because the answer
        is architecture-specific: `blocks` and `filters` identify a U-shaped model and
        mean nothing to a detector, whose head width is set by its anchor count instead.
        A fixed list in the weights module would have to grow a branch per task and would
        reach for a field that is not there.
        """
        return {
            "architecture": getattr(self.architecture, "value", self.architecture),
            "input_shape": list(self.input_shape),
            "channels": self.channels,
            "rank": self.rank,
        }

    @model_validator(mode="after")
    def swa_and_a_decaying_rate_cancel_each_other(self):
        """`swa` with `cosine_annealing` averages a set of nearly identical snapshots.

        Averaging is only worth something while the weights are still moving, and a cosine
        that has decayed towards its floor by the time averaging starts has stopped them -
        so the average is the last epoch and the callback did nothing. That is the worst
        kind of combination: both halves are configured, neither complains, and the result
        is indistinguishable from not having asked.

        `swa.learning_rate` is the other half of the recipe and is what a run should use
        instead; `reduce_lr_on_plateau` is not refused, because it lowers the rate on
        evidence rather than on schedule and may never fire at all.
        """
        names = {callback.name for callback in self.callbacks}
        if "swa" in names and "cosine_annealing" in names:
            raise ValueError(
                "`swa` and `cosine_annealing` undo each other: averaging needs the weights "
                "to still be moving, and a cosine has decayed to its floor by the time "
                "averaging begins, so the average is just the last epoch. Set "
                "`swa.learning_rate` to hold the rate while it averages, or drop one of "
                "the two."
            )
        return self

    @model_validator(mode="after")
    def callbacks_watch_something_that_exists(self):
        """A callback watching 'val_dice' when no Dice metric was requested would wait
        forever for a number that never arrives. The old package could not catch this
        because the monitor was a free string checked nowhere."""
        available = self.monitorable
        for callback in self.callbacks:
            watched = getattr(callback, "monitor", None)
            if watched is not None and watched not in available:
                raise ValueError(
                    f"callback '{callback.name}' watches '{watched}', which this model "
                    f"does not produce; available: {', '.join(sorted(available))}"
                )
        return self

    @model_validator(mode="after")
    def not_fitting_needs_weights(self):
        if not self.fit and self.weights is None:
            raise ValueError("fit=false only makes sense together with weights")
        return self


#: Metrics that cannot be rebuilt from the pieces of a tiled image, because they read the
#: shape of a whole mask rather than its overlap counts. Kept here as names rather than as
#: an import from the objectives, which would make the specification layer depend on torch.
_WHOLE_MASK_METRICS = frozenset({"cldice"})


class SegmentationModel(ModelSpec):
    architecture: Architecture = Field(
        Architecture.U_NET,
        description=(
            "Which U-shaped network. Measured on Data Science Bowl, one split, 60 epochs, "
            "the four span 0.9187 to 0.9217 of Dice - a spread of 0.0030, against 0.054 "
            "between images within any one of them, and 0.0012 from re-running one of them "
            "with another seed. On that problem the architecture is not where the result "
            "comes from. Adopted from the weights file when `weights` names one."
        ),
    )

    blocks: int = Field(
        4,
        ge=1,
        le=8,
        description=(
            "How many times the resolution is halved, so how much context the deepest layer "
            "sees. `input_shape` must divide by 2^blocks. Adopted from the weights file when "
            "`weights` is given and this is not; with an `encoder`, the backbone's own depth "
            "is the ceiling."
        ),
    )
    filters: int = Field(
        16,
        ge=1,
        description=(
            "Filters in the first block, doubled at every level, so this sets the model's "
            "size more than anything else does. Adopted from the weights file when `weights` "
            "is given and this is not."
        ),
    )
    block_width: int = Field(
        2,
        ge=1,
        le=4,
        description=(
            "Convolutions per block, at every level. 2 is what every U-Net paper uses and "
            "what the published weights were trained with; raising it adds depth without "
            "adding levels, which is the lever to reach for when the objects are small "
            "enough that halving the resolution again would lose them."
        ),
    )
    dropout: float = Field(
        0.0,
        ge=0,
        lt=1,
        description=(
            "Drop this fraction during training. 0 switches it off, which is the default "
            "because these models are small enough that batch normalisation usually carries "
            "the regularisation on its own."
        ),
    )

    batch_normalization: bool = Field(
        True,
        description=(
            "Normalise between convolutions. On by default, and worth knowing about when "
            "weights are averaged: `swa` has to recompute these statistics over the training "
            "data, because an averaged weight tensor inherits them from whichever epoch was "
            "last rather than averaging them."
        ),
    )
    separable_conv: bool = Field(
        False,
        description=(
            "Depthwise-separable convolutions: far fewer parameters for the same shape, and "
            "a little slower to converge."
        ),
    )
    spatial_dropout: bool = Field(
        True,
        description=(
            "Drop whole feature maps rather than individual activations. The right kind for "
            "images, because neighbouring pixels in one map are correlated and dropping them "
            "one at a time removes less than it appears to. Has no effect unless `dropout` is "
            "above 0."
        ),
    )
    upsample: bool = Field(False, description="Upsample+conv instead of transposed convolution.")
    deep_supervision: bool = Field(
        False,
        description=(
            "Train every decoder depth rather than only the last, each at input resolution. "
            "`u_net_plus_plus` only, since it is the architecture with intermediate outputs "
            "to supervise."
        ),
    )
    activation: Activation = Field(
        Activation.RELU,
        description="The non-linearity between convolutions, the same one throughout.",
    )
    initialiser: Initialiser = Field(
        Initialiser.HE_NORMAL,
        description=(
            "How the weights start. The `he_*` family is scaled for ReLU-like activations "
            "and `glorot_*` for symmetric ones, so this and `activation` are a pair."
        ),
    )

    loss: LossSpec = Field(
        default_factory=CceLoss,
        description=(
            "The objective trained against. The `loss` column of a score table is comparable "
            "only between models trained on the same one - a Focal-Tversky of 0.05 is not "
            "better than a cross-entropy of 0.14, it is not the same question - which is why "
            "the table carries the loss's name beside it."
        ),
    )
    metrics: list[MetricSpec] = Field(
        default_factory=lambda: [IouMetric()],
        description=(
            "What to report beside the loss, computed on the mask rather than on the "
            "objective, so these stay comparable across models. A callback watching "
            "`val_<metric>` is refused when nothing here will produce it."
        ),
    )
    splits: tuple[int, ...] | None = Field(
        None,
        description=(
            "Cut each image into a grid of tiles instead of shrinking it. One entry per "
            "spatial dimension, e.g. [2, 3] cuts into 2 rows by 3 columns. The source is "
            "read at splits * input_shape and divided; every tile is input_shape. Leave "
            "unset to resize the whole image instead."
        ),
    )

    encoder: str | None = Field(
        None,
        description=(
            "A timm backbone to use as the contracting path, e.g. 'resnet34'. Leave "
            "unset for the built-in encoder, which is the only one that works in 3D."
        ),
    )
    pretrained: bool = Field(
        False,
        description=(
            "Load the encoder's ImageNet weights. Separate from `encoder` on purpose: "
            "naming a backbone builds that architecture, and this is the flag that "
            "reaches the network, which matters where there is no network."
        ),
    )

    encoder_learning_rate: float | None = Field(
        None,
        gt=0,
        description=(
            "A separate, usually smaller learning rate for the layers that arrived "
            "pretrained. The full-resolution stage added in front of them is ours and "
            "starts random, so it trains at the optimizer's own rate. Unset means one "
            "rate for the whole network."
        ),
    )
    freeze_encoder: int = Field(
        0,
        ge=0,
        description=(
            "Keep the pretrained layers fixed for this many epochs, then train them. "
            "Lets the random decoder settle before its gradients reach weights worth "
            "keeping. A number at or above `epochs` freezes them for the whole run."
        ),
    )

    @property
    def load_shape(self) -> tuple[int, ...]:
        """The size an image is read at, before any tiling."""
        if self.splits is None:
            return tuple(self.input_shape)
        return tuple(s * i for s, i in zip(self.splits, self.input_shape, strict=True))

    @property
    def tiles_per_image(self) -> int:
        """How many training samples one source image yields."""
        if self.splits is None:
            return 1
        return math.prod(self.splits)

    @model_validator(mode="after")
    def tiling_and_whole_mask_metrics_do_not_mix(self):
        """A skeleton is a property of a whole mask, and a tile does not have one.

        `clDice` measures whether a structure is connected. Cut the image into tiles and
        every vessel is severed at four edges, so the number comes back low for a model
        that is perfectly connected - and comes back *silently*, which is worse than not
        being able to ask. Dice has no such problem: its pieces sum.

        Refused here rather than in the trainer because this is knowable before anything is
        read, and because an epoch's `val_cldice` would be wrong the same way.
        """
        if self.splits is None:
            return self
        whole = [m.name for m in self.metrics if m.name in _WHOLE_MASK_METRICS]
        if whole:
            raise ValueError(
                f"{', '.join(whole)} cannot be measured on a tiled run: it reads the shape "
                "of a whole mask, and `splits` hands the model pieces, so every structure "
                "is cut at the tile edges. Drop `splits` and give the model the whole "
                f"image, or score {whole[0]} separately on reassembled predictions."
            )
        return self

    @model_validator(mode="after")
    def splits_match_rank(self):
        """Tiling is rank-aware for the same reason input_shape is: a 2D pair cannot
        grow into a 3D triple without breaking every caller. Grid tiling in 2D and patch
        sampling in 3D are the same idea at different ranks."""
        if self.splits is None:
            return self
        if len(self.splits) != self.rank:
            raise ValueError(
                f"splits has {len(self.splits)} entries but input_shape is {self.rank}D; "
                f"they must match"
            )
        bad = [n for n in self.splits if n < 1]
        if bad:
            raise ValueError(f"every entry in splits must be at least 1; got {tuple(self.splits)}")
        if all(n == 1 for n in self.splits):
            raise ValueError(
                "splits of all ones does nothing; leave it unset to resize instead of tiling"
            )
        return self

    @model_validator(mode="after")
    def deep_supervision_needs_depth(self):
        if not self.deep_supervision:
            return self
        if self.blocks < 2:
            raise ValueError("deep_supervision needs at least 2 blocks to supervise")
        if self.architecture is not Architecture.U_NET_PLUS_PLUS:
            # Deep supervision reads the intermediate nodes X[0][j], which only the
            # nested architecture produces. Catching it here beats a shape error later.
            raise ValueError(
                f"deep_supervision requires architecture 'u_net_plus_plus'; "
                f"'{self.architecture.value}' has no intermediate outputs to supervise"
            )
        return self

    @model_validator(mode="after")
    def divisible_by_pooling(self):
        """Every block halves each spatial dimension; a size that will not halve cleanly
        produces shape mismatches deep in the decoder, which is a miserable error to
        debug. Catching it here costs nothing."""
        divisor = 2**self.blocks
        bad = [size for size in self.input_shape if size % divisor]
        if bad:
            raise ValueError(
                f"with blocks={self.blocks} every spatial dimension must be divisible by "
                f"{divisor}; input_shape={tuple(self.input_shape)} is not"
            )
        return self

    def weights_fingerprint(self) -> dict:
        # Without `n_class`: the data decides how many classes there are, so the model has
        # nothing to say about it. The engine contributes it through `_class_fingerprint`,
        # along with the colormap or labels it came from.
        return {**super().weights_fingerprint(), "blocks": self.blocks, "filters": self.filters}

    @property
    def monitorable(self) -> set[str]:
        """The loss, plus every metric this model was asked for."""
        return (
            super().monitorable
            | {f"val_{m.name}" for m in self.metrics}
            | {f"train_{m.name}" for m in self.metrics}
        )

    @model_validator(mode="after")
    def pretrained_needs_an_encoder(self):
        """`pretrained` on its own has nothing to load into: the built-in encoder is ours
        and no one published weights for it. Silently ignoring the flag would leave
        someone believing they had transfer learning when they had none."""
        if self.pretrained and self.encoder is None:
            raise ValueError(
                "pretrained=true needs `encoder` to say which backbone to load, e.g. "
                "encoder='resnet34'; the built-in encoder has no published weights"
            )
        return self

    @model_validator(mode="after")
    def encoder_settings_need_an_encoder(self):
        """Both of these address one problem - a random decoder's gradients arriving at
        transferred weights - so neither means anything without transferred weights."""
        for field, value in (
            ("encoder_learning_rate", self.encoder_learning_rate),
            ("freeze_encoder", self.freeze_encoder or None),
        ):
            if value is not None and self.encoder is None:
                raise ValueError(
                    f"{field} needs `encoder` to name a backbone; there is nothing "
                    f"transferred for it to apply to"
                )
        return self

    @model_validator(mode="after")
    def encoders_are_two_dimensional(self):
        """Refused here rather than at build time so a 3D spec fails while it is being
        read, before anything is downloaded or a GPU is touched."""
        if self.encoder is not None and self.rank != 2:
            raise ValueError(
                f"encoder='{self.encoder}' is a 2D backbone but input_shape is "
                f"{self.rank}D ({tuple(self.input_shape)}); ImageNet is images, so there "
                f"is nothing to transfer to a volume. Leave `encoder` unset - the "
                f"built-in encoder works at both ranks."
            )
        return self
