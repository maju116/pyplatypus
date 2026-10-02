"""One model to train.

`input_shape` carries the spatial size and nothing else; `rank` is read off its length.
That is PLAN.md rule 1, and it is why there is no `net_h`/`net_w` pair here - a pair
cannot become a triple without an API break.
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


class SegmentationModel(SpecModel):
    name: str = Field(min_length=1, description="Unique within a spec; names the outputs.")
    architecture: Architecture = Architecture.U_NET

    input_shape: SpatialShape
    channels: int = Field(3, ge=1)
    n_class: int = Field(2, ge=2)

    blocks: int = Field(4, ge=1, le=8)
    filters: int = Field(16, ge=1)
    block_width: int = Field(2, ge=1, le=4, description="Convolutions per block.")
    dropout: float = Field(0.0, ge=0, lt=1)

    batch_normalization: bool = True
    separable_conv: bool = False
    spatial_dropout: bool = True
    upsample: bool = Field(
        False, description="Upsample+conv instead of transposed convolution."
    )
    deep_supervision: bool = False
    activation: Activation = Activation.RELU
    initialiser: Initialiser = Initialiser.HE_NORMAL

    loss: LossSpec = Field(default_factory=CceLoss)
    metrics: list[MetricSpec] = Field(default_factory=lambda: [IouMetric()])
    optimizer: OptimizerSpec = Field(default_factory=Adam)
    callbacks: list[CallbackSpec] = Field(default_factory=list)
    augmentation: list[AugmentationStep] | None = None

    epochs: int = Field(10, ge=1)
    batch_size: int = Field(8, ge=1)

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

    weights: str | None = Field(
        None,
        description="Registry name (e.g. 'dsbowl2018') or a path to a local checkpoint.",
    )
    fit: bool = Field(True, description="Set false to load weights and skip training.")

    @property
    def rank(self) -> int:
        """2 for images, 3 for volumes. Derived, never declared."""
        return len(self.input_shape)

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
        divisor = 2 ** self.blocks
        bad = [size for size in self.input_shape if size % divisor]
        if bad:
            raise ValueError(
                f"with blocks={self.blocks} every spatial dimension must be divisible by "
                f"{divisor}; input_shape={tuple(self.input_shape)} is not"
            )
        return self

    @property
    def monitorable(self) -> set[str]:
        """Quantities a callback may watch, given this model's metrics."""
        return {"train_loss", "val_loss"} | {f"val_{m.name}" for m in self.metrics} \
            | {f"train_{m.name}" for m in self.metrics}

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
        for field, value in (("encoder_learning_rate", self.encoder_learning_rate),
                             ("freeze_encoder", self.freeze_encoder or None)):
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

    @model_validator(mode="after")
    def not_fitting_needs_weights(self):
        if not self.fit and self.weights is None:
            raise ValueError("fit=false only makes sense together with weights")
        return self
