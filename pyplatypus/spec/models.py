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
    def not_fitting_needs_weights(self):
        if not self.fit and self.weights is None:
            raise ValueError("fit=false only makes sense together with weights")
        return self
