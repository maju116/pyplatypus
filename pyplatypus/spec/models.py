"""One model to train.

`input_shape` carries the spatial size and nothing else; `rank` is read off its length.
That is PLAN.md rule 1, and it is why there is no `net_h`/`net_w` pair here - a pair
cannot become a triple without an API break.

`ModelSpec` is what every task's model has: a name, a size, how long to train, what to
train with, and whether to train at all. `EncoderModel` is the layer below that: what a
model has if it reduces an image to features, which is segmentation and classification
and not detection - YOLOv3 carries its own backbone and none of these knobs reach it. What it deliberately does *not* have is `loss`
and `metrics`. Each task that has a choice declares its own: segmentation picks a loss
from every one defined over classes, which is what `the_mask_needs_a_loss_over_classes`
below is for, and its metrics from the overlap menu; classification picks from a narrower
set of each, because an overlap is not what a label is scored by. YOLOv3 has one composite
objective that is part of the architecture, and mean average precision is not one option
among several. This paragraph said *a menu of ten and four* until 2026-10-27 and both
numbers were stale - the metrics had been four until `surface_dice` shipped and the losses
ten until two more did - so it names the rule now rather than a count nothing holds.
`available_losses()` and `available_metrics()` answer per task and are generated from
these unions, which is where a reader should get a number from. Offering a choice that does not exist is worse than offering
none - see the architecture comparison in the README for the same argument about weights.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import ClassVar

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
        """Quantities a callback may watch, given what this model reports.

        `foreground` is the share of the prediction that is not background, reported every
        epoch beside the loss. It is not a metric - nothing is compared against the truth -
        but it is a number an epoch produces, so a callback may watch it and `when` may
        have a condition on it.
        """
        return {"train_loss", "val_loss", "train_foreground", "val_foreground"}

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

    #: The callbacks that compute the learning rate from the epoch rather than from how
    #: the run is going. One of them at most, and none of them beside the reactive one -
    #: see `one_hand_on_the_learning_rate` for the measurement that settles it.
    SCHEDULES: ClassVar[frozenset[str]] = frozenset(
        {"cosine_annealing", "linear_annealing", "step_decay"}
    )

    @model_validator(mode="after")
    def one_hand_on_the_learning_rate(self):
        """At most one schedule, and no schedule beside `reduce_lr_on_plateau`.

        A schedule recomputes the rate from the optimiser's initial value at the end of
        every epoch. Two of them write the same number and the last one in the list wins by
        accident; a schedule beside the plateau callback **erases the plateau's cuts**,
        measured rather than supposed - with both configured the rate follows the cosine
        exactly, 0.0976, 0.0905, 0.0794 from 0.1, where the plateau alone reaches 1e-6 over
        the same six epochs.

        `cosine_annealing`'s docstring claimed a run may want both for as long as it has
        existed. It was never true of the code, and this is the refusal that replaces it.
        """
        named = [callback.name for callback in self.callbacks]
        schedules = [name for name in named if name in self.SCHEDULES]
        if len(schedules) > 1:
            raise ValueError(
                f"these callbacks are two learning rate schedules ({', '.join(schedules)}) "
                "and both write the rate every epoch, so whichever is last in the list "
                "silently wins. Keep one."
            )
        if schedules and "reduce_lr_on_plateau" in named:
            raise ValueError(
                f"'{schedules[0]}' and 'reduce_lr_on_plateau' cannot be combined: the "
                "schedule recomputes the rate from the optimiser's initial value every "
                "epoch, which erases what the plateau callback cut. Keep the schedule for "
                "a planned decay, or the plateau callback to react to the run."
            )
        return self

    @model_validator(mode="after")
    def one_averager(self):
        """`ema` and `swa` both replace the model's weights when training ends.

        Two of them would replace it twice, in whichever order the list happened to have
        them, and the second would average nothing that the first had not already
        rewritten. They answer the same question with different weightings, so a run picks
        one.
        """
        averagers = [c.name for c in self.callbacks if c.name in ("ema", "swa")]
        if len(averagers) > 1:
            raise ValueError(
                f"these callbacks are two weight averagers ({', '.join(averagers)}) and "
                "both replace the model when training ends, so the second would average "
                "what the first had already written. Keep one: `swa` folds the tail "
                "uniformly from a declared start, `ema` weights recent steps more."
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
            # A condition is a watch with a threshold on it, so it answers to the same
            # rule: a key this model will never report would wait forever.
            for key in getattr(callback, "when", None) or {}:
                if key not in available:
                    raise ValueError(
                        f"callback '{callback.name}' has a condition on '{key}', which "
                        f"this model does not produce; available: "
                        f"{', '.join(sorted(available))}"
                    )
        return self

    @model_validator(mode="after")
    def not_fitting_needs_weights(self):
        if not self.fit and self.weights is None:
            raise ValueError("fit=false only makes sense together with weights")
        return self


class EncoderModel(ModelSpec):
    """A model built on a contracting path, its own or somebody else's.

    Every field here is about **the half of a network that reduces an image to
    features**, which is the half two tasks share: a segmentation model hands those
    features to a decoder, a classification model pools them and asks one question of
    the result. Written once rather than twice, because a second copy of a published
    `description=` is the shape §3F of the workbench keeps a table of - and a second
    copy of `encoders_are_two_dimensional` would be a *mechanism* in two places, which
    is worse than a sentence.

    **What is deliberately not here: `blocks`, `filters` and `block_width`.** They are
    the most contracting-path-shaped fields of all, and their prose is the one thing
    about them that is task-specific - `blocks` says `input_shape` must divide by
    2^blocks, which is the decoder's requirement and false where the features are
    pooled instead of upsampled, and all three record what a published weights file
    fills in, which is true of the task that has published weights. Generalising those
    sentences would put a decoder's constraint on a classifier's page or drop the
    adoption note from the page a reader checks it on, so each task declares those
    three for itself. That is a second copy of a *declaration* with one copy of the
    mechanism, which is the trade this file is willing to make.
    """

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
            "pretrained. Whatever is ours trains at the optimizer's own rate instead - "
            "the decoder and the full-resolution stage in front of the backbone for a "
            "segmentation model, the pooled head for a classifier - because those start "
            "random. Unset means one rate for the whole network."
        ),
    )
    freeze_encoder: int = Field(
        0,
        ge=0,
        description=(
            "Keep the pretrained layers fixed for this many epochs, then train them. "
            "Lets the random part of the network settle - the decoder, or a classifier's "
            "head - before its gradients reach weights worth keeping. A number at or "
            "above `epochs` freezes them for the whole run."
        ),
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
        """Both of these address one problem - the gradients of whatever is ours and
        started random arriving at transferred weights - so neither means anything
        without transferred weights."""
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


class SegmentationModel(EncoderModel):
    #: Appended to the refusal of a continuous `loss`, by the subclass that has somewhere
    #: else to put one. Empty here, because a semantic model has one objective and no
    #: second place for it. `InstanceModel` overrides it so that refusal keeps naming both
    #: halves - the mask and the flow block - rather than only the half that was wrong.
    continuous_loss_elsewhere: ClassVar[str] = ""

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
    upsample: bool = Field(False, description="Upsample+conv instead of transposed convolution.")
    deep_supervision: bool = Field(
        False,
        description=(
            "Train every decoder depth rather than only the last, each at input resolution. "
            "`u_net_plus_plus` only, since it is the architecture with intermediate outputs "
            "to supervise."
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
            "unset to resize the whole image instead. Training then shuffles in windows "
            "rather than over every tile, so one decode serves all of a sample's tiles: "
            "measured on 2048-pixel retinas cut sixteen ways, the loader costs 254 ms a "
            "tile shuffled wholesale against 49 ms this way, and a batch still draws from "
            "several sources."
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
        """The loss, plus every metric an epoch reports - not every metric asked for.

        A whole-mask metric is measured on the reassembled image by `evaluate_cases`, never
        per batch, so a tiled run's epochs do not produce `val_cldice` or `val_surface_dice`
        and a callback cannot watch either. Saying so here is what makes
        `callbacks_watch_something_that_exists` refuse that pairing, with a message listing
        what *is* available - rather than a second validator saying the same thing in its
        own words.

        **Which metrics those are is asked of the component**, through `whole_mask`. It was
        a frozenset of names here until `0.8.0a27`, and `surface_dice` shipped without being
        added to it: the trainer asked the built metric's `accumulates` and this asked a
        list, so a tiled run accepted a callback watching a number its epochs would never
        put in `state.logs`. A flag the two halves share cannot go out of step that way, and
        a test pairs it with `accumulates` so that it cannot be set wrongly either.
        """
        reported = [m for m in self.metrics if self.splits is None or not m.whole_mask]
        return (
            super().monitorable
            | {f"val_{m.name}" for m in reported}
            | {f"train_{m.name}" for m in reported}
            # The worst case of the split, for the metrics whose counts pool into one -
            # validation only, because the training loader is shuffled and a case has no
            # identity there. This is the number `summarise_cases` tells every reader to
            # look at, and until it was in the logs no callback could watch it.
            | {f"val_{m.name}_worst" for m in reported if not m.whole_mask}
        )

    @model_validator(mode="after")
    def the_mask_needs_a_loss_over_classes(self):
        """A continuous loss over a one-hot target, refused at every segmentation task.

        **This validator used to live on `InstanceModel`, which is the subclass**, while
        the thing it protects - `loss` scoring a target of classes - belongs to this one.
        So the semantic task accepted `loss: mse` and ran: `engine.py` builds the loader
        with `with_foreground=loss_needs_mask(model.loss)` and the trainer hands a third
        tensor to any loss that asks for one, so nothing downstream refused it either.

        **And the number it reported was the good one, which is why this is refused rather
        than warned about.** `where` defaults to `foreground`, and for a semantic target
        the foreground is "not the background channel" - so the objective was evaluated
        only inside the labelled objects and the background was unconstrained. Measured on
        eight 32x32 samples whose object is where the picture is bright, 80 epochs, a task
        `cce` solves exactly: `mse` reported a validation loss of 0.0014 against the
        control's 0.0865 and a Dice of 0.54 against 1.00. The run looks like the best of
        the three and the mask is half missing. `pyplatypus#223`.

        The admissibility is read off the `continuous` flag on the loss itself rather than
        a list of names kept here, so a fourth regression loss is refused without this file
        being edited. The other half of the question is
        `Flows.a_vector_field_needs_a_continuous_loss`, which refuses an overlap loss on
        the vector channels.
        """
        if not self.loss.continuous:
            return self
        # Only said when it is true of what was written. `where` defaults to `foreground`
        # and almost nobody writes it, so this clause is usually the one that explains the
        # measurement - but quoting a setting the caller did not choose is the kind of
        # message that sends somebody to change the wrong line.
        masked = getattr(self.loss, "where", None) == "foreground"
        consequence = (
            " and `where: foreground`, which is what it is unless you said otherwise, "
            "would evaluate that only inside the objects and leave the background "
            "unconstrained - a small loss for a mask that is half missing"
            if masked
            else ""
        )
        raise ValueError(
            f"loss '{self.loss.name}' scores a continuous target, and `loss` is the "
            f"objective on the class channels - the mask. Run as it stands it would score "
            f"raw output against 0 and 1 with no activation between them"
            f"{consequence}. Use an overlap or cross-entropy loss here"
            f"{self.continuous_loss_elsewhere}."
        )
