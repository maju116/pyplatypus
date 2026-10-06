"""Losses, metrics, optimisers, callbacks, augmentation.

Each family is a discriminated union keyed on `name`. The old package typed these as
`Any`, which meant a misspelled loss or a nonsense parameter sailed through validation
and failed much later, deep inside training. Here `{"name": "focal", "gamm": 1}` fails
immediately and says which key it did not recognise.

The YAML for one of these reads:

    loss:
      name: focal
      gamma: 2.0
    metrics:
      - name: iou
      - name: tversky
        alpha: 0.7
"""

from __future__ import annotations

from difflib import get_close_matches
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from pyplatypus.spec.common import SpecModel

# --------------------------------------------------------------------------- losses
# Every loss here reduces over "all dimensions except batch and channel", which is what
# lets the same implementation serve 2D and 3D untouched (PLAN.md, rule 3).


class _Loss(SpecModel):
    pass


class IouLoss(_Loss):
    name: Literal["iou"] = "iou"
    smooth: float = Field(1.0, gt=0)


class DiceLoss(_Loss):
    name: Literal["dice"] = "dice"
    smooth: float = Field(1.0, gt=0)


class CceLoss(_Loss):
    name: Literal["cce"] = "cce"
    label_smoothing: float = Field(0.0, ge=0, lt=1)


class CceDiceLoss(_Loss):
    name: Literal["cce_dice"] = "cce_dice"
    cce_weight: float = Field(0.5, ge=0, le=1)
    smooth: float = Field(1.0, gt=0)


class FocalLoss(_Loss):
    name: Literal["focal"] = "focal"
    gamma: float = Field(2.0, ge=0)
    alpha: float | None = Field(None, ge=0, le=1)


class TverskyLoss(_Loss):
    name: Literal["tversky"] = "tversky"
    alpha: float = Field(0.5, ge=0, le=1)
    smooth: float = Field(1.0, gt=0)

    @property
    def beta(self) -> float:
        """Tversky's two weights sum to one; storing both invites them not to."""
        return 1.0 - self.alpha


class FocalTverskyLoss(_Loss):
    name: Literal["focal_tversky"] = "focal_tversky"
    alpha: float = Field(0.5, ge=0, le=1)
    gamma: float = Field(1.0, gt=0)
    smooth: float = Field(1.0, gt=0)


class ComboLoss(_Loss):
    name: Literal["combo"] = "combo"
    alpha: float = Field(0.5, ge=0, le=1)
    ce_ratio: float = Field(0.5, ge=0, le=1)


class LovaszLoss(_Loss):
    name: Literal["lovasz"] = "lovasz"
    per_image: bool = False


#: Everything but `boundary`, which takes one of these as its region term. Spelled out
#: rather than recursive: a boundary loss inside a boundary loss would add two surface
#: terms with no region term holding either of them down, which is the one combination
#: that cannot work.
RegionLossSpec = Annotated[
    IouLoss | DiceLoss | CceLoss | CceDiceLoss | FocalLoss | TverskyLoss | FocalTverskyLoss | ComboLoss | LovaszLoss,
    Field(discriminator="name"),
]


class BoundaryLoss(_Loss):
    """A region loss plus a term that knows how far a wrong voxel is from the truth.

    Dice and IoU count a voxel the same wherever it sits, which is why a model can reach
    0.88 on Dice while its volumes run a fifth too large: the overshoot is all at the
    boundary, and for a small lesion the boundary is most of the object. This adds
    `mean(phi * p)`, where `phi` is the signed distance to the truth's boundary.

    ```yaml
    loss:
      name: boundary
      region: {name: focal, gamma: 2.0}
      alpha: 0.5
    ```

    `alpha` weights the region term, `1 - alpha` the surface term. It cannot be 1, which
    would leave the surface term doing nothing and the name lying, and it cannot be 0: the
    surface term alone has no notion of how much of the object was found and is minimised
    by a confident prediction deep inside a shrunken one.

    **What it buys and what it costs, measured.** Against Dice alone on lesions with
    ambiguous edges, three seeds, at the default `alpha`: the systematic volume bias goes
    from -5.18% to -0.06%, and the per-case absolute volume error goes the wrong way, from
    16.90% to 22.17%. Dice itself falls by 0.011. So it trades *accuracy per case* for
    *being unbiased over a series* - and which of those you want is the question being
    asked. "Is there a lesion" wants the overlap; "has it grown since March" wants a volume
    that is not systematically wrong, and no single number answers both.

    The bias improvement is suggestive rather than established on three seeds, because the
    baseline's own bias wanders by ±3.11%. The costs are established.

    **It also costs time.** The signed distance transform runs per sample in the loader's
    workers - measured at 5.3 ms for a 256x256 image, 23.4 ms for a 64x64x32 volume and
    709.8 ms at 128^3 - so on large volumes give the loader workers, or the transform
    becomes the training.
    """

    name: Literal["boundary"] = "boundary"
    region: RegionLossSpec = Field(
        default_factory=lambda: DiceLoss(),
        description="The overlap term this is added to. Dice unless you say otherwise.",
    )
    alpha: float = Field(
        0.9,
        gt=0,
        lt=1,
        description=(
            "How much of the objective is the region term, `1 - alpha` the surface term.\n\n"
            "**0.9 because 0.5 was measured and is unusable.** Three seeds on lesions with "
            "ambiguous edges, 60 epochs:\n\n"
            "```\n"
            "                 Dice             volume bias       volume |error|\n"
            "dice alone       0.9525 ±0.0027   -5.18% ±3.11%     16.90% ±0.90%\n"
            "boundary a=0.5   0.9164 ±0.0240  +32.22% ±28.28%    42.95% ±19.12%\n"
            "boundary a=0.9   0.9411 ±0.0093   -0.06% ±4.15%     22.17% ±2.88%\n"
            "```\n\n"
            "At 0.5 the surface term has half the objective while the prediction is still "
            "random, which is where it is known to be unstable - the seed spread is six "
            "times the baseline's and every number is worse. Kervadec schedules this "
            "downwards from near 1 rather than fixing it; 0.9 is the closest a single "
            "number comes, and the spec deliberately has no schedule to hide behind.\n\n"
            "Excludes both ends: at 1 the surface term is absent and the name is a lie, "
            "and at 0 nothing measures how much of the object was found."
        ),
    )


LossSpec = Annotated[
    IouLoss | DiceLoss | CceLoss | CceDiceLoss | FocalLoss | TverskyLoss | FocalTverskyLoss | ComboLoss | LovaszLoss | BoundaryLoss,
    Field(discriminator="name"),
]

# -------------------------------------------------------------------------- metrics


class _Metric(SpecModel):
    pass


class _Overlap(_Metric):
    smooth: float = Field(
        1.0, ge=0,
        description=(
            "Zero is allowed here, unlike in the losses. Smoothing inflates a reported "
            "score: a class absent from an image scores a perfect 1.0 with smooth>0. "
            "Use 0 for an honest number, at the cost of NaN on absent classes."
        ),
    )
    include_background: bool = Field(
        True,
        description=(
            "Average over every class, background included. In medical images the "
            "background is often 95% of the pixels, so leaving it in flatters the score; "
            "papers usually report foreground only. Set false for that."
        ),
    )


class IouMetric(_Overlap):
    name: Literal["iou"] = "iou"


class DiceMetric(_Overlap):
    name: Literal["dice"] = "dice"


class TverskyMetric(_Overlap):
    name: Literal["tversky"] = "tversky"
    alpha: float = Field(0.5, ge=0, le=1)


MetricSpec = Annotated[IouMetric | DiceMetric | TverskyMetric,
                       Field(discriminator="name")]

# ----------------------------------------------------------------------- optimisers
# torch's set, not TensorFlow's. Ftrl is gone because torch has no Ftrl.


class _Optimizer(SpecModel):
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.0, ge=0)


class Adam(_Optimizer):
    name: Literal["adam"] = "adam"
    beta_1: float = Field(0.9, ge=0, lt=1)
    beta_2: float = Field(0.999, ge=0, lt=1)
    eps: float = Field(1e-8, gt=0)
    amsgrad: bool = False


class AdamW(_Optimizer):
    name: Literal["adamw"] = "adamw"
    beta_1: float = Field(0.9, ge=0, lt=1)
    beta_2: float = Field(0.999, ge=0, lt=1)
    eps: float = Field(1e-8, gt=0)
    weight_decay: float = Field(1e-2, ge=0)


class Sgd(_Optimizer):
    name: Literal["sgd"] = "sgd"
    momentum: float = Field(0.0, ge=0)
    nesterov: bool = False

    @model_validator(mode="after")
    def nesterov_needs_momentum(self):
        if self.nesterov and self.momentum == 0:
            raise ValueError("nesterov requires momentum greater than 0")
        return self


class RmsProp(_Optimizer):
    name: Literal["rmsprop"] = "rmsprop"
    alpha: float = Field(0.99, ge=0, lt=1)
    momentum: float = Field(0.0, ge=0)
    eps: float = Field(1e-8, gt=0)


class Adagrad(_Optimizer):
    name: Literal["adagrad"] = "adagrad"
    lr_decay: float = Field(0.0, ge=0)
    eps: float = Field(1e-10, gt=0)


class Adadelta(_Optimizer):
    name: Literal["adadelta"] = "adadelta"
    rho: float = Field(0.9, ge=0, lt=1)
    eps: float = Field(1e-6, gt=0)


class Adamax(_Optimizer):
    name: Literal["adamax"] = "adamax"
    beta_1: float = Field(0.9, ge=0, lt=1)
    beta_2: float = Field(0.999, ge=0, lt=1)
    eps: float = Field(1e-8, gt=0)


class NAdam(_Optimizer):
    name: Literal["nadam"] = "nadam"
    beta_1: float = Field(0.9, ge=0, lt=1)
    beta_2: float = Field(0.999, ge=0, lt=1)
    eps: float = Field(1e-8, gt=0)


OptimizerSpec = Annotated[
    Adam | AdamW | Sgd | RmsProp | Adagrad | Adadelta | Adamax | NAdam,
    Field(discriminator="name"),
]

# ------------------------------------------------------------------------ callbacks
# torch ships no callbacks; these are our own, specified now and implemented in step 5.


class _Callback(SpecModel):
    pass


class EarlyStopping(_Callback):
    name: Literal["early_stopping"] = "early_stopping"
    monitor: str = "val_loss"
    patience: int = Field(10, ge=1)
    min_delta: float = Field(0.0, ge=0)
    restore_best: bool = True


class ModelCheckpoint(_Callback):
    name: Literal["model_checkpoint"] = "model_checkpoint"
    path: str
    monitor: str = "val_loss"
    save_best_only: bool = True


class ReduceLrOnPlateau(_Callback):
    name: Literal["reduce_lr_on_plateau"] = "reduce_lr_on_plateau"
    monitor: str = "val_loss"
    factor: float = Field(0.1, gt=0, lt=1)
    patience: int = Field(5, ge=1)
    min_lr: float = Field(0.0, ge=0)


class CosineAnnealing(_Callback):
    """Decay the learning rate along a cosine, from its initial value to `min_lr`.

    No `monitor`: it is a function of how far through the run you are, not of how the run
    is going. That is the difference from `reduce_lr_on_plateau`, and a run may want both.
    """

    name: Literal["cosine_annealing"] = "cosine_annealing"
    min_lr: float = Field(0.0, ge=0, description="The rate at the last epoch.")
    epochs: int | None = Field(
        None,
        ge=1,
        description=(
            "How many epochs to spread the decay over. Unset means the model's `epochs`, "
            "which is what you want: a cosine that ends where the run ends. Setting it "
            "shorter parks the rate at `min_lr` for the remaining epochs, and longer "
            "stops the run part way down the curve."
        ),
    )


class Swa(_Callback):
    """Average the weights over the last part of the run instead of taking the final ones.

    Stochastic weight averaging. Past `start`, every epoch's weights are folded into a
    running average and at the end that average becomes the model - the claim being that a
    point in the middle of a flat region generalises better than the corner the last epoch
    stopped in.

    **Batch-normalisation statistics are recomputed afterwards**, by a pass over the
    training data. They have to be: an averaged weight tensor *inherits* the statistics of
    whichever epoch was last rather than averaging them, so without the pass the model is
    evaluated under the wrong normalisation and scores far worse than it should, with
    nothing to say why. It is the step an SWA implementation leaves out.

    **What it is worth, measured.** Three seeds, 60 epochs, lesions with ambiguous edges:

    ```
                 Dice             volume bias        |volume error|
    plain        0.9639 ±0.0100   +1.71% ±12.18%     12.60% ±4.19%
    swa 0.75     0.9667 ±0.0087   +2.08% ±2.15%      12.38% ±4.31%
    swa 0.5      0.9640 ±0.0085   +2.78% ±3.44%      13.75% ±4.21%
    ```

    **Dice does not move and neither does the per-case error** - both are inside the seed
    noise. What moves is *reproducibility*: the volume bias of the plain runs was +4.5%,
    +12.2% and -11.6% across seeds, and with averaging it was -0.1%, +4.2% and +2.1%. A
    spread 5.7 times tighter, and paired per seed the distance from unbiased improved on
    all three, by 7.34% at 4.9 standard errors.

    Which is what averaging is supposed to do and not what one looks for in a table of
    scores: a point in the middle of a flat region is the same point whichever corner the
    last epoch wandered into. Read it as "the same model twice" rather than "a better
    model".

    No `monitor`: like `cosine_annealing` this is a function of how far through the run you
    are, not of how the run is going.
    """

    name: Literal["swa"] = "swa"
    start: float = Field(
        0.75,
        gt=0,
        le=1,
        description=(
            "The fraction of the run after which averaging begins - 0.75 folds the last "
            "quarter. A fraction rather than an epoch so that it survives a change to "
            "`epochs`, and the final epoch is always folded, so 1.0 averages one set of "
            "weights rather than none."
        ),
    )
    learning_rate: float | None = Field(
        None,
        gt=0,
        description=(
            "Hold the rate at this value once averaging begins. Unset leaves whatever the "
            "run was doing.\n\n"
            "Worth setting, and the reason is the whole mechanism: averaging is only worth "
            "something while the weights are still moving. A rate that has decayed towards "
            "zero produces a set of nearly identical snapshots, and their average is the "
            "last one."
        ),
    )


class CsvLogger(_Callback):
    name: Literal["csv_logger"] = "csv_logger"
    path: str


class TerminateOnNaN(_Callback):
    name: Literal["terminate_on_nan"] = "terminate_on_nan"


CallbackSpec = Annotated[
    EarlyStopping | ModelCheckpoint | ReduceLrOnPlateau | CosineAnnealing | Swa
    | CsvLogger | TerminateOnNaN,
    Field(discriminator="name"),
]

# --------------------------------------------------------------------- augmentation
# Deliberately not a discriminated union. albumentations ships 131 transforms with dozens
# of parameters each, and mirroring them here would be a second, always slightly stale
# copy of someone else's API. A hardcoded list proved that point immediately: written
# against albumentations 1.1 it already named `Flip`, which 2.x removed.
#
# So the names are read from the installed albumentations, and the parameters are checked
# by albumentations itself when the pipeline is built. The import is lazy and cached, so
# a spec with no augmentation never pays for it.
#
# PLAN.md rule 4 said albumentations was "today's 2D implementation" and 3D would need a
# different backend. That turns out to be too pessimistic: albumentations 2.x carries 3D
# transforms (CenterCrop3D, CoarseDropout3D, CubicSymmetry), so the same backend may well
# serve both ranks. The interface stays either way.


@lru_cache(maxsize=2)
def available_transforms(rank: int = 2) -> frozenset[str]:
    """Transform names the installed albumentations actually offers.

    With `rank=3`, only those that can transform a volume. That list has to be found by trying
    rather than read from anywhere: albumentations supports volumes unevenly, and a transform
    that cannot raises from inside itself - `GaussNoise` comes back as `KeyError: 'images'`. Of
    the transforms in 2.0.8, 71 work on volumes and 33 do not.

    Empty when albumentations is missing, in which case name checking is skipped and the
    backend reports the problem later - better than refusing a spec the user cannot fix.
    """
    try:
        import albumentations
    except ImportError:  # pragma: no cover - exercised only without the optional backend
        return frozenset()

    names = frozenset(
        name for name in dir(albumentations)
        if name[:1].isupper() and not name.startswith(("Base", "Basic", "Dual"))
    )
    if rank != 3:
        return names
    return frozenset(name for name in names if _transforms_volumes(albumentations, name))


def _transforms_volumes(albumentations, name: str) -> bool:
    """Whether this transform can be ruled out for volumes. Tried, because nothing declares it.

    A transform that needs arguments - `CenterCrop3D` wants a size, and it is one of the
    3D-native ones - cannot be probed with defaults, and is kept rather than dropped. "I could
    not check" is not "it does not work", and dropping those would have hidden exactly the
    transforms written for volumes. The authoritative check happens when the pipeline is built,
    against the parameters the user actually gave.

    Warnings are silenced: albumentations advises about aliases and slow implementations, and a
    listing that emits twenty warnings is a listing nobody reads.
    """
    import warnings

    import numpy as np

    cls = getattr(albumentations, name, None)
    if cls is None:
        return False

    probe = np.zeros((4, 8, 8, 1), dtype=np.float32)
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            transform = cls(p=1)
        except Exception:  # noqa: BLE001 - needs arguments, so it cannot be ruled out here
            return True
        try:
            albumentations.Compose([transform])(volume=probe, mask3d=mask)
        except Exception:  # noqa: BLE001
            return False
    return True


class AugmentationStep(SpecModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def known_transform(cls, value: str) -> str:
        known = available_transforms()
        if not known or value in known:
            return value
        close = get_close_matches(value, known, n=3, cutoff=0.6)
        if not close:
            # get_close_matches is case-sensitive; a pure casing slip deserves a hit too.
            lowered = value.lower()
            close = [name for name in known if name.lower() == lowered]
        suggestion = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"unknown augmentation '{value}'.{suggestion}")
        return value
