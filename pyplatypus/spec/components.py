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


LossSpec = Annotated[
    IouLoss | DiceLoss | CceLoss | CceDiceLoss | FocalLoss | TverskyLoss | FocalTverskyLoss | ComboLoss | LovaszLoss,
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


class CsvLogger(_Callback):
    name: Literal["csv_logger"] = "csv_logger"
    path: str


class TerminateOnNaN(_Callback):
    name: Literal["terminate_on_nan"] = "terminate_on_nan"


CallbackSpec = Annotated[
    EarlyStopping | ModelCheckpoint | ReduceLrOnPlateau | CsvLogger | TerminateOnNaN,
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
