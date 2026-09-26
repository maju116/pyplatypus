"""Augmentation, behind an interface.

PLAN.md rule 4: the spec names transforms, it does not import a library. albumentations
is the backend; swapping or adding one later must not touch a single spec.

Masks travel through here as class indices, never as one-hot or RGB, so a geometric
transform moves labels around instead of blending them into colours that match no class.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pyplatypus.errors import PlatypusError
from pyplatypus.spec.components import AugmentationStep


class AugmentationError(PlatypusError):
    kind = "augmentation_error"


class Augmenter(Protocol):
    """Takes an image and its class-index mask, returns both transformed."""

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray | None]: ...


class AlbumentationsAugmenter:
    """albumentations, for images and for volumes.

    Volumes go in through a different pair of keys - `volume` and `mask3d` rather than `image`
    and `mask` - and support for them is uneven. Of the transforms albumentations 2.0.8 offers,
    71 accept a volume (Affine, ElasticTransform, D4, CubicSymmetry, the flips, the blurs) and
    33 raise something internal: `GaussNoise` and `ChannelDropout` come back as `KeyError:
    'images'`, which is not a sentence anyone can act on. So every step is tried once against a
    tiny probe volume while the pipeline is being built, and a transform that cannot do volumes
    is reported by name now rather than as a library traceback forty minutes into training.
    """

    def __init__(self, steps: list[AugmentationStep], rank: int = 2):
        try:
            import albumentations
        except ImportError:  # pragma: no cover
            raise AugmentationError(
                "augmentation was requested but albumentations is not installed"
            ) from None

        built = []
        for step in steps:
            factory = getattr(albumentations, step.name, None)
            if factory is None:
                raise AugmentationError(
                    f"albumentations {albumentations.__version__} has no transform "
                    f"'{step.name}'"
                )
            try:
                built.append(factory(**step.params))
            except TypeError as error:
                raise AugmentationError(
                    f"'{step.name}' rejected its parameters {step.params}: {error}"
                ) from None
        self.rank = rank
        self._pipeline = albumentations.Compose(built)
        self.steps = tuple(step.name for step in steps)
        if rank == 3:
            _refuse_flat_transforms(albumentations, steps, built)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray | None]:
        image_key, mask_key = ("volume", "mask3d") if self.rank == 3 else ("image", "mask")
        if mask is None:
            return self._pipeline(**{image_key: image})[image_key], None
        out = self._pipeline(**{image_key: image, mask_key: mask})
        return out[image_key], out[mask_key]


def _refuse_flat_transforms(albumentations, steps: list[AugmentationStep], built: list) -> None:
    """Try each transform on a tiny volume, and name the one that cannot.

    Per step rather than on the whole pipeline: composing them and catching the failure would
    say that something in the list is unsupported without saying which, and a spec with eight
    transforms would leave the user bisecting.

    Every probe forces `p=1`, and that is not a detail. Most transforms default to `p=0.5`, so
    probing with the user's own parameters asks the question only half the time: the first
    version of this check passed or failed at random, and a transform that cannot do volumes
    would have got through roughly every other run to crash mid-epoch instead.
    """
    import warnings

    probe = np.zeros((4, 8, 8, 1), dtype=np.float32)
    probe[:, 2:6, 2:6] = 1.0
    probe_mask = np.zeros((4, 8, 8), dtype=np.uint8)
    probe_mask[:, 2:6, 2:6] = 1

    for step, transform in zip(steps, built, strict=True):
        try:
            with warnings.catch_warnings():
                # The probe is not the user's run; its warnings are about the probe.
                warnings.simplefilter("ignore")
                albumentations.Compose([_always(transform, step)])(
                    volume=probe, mask3d=probe_mask
                )
        except Exception as error:  # noqa: BLE001 - albumentations raises whatever it likes
            raise AugmentationError(
                f"'{step.name}' cannot transform a volume: "
                f"{type(error).__name__}: {error}. albumentations "
                f"{albumentations.__version__} supports volumes for most geometric transforms "
                "and not for every intensity one - the failure comes from inside the library, "
                "which is why it is reported here instead. Drop it, or use a transform that "
                "does: see available_transforms(rank=3)."
            ) from None


def _always(transform, step: AugmentationStep):
    """The same transform with `p=1`, for probing. Falls back to the original if it will not
    take a probability, which some wrappers do not."""
    try:
        return type(transform)(**{**step.params, "p": 1.0})
    except Exception:  # noqa: BLE001
        return transform


def build_augmenter(steps: list[AugmentationStep] | None, rank: int = 2
                    ) -> Augmenter | None:
    """None when there is nothing to do, which keeps the caller free of special cases."""
    if not steps:
        return None
    if rank not in (2, 3):
        raise AugmentationError(f"augmentation handles 2D and 3D; this spec is {rank}D")
    return AlbumentationsAugmenter(steps, rank=rank)
