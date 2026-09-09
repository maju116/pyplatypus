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
    def __init__(self, steps: list[AugmentationStep]):
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
        self._pipeline = albumentations.Compose(built)
        self.steps = tuple(step.name for step in steps)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray | None]:
        if mask is None:
            return self._pipeline(image=image)["image"], None
        out = self._pipeline(image=image, mask=mask)
        return out["image"], out["mask"]


def build_augmenter(steps: list[AugmentationStep] | None, rank: int = 2
                    ) -> Augmenter | None:
    """None when there is nothing to do, which keeps the caller free of special cases."""
    if not steps:
        return None
    if rank != 2:
        # albumentations 2.x does ship 3D transforms (CenterCrop3D, CubicSymmetry and
        # friends) through a different call signature, so this is a v0.1 scope line
        # rather than a missing capability.
        raise AugmentationError(
            f"augmentation is implemented for 2D only; this spec is {rank}D"
        )
    return AlbumentationsAugmenter(steps)
