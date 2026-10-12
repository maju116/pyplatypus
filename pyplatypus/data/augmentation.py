"""Augmentation, behind an interface.

PLAN.md rule 4: the spec names transforms, it does not import a library. albumentations
is the backend; swapping or adding one later must not touch a single spec.

Masks travel through here as class indices, never as one-hot or RGB, so a geometric
transform moves labels around instead of blending them into colours that match no class.

**And never as a continuous target**, which is the decision `pyplatypus#210` settled rather
than a restatement of the line above. A flow field or a pair of horizontal and vertical
distances is a vector per voxel: a horizontal flip has to negate its x component, a rotation
has to rotate the vectors, and augmenting one as a mask produces a target that is
geometrically consistent and physically wrong - every arrow pointing somewhere else, with a
loss that still falls and a model that still trains.

Two measurements settled which way to fix that, both in `tests/data/test_vector_targets.py`
where a library bump re-runs them:

* **albumentations 2.0.8 never invents a label value.** Sixteen transforms, twenty seeded
  draws each, a label map of forty distinct ids: not one output value was absent from the
  input. So an *instance* label map can travel through here unharmed, and a continuous
  target can be built from it afterwards. Ids are lost where the geometry removes them -
  463 of 800 over twenty 32-pixel crops - which is the frame doing its job.
* **A crop moves an object's centre, and no correction can answer that.** Negating x after a
  horizontal flip is exact: 0.0% of foreground voxels disagree with a field rebuilt from the
  flipped labels, at 0.0 degrees. After a 32-pixel crop, **49.5% disagree, by a median of
  8.9 degrees and up to 180** - the surviving half of an object has a new centre, so the old
  arrows point at a place that is no longer there. There is no component to negate.

So the target is **rebuilt from the augmented label map**, never corrected, and that is
correct by construction for every transform rather than for an admissible list of them.
`loss_boundary` already establishes the precedent - it computes its distance transform in
the loader's workers because the alternative costs more than the training. It also means
tiling is covered: a tile is a crop, it happens before augmentation, and a 2048-pixel retina
cut into sixteen would otherwise carry sixteen fields built for the whole picture.

`_refuse_continuous_mask` is what keeps that decision from being quietly undone.
"""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np

from pyplatypus.detection.boxes import clip_boxes
from pyplatypus.errors import PlatypusError
from pyplatypus.spec.components import AugmentationStep


class AugmentationError(PlatypusError):
    kind = "augmentation_error"


class Augmenter(Protocol):
    """Takes an image and its class-index mask, returns both transformed.

    A mask here is one integer per voxel - a class index, or an instance label. Never a
    continuous target; see this module's docstring for why, and `_refuse_continuous_mask`
    for what happens if one is passed anyway.
    """

    def __call__(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]: ...


class BoxAugmenter(Protocol):
    """Takes an image and its boxes, returns both transformed."""

    def __call__(
        self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


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

    def __init__(
        self,
        steps: list[AugmentationStep],
        rank: int = 2,
        input_shape: tuple[int, ...] | None = None,
    ):
        try:
            import albumentations
        except ImportError:  # pragma: no cover
            raise AugmentationError(
                "augmentation was requested but albumentations is not installed"
            ) from None

        built = _build_steps(albumentations, steps)
        self.rank = rank
        self._pipeline = albumentations.Compose(built)
        self.steps = tuple(step.name for step in steps)
        if rank == 3:
            _refuse_flat_transforms(albumentations, steps, built, input_shape)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        image_key, mask_key = ("volume", "mask3d") if self.rank == 3 else ("image", "mask")
        if mask is not None:
            _refuse_continuous_mask(mask)
        if mask is None:
            return self._pipeline(**{image_key: image})[image_key], None
        out = self._pipeline(**{image_key: image, mask_key: mask})
        return out[image_key], out[mask_key]


def _refuse_continuous_mask(mask: np.ndarray) -> None:
    """Refuse a floating-point mask, because the failure it would cause is silent.

    This is the one guard in the package whose whole value is that it fires early. A flow
    field sent through as a mask comes back the right shape, the right dtype and the right
    range, with its arrows pointing the wrong way; the loss falls, the model trains and
    every number downstream is plausible. Nothing further on could tell - which is the same
    shape as a permuted channel, and `channels_from` refuses that one by construction too.

    A class index or an instance label is an integer, and every path in this package that
    reaches here sends one: `labels_to_classes`, `colours_to_classes` and
    `labels_to_regions` all return `int64`. So the check costs nothing it should not.
    """
    if np.issubdtype(np.asarray(mask).dtype, np.floating):
        raise AugmentationError(
            f"a mask of dtype {np.asarray(mask).dtype} is a continuous target, and a "
            "continuous target cannot be augmented as a mask: a horizontal flip would have "
            "to negate a flow field's x component and a crop moves the centre every arrow "
            "points at, which nothing can correct. Send the integer label map through here "
            "and build the target from what comes back - it is a function of the labels, so "
            "that is correct for every transform. Masks arrive as class indices or instance "
            "labels, which are integers."
        )


def _refuse_flat_transforms(
    albumentations,
    steps: list[AugmentationStep],
    built: list,
    input_shape: tuple[int, ...] | None = None,
) -> None:
    """Try each transform on a volume, and name the one that cannot.

    Per step rather than on the whole pipeline: composing them and catching the failure would
    say that something in the list is unsupported without saying which, and a spec with eight
    transforms would leave the user bisecting.

    Every probe forces `p=1`, and that is not a detail. Most transforms default to `p=0.5`, so
    probing with the user's own parameters asks the question only half the time: the first
    version of this check passed or failed at random, and a transform that cannot do volumes
    would have got through roughly every other run to crash mid-epoch instead.

    **At the model's own size when it is known.** This probed a fixed 4x8x8 volume until the
    box probe hit the same defect and made it obvious: a crop larger than the probe is refused
    with "crop size exceeds image dimensions", which is the check failing for its own reasons
    while reporting the user's transform as unsupported. A probe that fails for the wrong
    reason is indistinguishable from a real refusal.
    """
    depth, height, width = (
        (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
        if input_shape is not None and len(input_shape) == 3
        else (4, 8, 8)
    )
    probe = np.zeros((depth, height, width, 1), dtype=np.float32)
    inset_y, inset_x = max(height // 4, 1), max(width // 4, 1)
    probe[:, inset_y : height - inset_y, inset_x : width - inset_x] = 1.0
    probe_mask = np.zeros((depth, height, width), dtype=np.uint8)
    probe_mask[:, inset_y : height - inset_y, inset_x : width - inset_x] = 1

    for step, transform in zip(steps, built, strict=True):
        try:
            with warnings.catch_warnings():
                # The probe is not the user's run; its warnings are about the probe.
                warnings.simplefilter("ignore")
                albumentations.Compose([_always(transform, step)])(volume=probe, mask3d=probe_mask)
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


def build_augmenter(
    steps: list[AugmentationStep] | None, rank: int = 2, input_shape: tuple[int, ...] | None = None
) -> Augmenter | None:
    """None when there is nothing to do, which keeps the caller free of special cases.

    `input_shape` only affects the 3D probe, which is honest at the size the transforms
    will really see. Optional rather than required because the 2D path has nothing to
    probe, and because this is called from tests that have no model.
    """
    if not steps:
        return None
    if rank not in (2, 3):
        raise AugmentationError(f"augmentation handles 2D and 3D; this spec is {rank}D")
    return AlbumentationsAugmenter(steps, rank=rank, input_shape=input_shape)


#: How much of a box has to survive a crop for the box to be kept, as a fraction of its
#: original area. A **convention, not a measurement**, and the direction is the part worth
#: defending: the two failures are not symmetric. A box that keeps two pixels of a white
#: blood cell teaches the model that a two-pixel fragment is a whole cell, which produces
#: false positives across every image; dropping a heavily truncated object teaches it
#: nothing about that object, which is mild by comparison. Pascal VOC marks such objects
#: `truncated` for the same reason, and most recipes drop the worst of them.
#:
#: It only ever applies when a transform can remove part of the frame. With flips and
#: rotations nothing is ever lost, which is the whole of the BCCD recipe.
DEFAULT_MIN_VISIBILITY = 0.25


class AlbumentationsBoxAugmenter:
    """albumentations, for images and the boxes that label them.

    **Every geometric transform in albumentations 2.0.8 moves boxes with the pixels**, which
    was measured rather than assumed: a bright rectangle with the box labelling it, twenty
    seeded draws per transform, comparing the box that came back with the bounding box of the
    bright pixels in the output. Two groups, and the difference between them is real:

    * **exact, 0.00 px** - the flips, `RandomRotate90`, `Transpose`, `D4`, a pure
      translation, and the crops. These move pixels without resampling them.
    * **within 1.00 px** - `Affine` with a rotation or a scale, `Rotate`, `Perspective`,
      `GridDistortion`. These resample, and the two quantities stop being the same
      measurement: the box bounds the warped *corners* while the pixels are what survived
      interpolation above a threshold, and near a warped corner the object thins to a point.
      So the test asserts that the box still **covers** the object rather than equals it.

    That measurement is a property of the library, so it lives in the tests - where a version
    bump re-runs it - rather than as a probe on every pipeline. What *is* probed here is the
    cheap half: each step is tried once against an image the size of the real one, so a
    transform that cannot handle boxes at all is named now instead of forty minutes into
    training.

    The probe cannot check correspondence for an intensity transform, and that is a limit
    worth stating rather than working around: noise or a brightness change destroys the
    measurement the check would be made of. The first version tried, and reported `GaussNoise`
    as wrong by 46 pixels - a fault in the probe, not the transform.
    """

    def __init__(
        self,
        steps: list[AugmentationStep],
        *,
        input_shape: tuple[int, int],
        min_visibility: float = DEFAULT_MIN_VISIBILITY,
        seed: int | None = None,
    ):
        try:
            import albumentations
        except ImportError:  # pragma: no cover
            raise AugmentationError(
                "augmentation was requested but albumentations is not installed"
            ) from None

        built = _build_steps(albumentations, steps)
        self.min_visibility = float(min_visibility)
        self.steps = tuple(step.name for step in steps)
        self._params = albumentations.BboxParams(
            format="pascal_voc",  # absolute [x_min, y_min, x_max, y_max]
            label_fields=["labels"],
            min_visibility=self.min_visibility,
        )
        with warnings.catch_warnings():
            # "Got processor for bboxes, but no transform to process it" - true, and not
            # the user's problem: a pipeline of nothing but intensity transforms still has
            # to declare how boxes are carried, because the dataset always passes them.
            # Narrow enough a filter that a different warning still gets through.
            warnings.filterwarnings("ignore", message=".*no transform to process it.*")
            # `seed` reaches albumentations' own generator, which is the only thing that
            # makes a pipeline reproducible: seeding numpy does not, and a test that
            # seeded numpy and believed it was controlled failed in CI and passed
            # locally. None leaves it unseeded, which is what training wants - the engine
            # seeds torch, and a fixed augmentation sequence across epochs would mean
            # every epoch saw the same distortions.
            self._pipeline = albumentations.Compose(built, bbox_params=self._params, seed=seed)
        _refuse_transforms_without_boxes(albumentations, steps, built, self._params, input_shape)

    def __call__(
        self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[0], image.shape[1]
        # Clipped first: albumentations validates that a box lies inside its frame, and a
        # letterboxed box can sit a hair outside after the scale - which would stop a run
        # with a validation error about a box that is wrong by 1e-9.
        inside = clip_boxes(boxes, (height, width)) if len(boxes) else boxes
        out = self._pipeline(
            image=image,
            bboxes=[list(map(float, b)) for b in inside],
            labels=[int(v) for v in labels],
        )
        kept = np.asarray(out["bboxes"], dtype=float).reshape(-1, 4)
        return out["image"], kept, np.asarray(out["labels"], dtype=int)


def _build_steps(albumentations, steps: list[AugmentationStep]) -> list:
    """Shared by both augmenters: a named transform with its parameters, or a refusal."""
    built = []
    for step in steps:
        factory = getattr(albumentations, step.name, None)
        if factory is None:
            raise AugmentationError(
                f"albumentations {albumentations.__version__} has no transform '{step.name}'"
            )
        try:
            built.append(factory(**step.params))
        except (TypeError, ValueError) as error:
            # ValueError as well as TypeError: albumentations 2.x validates its arguments
            # with pydantic, so a missing or misspelled one arrives as a ValueError full
            # of schema prose rather than as Python's own "unexpected keyword argument".
            raise AugmentationError(
                f"'{step.name}' rejected its parameters {step.params}: {error}"
            ) from None
    return built


def _refuse_transforms_without_boxes(
    albumentations, steps: list[AugmentationStep], built: list, params, input_shape: tuple[int, int]
) -> None:
    """Try each transform on an image the size the real one will be, and name the one that
    cannot handle boxes.

    Per step, for the same reason the volume probe is per step: a pipeline of eight
    transforms that fails as a whole leaves the user bisecting. And with `p=1`, for the
    reason the volume probe learned the hard way - most transforms default to `p=0.5`, so
    probing with the user's own parameters asks the question half the time and an
    unsupported transform gets through every other run.

    **At the model's input size, and that is not a detail.** The first version probed on a
    fixed 32x32 image and refused `RandomCrop(height=64, width=64)` - a perfectly good
    transform - with "Crop size exceeds image dimensions". A probe that fails for its own
    reasons is indistinguishable from a real refusal, and this one would have told the user
    their transform was unsupported when it was the check that was too small. Probing at
    the size the transform will really see also means a crop genuinely larger than the
    input is still refused, which is correct: at training time it would fail on every
    sample.
    """
    height, width = int(input_shape[0]), int(input_shape[1])
    probe = np.zeros((height, width, 3), dtype=np.float32)
    inset_y, inset_x = max(height // 4, 1), max(width // 4, 1)
    probe[inset_y : height - inset_y, inset_x : width - inset_x] = 1.0
    box = [[float(inset_x), float(inset_y), float(width - inset_x), float(height - inset_y)]]

    for step, transform in zip(steps, built, strict=True):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                albumentations.Compose([_always(transform, step)], bbox_params=params)(
                    image=probe, bboxes=box, labels=[0]
                )
        except Exception as error:  # noqa: BLE001 - albumentations raises whatever it likes
            raise AugmentationError(
                f"'{step.name}' cannot transform bounding boxes: "
                f"{type(error).__name__}: {error}. The failure comes from inside "
                f"albumentations {albumentations.__version__}, which is why it is reported "
                f"here. Drop the transform, or use one that handles boxes - the flips, the "
                f"rotations, Affine and the crops all do."
            ) from None


def build_box_augmenter(
    steps: list[AugmentationStep] | None,
    *,
    input_shape: tuple[int, int],
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    seed: int | None = None,
) -> BoxAugmenter | None:
    """None when there is nothing to do, which keeps the caller free of special cases.

    `input_shape` is required rather than defaulted because the probe is only honest at the
    size the transforms will really see - see `_refuse_transforms_without_boxes`.

    `seed` is for tests, and it is not a spec field on purpose: pinning the augmentation
    sequence would make every epoch see the same distortions, which is the opposite of what
    augmentation is for. The specification's own `seed` covers weight initialisation and
    the order examples arrive in, which is what reproducibility means here.
    """
    if not steps:
        return None
    return AlbumentationsBoxAugmenter(
        steps, input_shape=input_shape, min_visibility=min_visibility, seed=seed
    )
