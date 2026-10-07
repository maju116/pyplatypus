"""Augmenting an image and the boxes that label it.

The assertion worth making here is **correspondence**: after a transform, does the box
still cover the object? A test that checked shapes would pass on a pipeline that moved
every pixel and left every box where it was - which is the whole failure mode, and it
trains without complaint while teaching the model that empty background is a blood cell.

So every geometric test below paints a bright square, transforms it, and compares the box
that came back with the bounding box of the bright pixels in the output. That is also how
the claim in `AlbumentationsBoxAugmenter`'s docstring was established, and keeping it as a
test rather than a note means an albumentations upgrade re-checks it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus.data.augmentation import (
    AugmentationError,
    build_box_augmenter,
)
from pyplatypus.spec.components import AugmentationStep

SIZE = 96


def steps(*named):
    """`[("HorizontalFlip", {}), ...]` as the spec would carry it, always applied."""
    return [AugmentationStep(name=name, params={**params, "p": 1.0}) for name, params in named]


def measured(image):
    """Where the bright region is now, read off the pixels themselves."""
    grey = image.mean(axis=2)
    span = grey.max() - grey.min()
    if span < 1e-6:
        return None
    hot = grey > grey.min() + 0.5 * span
    if not hot.any():
        return None
    ys, xs = np.where(hot)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def bright_square():
    """A rectangle, not a square: a transposed axis is invisible on a square."""
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    image[20:50, 30:70] = 1.0
    return image, np.array([[30.0, 20.0, 70.0, 50.0]]), np.array([0])


#: Transforms whose box lands exactly where the pixels land: measured 0.00 px of
#: disagreement over twenty seeded draws each. Flips and quarter turns move pixels without
#: resampling them, a pure translation is a shift, and a crop is a window.
EXACT = [
    ("HorizontalFlip", {}),
    ("VerticalFlip", {}),
    ("RandomRotate90", {}),
    ("Transpose", {}),
    ("D4", {}),
    ("Affine", {"translate_percent": 0.2}),
    ("RandomCrop", {"height": 64, "width": 64}),
    ("CenterCrop", {"height": 64, "width": 64}),
]

#: Transforms that resample, where the box is the bounding box of the warped *corners* and
#: the bright pixels are what survived interpolation and a threshold. Those are not the
#: same measurement: near a warped corner the object thins to a point and the last of it
#: falls below any threshold. Measured worst disagreement over twenty seeded draws each:
#: 1.00 px, in both directions, `GridDistortion` being the worst of them.
WARPING = [
    ("Affine", {"rotate": 30}),
    ("Affine", {"scale": 1.4}),
    ("Rotate", {"limit": (30, 30)}),
    ("Perspective", {}),
    ("GridDistortion", {}),
]

#: What the measurement above found, plus a little. Not a guess: at 1.0 the suite would sit
#: exactly on the worst observed draw.
WARP_TOLERANCE = 1.5

DRAWS = 20


def ids_for(cases):
    return [f"{n}{tuple(p)}" if p else n for n, p in cases]


@pytest.mark.parametrize(("name", "params"), EXACT, ids=ids_for(EXACT))
def test_an_exact_transform_puts_the_box_exactly_where_the_pixels_are(name, params):
    for seed in range(DRAWS):
        augmenter = build_box_augmenter(
            steps((name, params)), input_shape=(SIZE, SIZE), min_visibility=0.0, seed=seed
        )
        image, boxes, labels = bright_square()
        out_image, out_boxes, out_labels = augmenter(image, boxes, labels)
        where = measured(out_image)
        if len(out_boxes) == 0 or where is None:
            continue
        assert len(out_labels) == len(out_boxes)
        assert out_boxes[0] == pytest.approx(where, abs=0.5), (
            f"{name} seed {seed} returned {out_boxes[0]} for an object at {where}"
        )


@pytest.mark.parametrize(("name", "params"), WARPING, ids=ids_for(WARPING))
def test_a_warping_transform_still_bounds_the_object(name, params):
    """Containment is the claim, not equality.

    A resampling transform's box is the bounding box of the warped corners; the bright
    pixels are what came through interpolation above a threshold. The first version of this
    asserted equality, and `Perspective` failed it in CI by 2.6 px while passing locally -
    because the draws were not actually seeded. numpy was, and albumentations 2.x draws from
    its own generator, so "eight seeds" was eight uncontrolled runs. `Compose(seed=...)` is
    what controls it, and with that the worst disagreement over twenty draws is 1.00 px.

    What must hold is that the box **covers the object**: a box that no longer does is the
    failure this whole file exists for, and it would train a model to find blood cells in
    background.
    """
    for seed in range(DRAWS):
        augmenter = build_box_augmenter(
            steps((name, params)), input_shape=(SIZE, SIZE), min_visibility=0.0, seed=seed
        )
        image, boxes, labels = bright_square()
        out_image, out_boxes, _ = augmenter(image, boxes, labels)
        where = measured(out_image)
        if len(out_boxes) == 0 or where is None:
            continue
        got = out_boxes[0]
        outside = max(
            got[0] - where[0], got[1] - where[1], where[2] - got[2], where[3] - got[3], 0.0
        )
        slack = max(where[0] - got[0], where[1] - got[1], got[2] - where[2], got[3] - where[3], 0.0)
        assert outside <= WARP_TOLERANCE, (
            f"{name} seed {seed}: {outside:.2f} px of the object is outside the box "
            f"{got} - the box no longer covers what it labels"
        )
        assert slack <= WARP_TOLERANCE, (
            f"{name} seed {seed}: the box {got} is {slack:.2f} px larger than the object at {where}"
        )


def test_the_box_is_not_simply_left_alone():
    """The test above is only worth anything if the box actually had to move. A pipeline
    that returned every box unchanged would satisfy a correspondence check on an identity
    transform, so this pins that the flip is a flip."""
    augmenter = build_box_augmenter(steps(("HorizontalFlip", {})), input_shape=(SIZE, SIZE))
    image, boxes, labels = bright_square()
    _, out_boxes, _ = augmenter(image, boxes, labels)
    assert out_boxes[0] == pytest.approx([SIZE - 70.0, 20.0, SIZE - 30.0, 50.0])
    assert not np.allclose(out_boxes[0], boxes[0])


def test_labels_follow_their_boxes():
    """Three boxes of three classes, flipped. The order albumentations returns them in is
    its own business; what must hold is that each label still belongs to its box."""
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    boxes = np.array([[4.0, 4.0, 20.0, 20.0], [40.0, 40.0, 60.0, 60.0], [70.0, 10.0, 90.0, 30.0]])
    labels = np.array([0, 1, 2])
    augmenter = build_box_augmenter(
        steps(("HorizontalFlip", {})), input_shape=(SIZE, SIZE), min_visibility=0.0
    )
    _, out_boxes, out_labels = augmenter(image, boxes, labels)

    assert len(out_boxes) == 3
    pairs = {int(label): tuple(np.round(box, 3)) for label, box in zip(out_labels, out_boxes)}
    for index, box in enumerate(boxes):
        expected = (SIZE - box[2], box[1], SIZE - box[0], box[3])
        assert pairs[index] == pytest.approx(expected)


def test_an_intensity_transform_leaves_the_boxes_where_they_are():
    """Nothing geometric happened, so nothing should move. Checked because the pipeline
    carries box parameters for every step, and albumentations warns when it is given boxes
    and no transform that uses them.

    The brightness is a fixed +0.5 rather than the default random range. With the default,
    `brightness_limit=0.2` draws a factor that can come out at approximately zero, so
    "the transform did nothing at all" failed about one run in six - a test that fails
    intermittently is worse than no test, because the failure reads as a real one.
    """
    augmenter = build_box_augmenter(
        steps(("RandomBrightnessContrast", {"brightness_limit": (0.5, 0.5), "contrast_limit": 0})),
        input_shape=(SIZE, SIZE),
    )
    image, boxes, labels = bright_square()
    out_image, out_boxes, _ = augmenter(image, boxes, labels)
    assert out_boxes == pytest.approx(boxes)
    assert not np.allclose(out_image, image), "the transform did nothing at all"


# --- what happens to a box a crop cuts away ---------------------------------------------


def test_min_visibility_drops_a_box_that_keeps_almost_nothing():
    """The direction is what this asserts, not the number. A box keeping a sliver of its
    object labels background as a cell, and that is worse than not being taught about a
    truncated object at all."""
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    # A 40x30 box whose right-hand 4 pixels are all that a left crop keeps: 10% of it.
    boxes = np.array([[56.0, 20.0, 96.0, 50.0]])
    labels = np.array([0])

    cropping = steps(("Crop", {"x_min": 0, "y_min": 0, "x_max": 60, "y_max": 96}))
    strict = build_box_augmenter(cropping, input_shape=(SIZE, SIZE), min_visibility=0.25)
    lenient = build_box_augmenter(cropping, input_shape=(SIZE, SIZE), min_visibility=0.0)

    assert len(strict(image, boxes, labels)[1]) == 0
    assert len(lenient(image, boxes, labels)[1]) == 1


def test_a_box_that_survives_a_crop_is_clipped_to_the_frame():
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    boxes = np.array([[20.0, 20.0, 80.0, 50.0]])
    augmenter = build_box_augmenter(
        steps(("Crop", {"x_min": 0, "y_min": 0, "x_max": 60, "y_max": 96})),
        input_shape=(SIZE, SIZE),
        min_visibility=0.25,
    )
    _, out_boxes, _ = augmenter(image, boxes, np.array([0]))
    assert len(out_boxes) == 1
    assert out_boxes[0] == pytest.approx([20.0, 20.0, 60.0, 50.0])


def test_an_image_with_no_boxes_still_goes_through():
    """A frame with nothing in it is a legitimate training example - it is what teaches
    the model what background looks like - and it must not be special-cased out."""
    augmenter = build_box_augmenter(steps(("HorizontalFlip", {})), input_shape=(SIZE, SIZE))
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    image[:, :10] = 1.0
    out_image, out_boxes, out_labels = augmenter(image, np.zeros((0, 4)), np.zeros(0, dtype=int))
    assert out_boxes.shape == (0, 4)
    assert out_labels.shape == (0,)
    assert out_image[:, -10:].mean() > 0.9, "the image was flipped"


def test_a_box_sitting_a_hair_outside_the_frame_does_not_stop_the_run():
    """Letterboxing multiplies by a float, so a box on the edge can land at 96.0000001 -
    and albumentations validates that a box is inside its frame. Clipped before it gets
    there, because stopping a run over 1e-7 of a pixel is not a useful refusal."""
    augmenter = build_box_augmenter(steps(("HorizontalFlip", {})), input_shape=(SIZE, SIZE))
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    boxes = np.array([[30.0, 20.0, SIZE + 1e-7, 50.0]])
    _, out_boxes, _ = augmenter(image, boxes, np.array([0]))
    assert len(out_boxes) == 1


# --- refusals ---------------------------------------------------------------------------


def test_an_unknown_transform_is_refused_while_the_spec_is_read():
    """Earlier than this file, and better: `AugmentationStep` checks the name against what
    albumentations offers and suggests the nearest one. Asserted here because that is
    where someone looks for it, and because the augmenter's own "has no transform" branch
    is therefore a safety net rather than the path anybody takes."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="unknown augmentation 'Nope'"):
        steps(("Nope", {}))


def test_bad_parameters_are_refused_by_name():
    with pytest.raises(AugmentationError, match="RandomCrop"):
        build_box_augmenter(steps(("RandomCrop", {"nonsense": 1})), input_shape=(SIZE, SIZE))


def test_nothing_to_do_gives_no_augmenter():
    assert build_box_augmenter(None, input_shape=(SIZE, SIZE)) is None
    assert build_box_augmenter([], input_shape=(SIZE, SIZE)) is None
