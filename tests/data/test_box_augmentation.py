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
    return [AugmentationStep(name=name, params={**params, "p": 1.0})
            for name, params in named]


def bright_square():
    """A rectangle, not a square: a transposed axis is invisible on a square."""
    image = np.zeros((SIZE, SIZE, 3), np.float32)
    image[20:50, 30:70] = 1.0
    return image, np.array([[30.0, 20.0, 70.0, 50.0]]), np.array([0])


def measured(image):
    """Where the bright region is now, read off the pixels themselves."""
    grey = image.mean(axis=2)
    span = grey.max() - grey.min()
    if span < 1e-6:
        return None
    hot = grey > grey.min() + 0.5 * span
    ys, xs = np.where(hot)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


#: Every geometric transform in albumentations 2.0.8, with the worst disagreement measured
#: between the box it returns and the bounding box of the pixels it moved. The non-zero
#: ones are anti-aliasing at an edge, not a transform getting it wrong - which is why the
#: tolerance is a pixel and a half rather than zero, and why it is per transform rather
#: than one number for all of them: a regression in `HorizontalFlip` must not hide behind
#: `Perspective`'s rounding.
GEOMETRIC = [
    ("HorizontalFlip", {}, 0.0),
    ("VerticalFlip", {}, 0.0),
    ("RandomRotate90", {}, 0.0),
    ("Transpose", {}, 0.0),
    ("D4", {}, 0.0),
    ("Affine", {"translate_percent": 0.2}, 0.0),
    ("Affine", {"rotate": 30}, 1.0),
    ("Affine", {"scale": 1.4}, 1.0),
    ("Perspective", {}, 1.5),
    ("GridDistortion", {}, 1.5),
    ("RandomCrop", {"height": 64, "width": 64}, 0.0),
    ("CenterCrop", {"height": 64, "width": 64}, 0.0),
    ("Rotate", {"limit": (30, 30)}, 1.0),
]


@pytest.mark.parametrize(("name", "params", "tolerance"), GEOMETRIC,
                         ids=[f"{n}{tuple(p)}" if p else n for n, p, _ in GEOMETRIC])
def test_a_geometric_transform_moves_the_box_with_the_pixels(name, params, tolerance):
    """Run several times, because most of these choose their parameters at random.

    A single run of `Affine(rotate=30)` that happened to rotate by nothing would pass on a
    pipeline that never moves a box. Eight runs is the habit this project arrived at after
    a `p=0.5` probe passed every other time.
    """
    augmenter = build_box_augmenter(steps((name, params)), input_shape=(SIZE, SIZE), min_visibility=0.0)

    for seed in range(8):
        np.random.seed(seed)
        image, boxes, labels = bright_square()
        out_image, out_boxes, out_labels = augmenter(image, boxes, labels)
        where = measured(out_image)
        if len(out_boxes) == 0 or where is None:
            continue              # cropped out of frame entirely; nothing to compare
        assert len(out_labels) == len(out_boxes)
        assert out_boxes[0] == pytest.approx(where, abs=tolerance + 0.5), (
            f"{name} returned {out_boxes[0]} for an object at {where}"
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
    boxes = np.array([[4.0, 4.0, 20.0, 20.0],
                      [40.0, 40.0, 60.0, 60.0],
                      [70.0, 10.0, 90.0, 30.0]])
    labels = np.array([0, 1, 2])
    augmenter = build_box_augmenter(steps(("HorizontalFlip", {})), input_shape=(SIZE, SIZE), min_visibility=0.0)
    _, out_boxes, out_labels = augmenter(image, boxes, labels)

    assert len(out_boxes) == 3
    pairs = {int(label): tuple(np.round(box, 3))
             for label, box in zip(out_labels, out_boxes)}
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
        steps(("RandomBrightnessContrast",
               {"brightness_limit": (0.5, 0.5), "contrast_limit": 0})),
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
        input_shape=(SIZE, SIZE), min_visibility=0.25,
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
    out_image, out_boxes, out_labels = augmenter(image, np.zeros((0, 4)),
                                                 np.zeros(0, dtype=int))
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
