"""Augmenting a target that is a vector per voxel, which is the failure that stays green.

A flow field is not a mask. A horizontal flip has to negate its x component and a quarter
turn has to swap and sign its components; carry one through as a mask and the arrows point
somewhere else, while the loss still falls and the model still trains. `pyplatypus#210`
settled this by **rebuilding the target from the augmented label map** rather than
correcting it, and these tests are the two measurements that decision rests on plus the
guard that keeps it from being undone.

The flow field here is built by this file rather than by the package: the builder belongs to
`separation: flows` (`pyplatypus#212`) and what is being tested is the route, not the target.
"""

import albumentations as A
import numpy as np
import pytest

from pyplatypus.data import SegmentationDataset, discover
from pyplatypus.data.augmentation import AugmentationError, build_augmenter
from pyplatypus.spec.components import AugmentationStep
from pyplatypus.spec.models import SegmentationModel

SIZE = 64
SIDE = 10
SEEDS = 20


def labels_and_image(size=SIZE, side=SIDE):
    """Squares on a grid, each its own instance id."""
    labels = np.zeros((size, size), np.int64)
    corners = [(r, c) for r in range(2, size - side, 16) for c in range(2, size - side, 16)]
    for identifier, (row, column) in enumerate(corners, start=1):
        labels[row : row + side, column : column + side] = identifier
    image = (labels > 0).astype(np.float32)[..., None].repeat(3, axis=-1)
    return labels, image


def flows(labels):
    """Unit vectors towards each object's own centre of mass, channels (dy, dx) last.

    Exactly zero at a centre, which is a property the regression losses' mask depends on
    not being read off this array - see `tests/objectives/test_regression.py`.
    """
    out = np.zeros((*labels.shape, 2), np.float32)
    rows, columns = np.indices(labels.shape)
    for label in np.unique(labels):
        if label == 0:
            continue
        inside = labels == label
        offset_y = rows[inside].mean() - rows[inside]
        offset_x = columns[inside].mean() - columns[inside]
        norm = np.maximum(np.hypot(offset_y, offset_x), 1e-8)
        out[..., 0][inside] = offset_y / norm
        out[..., 1][inside] = offset_x / norm
    return out


def carried(transform, labels, image, field, seed=0):
    """The label map and the field put through one transform together, as a naive pipeline
    would do it: the field riding along in albumentations' `masks` list."""
    out = A.Compose([transform], seed=seed)(
        image=image, mask=labels, masks=[field[..., 0], field[..., 1]]
    )
    return out["mask"], np.stack(out["masks"], axis=-1)


def angles(a, b, inside):
    """Degrees between two fields, over the voxels `inside` selects."""
    first, second = a[inside], b[inside]
    norms = np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), 1e-8)
    cosine = np.clip((first * second).sum(-1) / norms, -1, 1)
    return np.degrees(np.arccos(cosine))


# --- the corrections that do exist, and are exact -------------------------------------


def test_rebuilding_agrees_exactly_with_the_correction_a_flip_needs():
    """Negating x after a horizontal flip is right, and rebuilding gives the same field to
    the bit. That is what makes the crop test below a statement about crops rather than
    about this file's arithmetic."""
    labels, image = labels_and_image()
    field = flows(labels)
    moved_labels, moved_field = carried(A.HorizontalFlip(p=1), labels, image, field)
    moved_field[..., 1] *= -1

    rebuilt = flows(moved_labels)
    inside = moved_labels > 0
    assert float(angles(moved_field, rebuilt, inside).max()) == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(moved_field[inside], rebuilt[inside], atol=1e-6)


def test_rebuilding_agrees_exactly_with_the_correction_a_quarter_turn_needs():
    """A transpose is the quarter turn with no interpolation in it: (y, x) becomes (x, y),
    so the two components swap. A known arrow ends up pointing where it should, which is
    asserted on one voxel as well as over the field - the aggregate would pass if two
    errors cancelled."""
    labels, image = labels_and_image()
    field = flows(labels)
    moved_labels, moved_field = carried(A.Transpose(p=1), labels, image, field)
    moved_field = moved_field[..., ::-1].copy()

    rebuilt = flows(moved_labels)
    inside = moved_labels > 0
    assert np.allclose(moved_field[inside], rebuilt[inside], atol=1e-6)

    # The top-left square spans rows and columns 2..11, so its centre is at 6.5 and the
    # voxel at (2, 2) points down and right at 45 degrees. Transposed it is still (2, 2)
    # and still 45 degrees - and that is the voxel a swap that forgot to swap would pass.
    assert rebuilt[2, 2].tolist() == pytest.approx([np.sqrt(0.5), np.sqrt(0.5)], abs=1e-6)
    assert rebuilt[11, 2].tolist() == pytest.approx([-np.sqrt(0.5), np.sqrt(0.5)], abs=1e-6)


def test_carrying_a_field_through_a_flip_without_the_correction_is_wrong():
    """The control the two tests above need: the agreement is the correction's doing, not
    something a flip does for free. Mirrored, every arrow's x component has the wrong sign,
    which leaves half the field pointing more than a right angle away from the truth."""
    labels, image = labels_and_image()
    field = flows(labels)
    moved_labels, moved_field = carried(A.HorizontalFlip(p=1), labels, image, field)

    inside = moved_labels > 0
    wrong = angles(moved_field, flows(moved_labels), inside)
    # 90.0 and 167.3 degrees on this fixture, against 0.0 once x is negated. Bounded rather
    # than pinned because the extreme depends on the square's size, not on the mirroring.
    assert float(np.median(wrong)) > 45
    assert float(wrong.max()) > 160


# --- the correction that does not exist ------------------------------------------------


def test_a_crop_moves_the_centre_and_no_correction_can_answer_that():
    """The measurement that settled the decision. A crop has no component to negate and is
    still wrong: the surviving part of an object has a new centre, so the arrows carried
    into the crop point at a place that is no longer in the picture.

    Measured on this machine over twenty seeded 32-pixel crops: **49.5% of foreground
    voxels disagree, by a median of 8.9 degrees and up to 180**. The bounds here are wide
    on purpose - the crop positions come from albumentations' own generator, so the exact
    share is a property of the library's version rather than of the geometry - and they are
    calibrated at both ends: a correct rebuild against a carried field disagrees on about
    half the voxels, while the flip above disagrees on none at all.
    """
    labels, image = labels_and_image()
    field = flows(labels)
    shares, worst = [], []
    for seed in range(SEEDS):
        moved_labels, moved_field = carried(
            A.RandomCrop(height=32, width=32, p=1), labels, image, field, seed=seed
        )
        inside = moved_labels > 0
        if not inside.any():
            continue
        difference = angles(moved_field, flows(moved_labels), inside)
        shares.append(float((difference > 1).mean()))
        worst.append(float(difference.max()))

    assert 0.3 < float(np.mean(shares)) < 0.7, float(np.mean(shares))
    assert max(worst) > 90, "no arrow points more than a right angle away; this fixture should"


def test_albumentations_never_invents_a_label():
    """What makes rebuilding available at all: an *instance* label map can go through the
    augmenter unharmed. Blending 3 and 5 into 4 would not look like damage - 4 is another
    object's id - so the target would be built for the wrong nucleus with nothing out of
    range anywhere.

    Sixteen transforms, twenty seeded draws, forty distinct ids, measured here rather than
    assumed; a version bump re-runs it. Ids are *lost* where the geometry removes them,
    which is the frame doing its job and is asserted separately so this cannot pass by
    nothing happening.
    """
    labels, image = labels_and_image()
    present = set(np.unique(labels).tolist())
    transforms = {
        "HorizontalFlip": A.HorizontalFlip(p=1),
        "VerticalFlip": A.VerticalFlip(p=1),
        "RandomRotate90": A.RandomRotate90(p=1),
        "Transpose": A.Transpose(p=1),
        "D4": A.D4(p=1),
        "Affine rotate": A.Affine(rotate=17, p=1),
        "Affine scale": A.Affine(scale=1.3, p=1),
        "Affine shear": A.Affine(shear=10, p=1),
        "Rotate": A.Rotate(limit=30, p=1),
        "Perspective": A.Perspective(p=1),
        "ElasticTransform": A.ElasticTransform(p=1),
        "GridDistortion": A.GridDistortion(p=1),
        "RandomCrop": A.RandomCrop(height=32, width=32, p=1),
        "Resize": A.Resize(height=48, width=48, p=1),
        "RandomResizedCrop": A.RandomResizedCrop(size=(48, 48), p=1),
        "RandomBrightnessContrast": A.RandomBrightnessContrast(p=1),
    }
    lost_by_the_crop = 0
    for name, transform in transforms.items():
        for seed in range(SEEDS):
            out = A.Compose([transform], seed=seed)(image=image, mask=labels)["mask"]
            values = set(np.unique(out).tolist())
            assert not values - present, f"{name} invented {sorted(values - present)}"
            if name == "RandomCrop":
                lost_by_the_crop += len(present - values - {0})
    assert lost_by_the_crop > 0, "the crop lost no ids, so nothing was being cropped"

    # And the control, because a test that cannot fail reads as proof and is none: the same
    # transform with the label map sent down the *image* path interpolates, and invents
    # hundreds of values that were in no voxel. The mask path is what makes this safe, not
    # the geometry.
    through_the_image = A.Compose([A.Affine(rotate=17, p=1)], seed=0)(
        image=labels.astype(np.float32)
    )["image"]
    assert set(np.unique(through_the_image).tolist()) - present


# --- and the guard ---------------------------------------------------------------------


def test_a_continuous_mask_is_refused_before_it_can_be_moved():
    augmenter = build_augmenter([AugmentationStep(name="HorizontalFlip", params={"p": 1.0})])
    image = np.zeros((8, 8, 3), np.float32)
    with pytest.raises(AugmentationError, match="continuous target"):
        augmenter(image, np.zeros((8, 8), np.float32))
    # And the integer mask every path in the package sends still goes through.
    _, moved = augmenter(image, np.zeros((8, 8), np.int64))
    assert moved is not None


def test_the_dataset_hands_the_augmenter_integers_and_builds_the_target_afterwards(
    nested_root, binary_data
):
    """The route, pinned in the data path rather than in a docstring. One-hot happens after
    augmentation, so a target built from what the augmenter returned is a target built from
    the geometry the model will see."""
    seen = {}

    def recording_augmenter(image, mask=None):
        seen["dtype"] = mask.dtype
        seen["ndim"] = mask.ndim
        return image, mask

    model = SegmentationModel(name="m", input_shape=(32, 32), channels=3, blocks=1, filters=2)
    samples = discover(nested_root, binary_data).samples
    data = SegmentationDataset(samples, model, binary_data, augmenter=recording_augmenter)
    _, onehot = data[0]

    assert np.issubdtype(seen["dtype"], np.integer), seen["dtype"]
    assert seen["ndim"] == 2, "the augmenter saw a channel axis, so this is not class indices"
    assert onehot.shape[-1] == binary_data.n_class
