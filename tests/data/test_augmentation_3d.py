"""Augmenting volumes, and being honest about what albumentations can do with them.

Volumes go in through `volume` and `mask3d` rather than `image` and `mask`, and support is
uneven: in 2.0.8 most geometric transforms work and several intensity ones raise from inside
the library - `GaussNoise` arrives as `KeyError: 'images'`. The tests here are mostly about
turning that into something a user can act on before training starts.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus.data.augmentation import AugmentationError, build_augmenter
from pyplatypus.spec.components import AugmentationStep, available_transforms


def volume_and_mask(shape=(4, 8, 8)):
    volume = np.zeros((*shape, 1), dtype=np.float32)
    volume[:, 1:4, 1:4] = 1.0
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, 1:4, 1:4] = 1
    return volume, mask


def steps(*names, p: float = 1.0):
    """Always `p=1` unless a test says otherwise.

    Most albumentations transforms default to `p=0.5`, so a test that asserts "the volume
    changed" is a coin toss - which is exactly how the first version of this file behaved, and
    how the probe it tests behaved before that.
    """
    return [AugmentationStep(name=name, params={"p": p}) for name in names]


# ------------------------------------------------------------------ the basics
def test_a_volume_augmenter_is_built_for_rank_three():
    augmenter = build_augmenter(steps("HorizontalFlip"), rank=3)
    assert augmenter is not None
    assert augmenter.rank == 3


def test_a_geometric_transform_moves_a_volume_and_its_mask_together():
    """The property that matters: whatever happens to the image happens to the labels."""
    augmenter = build_augmenter(steps("HorizontalFlip"), rank=3)
    volume, mask = volume_and_mask()
    moved_volume, moved_mask = augmenter(volume, mask)

    assert moved_volume.shape == volume.shape
    assert moved_mask.shape == mask.shape
    assert not np.array_equal(moved_volume, volume)          # something happened
    # And the mask still marks exactly where the bright voxels are.
    bright = moved_volume[..., 0] > 0.5
    assert np.array_equal(bright, moved_mask.astype(bool))


def test_the_mask_is_optional():
    augmenter = build_augmenter(steps("HorizontalFlip"), rank=3)
    volume, _ = volume_and_mask()
    out, mask = augmenter(volume)
    assert out.shape == volume.shape
    assert mask is None


def test_every_slice_gets_the_same_geometry():
    """A per-slice augmentation would tear the anatomy apart: slice 3 rotated and slice 4 not
    is not a scan of anything. albumentations applies one set of parameters to the volume, and
    this is the test that says so."""
    augmenter = build_augmenter(steps("Affine"), rank=3)
    volume = np.zeros((4, 8, 8, 1), dtype=np.float32)
    volume[:, 2:5, 2:5] = 1.0                                # identical in every slice
    out, _ = augmenter(volume)
    first = out[0, ..., 0]
    assert all(np.array_equal(out[index, ..., 0], first) for index in range(1, 4))


def test_two_dimensional_augmentation_still_works():
    augmenter = build_augmenter(steps("HorizontalFlip"), rank=2)
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[1:4, 1:4] = 1.0
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    out, out_mask = augmenter(image, mask)
    assert out.shape == image.shape and out_mask.shape == mask.shape


# -------------------------------------------------------- the unsupported ones
def test_a_transform_that_cannot_do_volumes_is_refused_by_name():
    """`GaussNoise` on a volume raises `KeyError: 'images'` from inside albumentations, forty
    minutes into training if nobody checked. The check happens while the pipeline is built."""
    with pytest.raises(AugmentationError, match="GaussNoise"):
        build_augmenter(steps("GaussNoise"), rank=3)


def test_the_refusal_says_what_to_do_about_it():
    with pytest.raises(AugmentationError, match="available_transforms"):
        build_augmenter(steps("GaussNoise"), rank=3)


def test_the_offender_is_named_even_in_a_long_pipeline():
    """Probing the composed pipeline would say that something in the list is unsupported
    without saying which, leaving the user to bisect eight transforms by hand."""
    with pytest.raises(AugmentationError, match="ChannelDropout"):
        build_augmenter(
            steps("HorizontalFlip", "VerticalFlip", "Affine", "ChannelDropout", "Blur"),
            rank=3,
        )


def test_the_same_transform_is_fine_in_two_dimensions():
    # Which is the point of checking per rank rather than banning it outright.
    assert build_augmenter(steps("GaussNoise"), rank=2) is not None


# ------------------------------------------------------------------- listings
def test_the_three_dimensional_listing_is_smaller_and_not_empty():
    two, three = available_transforms(), available_transforms(rank=3)
    assert three < two
    assert len(three) > 50


def test_geometric_transforms_are_listed_for_volumes():
    three = available_transforms(rank=3)
    for name in ("Affine", "ElasticTransform", "CubicSymmetry", "HorizontalFlip", "D4"):
        assert name in three


def test_transforms_needing_arguments_are_not_dropped():
    """`CenterCrop3D` needs a size, so it cannot be probed with defaults - and it is one of the
    transforms written for volumes. "Could not check" is not "does not work"; dropping those
    would have hidden exactly the 3D-native ones."""
    three = available_transforms(rank=3)
    for name in ("CenterCrop3D", "RandomCrop3D", "Pad3D"):
        assert name in three


def test_the_ones_that_demonstrably_fail_are_excluded():
    three = available_transforms(rank=3)
    for name in ("GaussNoise", "ChannelDropout", "ChromaticAberration"):
        assert name not in three


def test_an_unsupported_rank_is_refused():
    with pytest.raises(AugmentationError, match="2D and 3D"):
        build_augmenter(steps("HorizontalFlip"), rank=4)


# ---------------------------------------------------------------- end to end
def test_a_3d_model_trains_with_augmentation(volume_root):
    """Augmentation on, one epoch, through the engine - the only proof that the pipeline the
    probe approved is the pipeline training uses."""
    pytest.importorskip("nibabel")
    from pyplatypus import Engine, from_dict

    spec = from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(volume_root), "validation_path": str(volume_root),
                 "labels": [0, 1], "shuffle": False},
        "models": [{"name": "unet3d", "input_shape": [8, 8, 4], "n_class": 2, "channels": 1,
                    "blocks": 2, "filters": 4, "batch_size": 1, "epochs": 1,
                    "metrics": [{"name": "dice"}],
                    "augmentation": [{"name": "HorizontalFlip"},
                                     {"name": "VerticalFlip"},
                                     {"name": "CubicSymmetry"}]}],
    })
    engine = Engine(spec, device="cpu")
    history = engine.fit()["unet3d"]
    assert len(history) == 1
    assert "val_dice" in history.records[0]


def test_validation_is_not_augmented(volume_root):
    """Measuring a model on distorted data measures the distortion. The engine builds an
    augmenter for training only, and this checks it at rank 3 too."""
    pytest.importorskip("nibabel")
    from pyplatypus import Engine, from_dict

    spec = from_dict({
        "task": "semantic_segmentation",
        "data": {"train_path": str(volume_root), "validation_path": str(volume_root),
                 "labels": [0, 1]},
        "models": [{"name": "unet3d", "input_shape": [8, 8, 4], "n_class": 2, "channels": 1,
                    "blocks": 2, "filters": 4, "batch_size": 1, "epochs": 1,
                    "augmentation": [{"name": "HorizontalFlip"}]}],
    })
    engine = Engine(spec, device="cpu")
    train = engine.dataset(spec.models[0], "train", augmented=True)
    validation = engine.dataset(spec.models[0], "validation")
    assert train.augmenter is not None
    assert validation.augmenter is None


def test_a_crop_larger_than_the_probe_is_not_refused_for_the_wrong_reason():
    """The probe used to be a fixed 4x8x8 volume, so a crop bigger than that was refused
    with "crop size exceeds image dimensions" - the check failing for its own reasons
    while reporting the user's transform as unsupported.

    Found in the box probe, where the same fixed-size probe refused
    `RandomCrop(height=64, width=64)`, and fixed in both: the probe is the size the
    transforms will really see.
    """
    from pyplatypus.spec.components import AugmentationStep

    crop = [AugmentationStep(name="RandomCrop3D",
                             params={"size": (4, 16, 16), "p": 1.0})]
    assert build_augmenter(crop, rank=3, input_shape=(8, 32, 32)) is not None

    # The 2D counterpart of this is a refusal, because albumentations' 2D crops raise
    # `CropSizeError` when asked for more than the image holds. Its 3D crop does not: a
    # `RandomCrop3D(size=(4, 64, 64))` on a 32x32 volume returns a 32x32 one, measured,
    # without a word. So there is no second half to this test, and a 3D spec asking for a
    # patch larger than its input gets a quietly smaller patch - which is the library's
    # behaviour and worth knowing rather than worth asserting.
