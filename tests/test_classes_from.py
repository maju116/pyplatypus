"""`classes_from`: one binary mask file per class, for classes that overlap independently.

The last piece of `pyplatypus#91`, and the one the decision of 2026-10-12 deferred rather
than dropped. `regions` answered the common case - BraTS and the lesion-inside-an-organ
family, where the ground truth gives every voxel one value and the scored classes are
unions of those values, so the overlap is a *convention applied to exclusive data*. This
answers the case that cannot be written that way at all: two things drawn separately over
the same voxel, a vessel crossing an organ or two annotators' masks of one structure.

**The whole feature is a reader.** Everything from the target onwards - a sigmoid per
channel, the binary forms of `cce` and `focal`, metrics thresholding each channel, a
prediction that is a stack, a colour map that refuses an overlapping prediction - arrived
with `regions` and reads `data.multilabel`, which this naming now also answers. What is new
is the step before: `unite_masks` takes the element-wise maximum of a sample's mask files,
and that maximum is exactly where per-class identity is destroyed. The test that matters
most here is the control for it - the same two files, read the old way, are one blob.
"""

import warnings

import numpy as np
import pytest
from PIL import Image

from pyplatypus import Engine, from_dict
from pyplatypus.data.channels import ChannelError
from pyplatypus.data.masks import MaskError, codes_to_onehot, masks_to_codes, unite_masks
from pyplatypus.spec.data import SegmentationData

#: One pattern per class, in class order. Written out rather than inferred for the reason
#: `channels_from` is: sorting these names gives an order that is reproducible and means
#: nothing, and a model trained with the vessel in channel 0 and used where channel 0 is the
#: organ answers plausibly, in range, with nothing downstream able to tell.
PATTERNS = [r"_vessel\.png$", r"_organ\.png$"]

SIZE = 64


def data(**overrides):
    block = {"train_path": "t", "validation_path": "v", "classes_from": PATTERNS}
    block.update(overrides)
    return SegmentationData(**block)


# --------------------------------------------------------------------- the naming


def test_classes_from_is_a_fourth_naming_and_the_four_are_exclusive():
    assert data().n_class == 2
    assert data().multilabel is True, "independent channels, like regions"
    assert data().label_map is False, "there is no label map - each file is one class"
    assert data().multilabel_naming == "classes_from"
    for block, why in [
        ({"labels": [0, 1]}, "labels and classes_from"),
        ({"colormap": [[0, 0, 0], [255, 255, 255]]}, "colormap and classes_from"),
        ({"regions": [{"name": "a", "of": [1]}, {"name": "b", "of": [1, 2]}]}, "both overlaps"),
        ({"classes_from": None}, "no naming at all"),
    ]:
        with pytest.raises(ValueError, match="exactly one of"):
            data(**block)


def test_two_patterns_are_the_minimum_and_they_must_be_distinct():
    with pytest.raises(ValueError, match="at least 2"):
        data(classes_from=[r"_vessel\.png$"])
    with pytest.raises(ValueError, match="class patterns must be distinct"):
        data(classes_from=[r"_vessel\.png$", r"_vessel\.png$"])


def test_the_channel_pattern_message_still_says_channel():
    """One matcher, two nouns, and the noun has to follow the field somebody wrote."""
    with pytest.raises(ValueError, match="channel patterns must be distinct"):
        SegmentationData(train_path="t", labels=[0, 1], channels_from=[r"_t1\.nii$", r"_t1\.nii$"])


# ------------------------------------------------------------------ the coding


def binary(hot) -> np.ndarray:
    return (np.asarray(hot, dtype=np.uint8) * 255)[..., None]


def test_a_bit_per_class_and_a_voxel_may_be_in_both():
    vessel = binary([[1, 1, 0, 0]])
    organ = binary([[0, 1, 1, 0]])
    codes, unmatched = masks_to_codes([vessel, organ])
    assert codes.tolist() == [[1, 3, 2, 0]], "bit 0 the vessel, bit 1 the organ, 3 both"
    assert unmatched == pytest.approx(0.25), "one voxel of four is in no class at all"


def test_the_expansion_reads_the_bits_back():
    onehot = codes_to_onehot(np.array([[1, 3, 2, 0]]), 2)
    assert onehot.shape == (1, 4, 2)
    assert onehot[0].tolist() == [[1, 0], [1, 1], [0, 1], [0, 0]]
    assert onehot.dtype == np.float32


def test_the_expansion_refuses_a_class_nobody_declared():
    with pytest.raises(MaskError, match="only 2 classes are defined"):
        codes_to_onehot(np.array([[4]]), 2)


def test_class_files_of_different_sizes_are_refused():
    with pytest.raises(MaskError, match="one labelling of one image"):
        masks_to_codes([binary([[1, 0]]), binary([[1, 0, 0]])])


def test_a_zero_threshold_rather_than_a_midpoint():
    """A midpoint is the obvious rule and would read a 0/1 label file as empty everywhere -
    a whole dataset silently background, which is the failure this default avoids."""
    codes, unmatched = masks_to_codes([np.array([[[0], [1]]]), np.array([[[1], [0]]])])
    assert codes.tolist() == [[2, 1]]
    assert unmatched == 0.0


def test_more_classes_than_one_integer_can_carry_are_refused():
    with pytest.raises(MaskError, match="62"):
        masks_to_codes([binary([[1]]) for _ in range(63)])


# ----------------------------------------------------------------- the data path


def write_case(root, name, *, size=SIZE, vessel=True):
    """A disc and a stripe, drawn as two files, overlapping where they cross.

    This is the shape the feature exists for: neither file can be derived from the other,
    and no label map can hold both without inventing a value for the crossing.
    """
    sample = root / name
    (sample / "images").mkdir(parents=True, exist_ok=True)
    (sample / "masks").mkdir(parents=True, exist_ok=True)
    yy, xx = np.ogrid[:size, :size]
    organ = (np.sqrt((yy - size // 2) ** 2 + (xx - size // 2) ** 2) < size // 3).astype(np.uint8)
    stripe = (np.abs(yy - xx) < 4).astype(np.uint8) if vessel else np.zeros((size, size), np.uint8)
    Image.fromarray(organ * 255).save(sample / "masks" / f"{name}_organ.png")
    Image.fromarray(stripe * 255).save(sample / "masks" / f"{name}_vessel.png")
    picture = np.clip((organ * 90 + stripe * 120), 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([picture] * 3, axis=-1)).save(sample / "images" / "scan.png")
    return sample


def spec_for(root, *, patterns=None, loss="cce_dice", metrics=None, **model):
    block = {
        "name": "m",
        "architecture": "u_net",
        "input_shape": [SIZE, SIZE],
        "channels": 3,
        "blocks": 2,
        "filters": 4,
        "batch_size": 2,
        "epochs": 1,
        "loss": {"name": loss},
        "metrics": metrics if metrics is not None else [{"name": "dice"}],
    }
    block.update(model)
    return from_dict(
        {
            "task": "semantic_segmentation",
            "seed": 1,
            "data": {
                "train_path": str(root),
                "validation_path": str(root),
                "mode": "nested_dirs",
                "classes_from": PATTERNS if patterns is None else patterns,
            },
            "models": [block],
        }
    )


@pytest.fixture
def two_annotations(tmp_path):
    root = tmp_path / "train"
    for index in range(4):
        write_case(root, f"case_{index}")
    return root


def target_of(spec):
    _, target = Engine(spec).dataset(spec.models[0], "train")[0]
    return target


def test_the_target_keeps_both_classes_where_they_overlap(two_annotations):
    target = target_of(spec_for(two_annotations))
    assert target.shape == (SIZE, SIZE, 2)
    vessel, organ = (target[..., c] > 0 for c in range(2))
    assert vessel.any() and organ.any()
    assert (vessel & organ).any(), "the stripe crosses the disc; that is the whole point"
    assert not np.all(target.sum(axis=-1) == 1), "not one-hot: a voxel can be in both"


def test_uniting_the_same_files_is_what_this_reader_exists_instead_of(two_annotations):
    """The control, and the argument for the feature in one assertion.

    `unite_masks` is what every other naming uses, and on these two files it answers with
    one blob: the crossing is indistinguishable from the stripe and from the disc. Read as
    classes they stay two. Without this the reader could be replaced by a validator and
    nothing would show the difference.
    """
    case = sorted((two_annotations / "case_0" / "masks").glob("*.png"))
    planes = [np.asarray(Image.open(path))[..., None] for path in case]
    united = unite_masks(planes)
    assert set(np.unique(united).tolist()) == {0, 255}, "one blob, two values, no identity"

    codes, _ = masks_to_codes(planes)
    assert set(np.unique(codes).tolist()) == {0, 1, 2, 3}, "both classes and their overlap"


def test_the_pattern_order_is_the_channel_order(two_annotations):
    """The trap `channels_from` documents, with worse consequences on a target: nothing
    downstream can tell a model trained with the vessel in channel 0 from one trained with
    the organ there, because both answer plausibly and in range."""
    forward = target_of(spec_for(two_annotations))
    reversed_ = target_of(spec_for(two_annotations, patterns=list(reversed(PATTERNS))))
    assert np.array_equal(forward[..., 0], reversed_[..., 1])
    assert forward[..., 0].sum() < forward[..., 1].sum(), "the stripe is thinner than the disc"


def test_a_pattern_matching_nothing_names_the_class_rather_than_a_channel(two_annotations):
    spec = spec_for(two_annotations, patterns=[r"_vessel\.png$", r"_tumour\.png$"])
    with pytest.raises(ChannelError, match="class 2 pattern"):
        target_of(spec)


def test_a_pattern_matching_two_files_is_refused(two_annotations):
    spec = spec_for(two_annotations, patterns=[r"_vessel\.png$", r"\.png$"])
    with pytest.raises(ChannelError, match="matches 2 files"):
        target_of(spec)


def test_a_mask_file_matching_no_pattern_is_refused(two_annotations):
    """Left out silently it would be a class the model is never shown, which is the same
    failure as a pattern matching nothing and is invisible from the other direction."""
    extra = two_annotations / "case_0" / "masks" / "case_0_nerve.png"
    Image.fromarray(np.zeros((SIZE, SIZE), np.uint8)).save(extra)
    with pytest.raises(ChannelError, match="match no class pattern"):
        target_of(spec_for(two_annotations))


def test_tiling_and_augmentation_need_to_know_nothing_about_this(two_annotations):
    """Why the coded form is one integer per voxel rather than a stack of channels.

    Both operate on a single-channel array today. A target built before them would have had
    to teach each of them a new shape; built after them, the overlap arrives through a tile
    and a flip unchanged.
    """
    spec = spec_for(
        two_annotations,
        splits=[2, 2],
        augmentation=[{"name": "HorizontalFlip", "params": {"p": 1.0}}],
    )
    dataset = Engine(spec).dataset(spec.models[0], "train", augmented=True)
    assert len(dataset) == 4 * 4, "four samples, four tiles each"
    both = 0
    for index in range(len(dataset)):
        _, target = dataset[index]
        # `input_shape` is the tile, so a 2x2 grid reads the source at twice it and cuts
        # four tiles of exactly the model's size - which is what `splits` is for.
        assert target.shape == (SIZE, SIZE, 2)
        both += int(((target[..., 0] > 0) & (target[..., 1] > 0)).sum())
    assert both > 0, "the crossing survives tiling and a flip"


def test_a_class_whose_file_is_empty_everywhere_is_reported_and_refused(tmp_path):
    """The failure this naming invites, and it is the one `missing_classes` was built for:
    a pattern that matches a file which is empty in every sample is a channel of zeros. It
    trains to predict nothing and scores a Dice of 0 while every other number looks
    ordinary."""
    root = tmp_path / "train"
    for index in range(4):
        write_case(root, f"case_{index}", vessel=False)
    spec = spec_for(root)
    report = Engine(spec).dataset(spec.models[0], "train").inspect_masks(limit=4)
    assert report.present_classes == [1] and report.missing_classes == [0]
    with pytest.raises(Exception, match="files matching"):
        Engine(spec).fit()


def test_the_unmatched_fraction_is_high_and_not_warned_about(two_annotations):
    """There is no background channel here either, so every background voxel is unmatched.
    Warning about it would fire on every correct run of this naming."""
    spec = spec_for(two_annotations)
    engine = Engine(spec)
    report = engine.dataset(spec.models[0], "train").inspect_masks(limit=4)
    assert report.unmatched > 0.5
    assert report.missing_classes == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine.fit()


# ------------------------------------------------------------------ end to end


def test_lovasz_is_refused_and_the_message_names_this_field(two_annotations):
    """The refusal arrived with `regions` and said so in its own words. A reader who wrote
    `classes_from` would have gone looking for a field they have not got."""
    with pytest.raises(Exception, match="`classes_from` makes each channel"):
        spec_for(two_annotations, loss="lovasz")


def test_the_class_fingerprint_tells_two_independent_classes_from_two_exclusive_ones(
    two_annotations,
):
    """Written into every run record and every weights sidecar. Two classes read from two
    files and two read from a colormap are the same count and the same head, and weights
    from one read back as the other label every channel wrongly."""
    fingerprint = Engine(spec_for(two_annotations))._class_fingerprint()
    assert fingerprint == {"n_class": 2, "classes_from": PATTERNS}


def test_a_run_with_classes_from_trains_predicts_and_evaluates(two_annotations):
    """The join, not the halves. Green on both sides proves neither."""
    spec = spec_for(two_annotations)
    engine = Engine(spec)
    history = engine.fit()["m"]
    assert len(history) == 1
    assert {"train_loss", "train_dice", "val_dice"} <= set(history.columns)

    prediction = engine.predict("m", "validation")[0]
    assert prediction.shape == (SIZE, SIZE, 2)
    assert prediction.sum(axis=-1).max() > 1.0, "sigmoid channels, not a softmax"

    row = engine.evaluate("validation")[0]
    assert 0.0 <= row["dice"] <= 1.0
