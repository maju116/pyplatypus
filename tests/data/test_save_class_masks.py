"""One picture per class, which is what `onehot_to_colours`'s refusal points at.

The pair is the whole point of these tests: the refusal has been in the package since
`regions` shipped in 0.8.0a14 and said "overlapping classes are written one file per class",
and for nine releases there was no such writer in either package. So the first test here is
the two halves meeting - take a prediction a colour map cannot hold, watch it be refused,
then write it the way the refusal says to.
"""

import numpy as np
import pytest
from PIL import Image

from pyplatypus import save_class_masks
from pyplatypus.data.masks import MaskError, class_filename, onehot_to_colours

NAMES = ["whole tumour", "tumour core", "enhancing"]


def brats_like() -> np.ndarray:
    """Three nested regions, channels-last: every `enhancing` voxel is in all three.

    Built the way the data is - a label map of `{0, 1, 2, 4}` turned into unions - rather
    than by setting channels directly, so the overlap is a consequence rather than an
    assertion about one.
    """
    labels = np.zeros((4, 4), np.int32)
    labels[0, :] = 2  # oedema: whole tumour only
    labels[1, :] = 1  # non-enhancing core: whole tumour and core
    labels[2, :] = 4  # enhancing: all three
    channels = [np.isin(labels, group) for group in ([1, 2, 4], [1, 4], [4])]
    return np.stack(channels, axis=-1).astype(np.float32)


def test_a_colour_map_refuses_this_prediction_and_names_what_to_do():
    """The premise. Without it the rest of this file is testing a writer nobody needs."""
    with pytest.raises(MaskError, match="one file per class"):
        onehot_to_colours(brats_like(), [(0, 0, 0), (255, 0, 0), (0, 255, 0)])


def test_one_file_per_class_and_each_holds_its_own_channel(tmp_path):
    written = save_class_masks(brats_like(), tmp_path, names=NAMES)

    assert [path.name for path in written] == [
        "mask_0001_whole_tumour.png",
        "mask_0001_tumour_core.png",
        "mask_0001_enhancing.png",
    ]
    # Read back rather than trusting the return value: this project has written a test that
    # asserted on a return while the files on disk were wrong.
    rows = {path.name: np.asarray(Image.open(path)) for path in written}
    assert rows["mask_0001_whole_tumour.png"][1].max() == 255
    assert rows["mask_0001_tumour_core.png"][1].max() == 255
    assert rows["mask_0001_enhancing.png"][1].max() == 0
    # The voxel that is in all three is the one a colour map could not hold.
    assert all(rows[name][2].min() == 255 for name in rows)
    assert all(rows[name][3].max() == 0 for name in rows)


def test_the_pictures_are_binary_and_not_a_probability(tmp_path):
    soft = np.full((2, 2, 2), 0.7, np.float32)
    soft[..., 1] = 0.2
    written = save_class_masks(soft, tmp_path, names=["a", "b"])
    first, second = (np.asarray(Image.open(path)) for path in written)
    assert set(np.unique(first)) == {255} and set(np.unique(second)) == {0}


def test_the_threshold_is_the_one_the_metrics_use_and_can_be_moved(tmp_path):
    soft = np.full((2, 2, 1), 0.3, np.float32)
    assert np.asarray(Image.open(save_class_masks(soft, tmp_path, names=["a"])[0])).max() == 0
    moved = save_class_masks(soft, tmp_path, names=["a"], threshold=0.2, suffix="_low")
    assert np.asarray(Image.open(moved[0])).max() == 255


def test_a_channel_nothing_reaches_is_an_empty_picture_rather_than_a_missing_file(tmp_path):
    """A missing file says the run did not produce that class; an empty one says it did and
    the answer was nothing. Only one of those is true, and they are different findings."""
    written = save_class_masks(np.zeros((2, 2, 3), np.float32), tmp_path, names=NAMES)
    assert len(written) == 3
    assert all(path.exists() for path in written)


def test_a_stack_writes_every_image_and_takes_its_case_names(tmp_path):
    stack = np.stack([brats_like(), brats_like()])
    written = save_class_masks(stack, tmp_path, names=NAMES, cases=["TCGA-01", "TCGA-02"])
    assert len(written) == 6
    assert [path.name for path in written][:1] == ["TCGA-01_whole_tumour.png"]
    assert [path.name for path in written][3:4] == ["TCGA-02_whole_tumour.png"]


def test_the_suffix_lets_two_models_share_a_directory(tmp_path):
    save_class_masks(brats_like(), tmp_path, names=NAMES, suffix="_unet")
    save_class_masks(brats_like(), tmp_path, names=NAMES, suffix="_linknet")
    assert (tmp_path / "mask_0001_unet_whole_tumour.png").exists()
    assert (tmp_path / "mask_0001_linknet_whole_tumour.png").exists()


def test_the_directory_is_created(tmp_path):
    target = tmp_path / "predictions" / "brats"
    save_class_masks(brats_like(), target, names=NAMES)
    assert (target / "mask_0001_enhancing.png").exists()


def test_a_name_per_channel_or_it_refuses():
    with pytest.raises(MaskError, match="3 channels and 2 names"):
        save_class_masks(brats_like(), "unused", names=["a", "b"])


def test_two_names_that_become_one_filename_are_refused(tmp_path):
    """The failure this would otherwise produce is the quiet kind: one file holding the
    second channel, the first silently gone, and the count of returned paths still right."""
    with pytest.raises(MaskError, match="same filename"):
        save_class_masks(brats_like(), tmp_path, names=["core 1", "core_1", "other"])
    assert not list(tmp_path.iterdir())


def test_a_case_per_image_or_it_refuses(tmp_path):
    stack = np.stack([brats_like(), brats_like()])
    with pytest.raises(MaskError, match="2 predictions and 1 case"):
        save_class_masks(stack, tmp_path, names=NAMES, cases=["only-one"])


def test_a_volume_stack_is_refused_because_a_png_carries_no_geometry(tmp_path):
    with pytest.raises(MaskError, match="voxel spacing"):
        save_class_masks(np.zeros((2, 3, 4, 4, 3), np.float32), tmp_path, names=NAMES)


def test_a_rank_that_is_neither_is_refused(tmp_path):
    with pytest.raises(MaskError, match="channels-last"):
        save_class_masks(np.zeros((4, 4), np.float32), tmp_path, names=["a"])


def test_a_region_name_becomes_a_filename_without_being_lowercased():
    assert class_filename("whole tumour") == "whole_tumour"
    assert class_filename("Core / rim") == "Core_rim"
    assert class_filename("oedema (T2)") == "oedema_T2"


def test_a_name_with_nothing_a_filename_can_hold_is_refused():
    with pytest.raises(MaskError, match="nothing to call its file"):
        class_filename("///")
