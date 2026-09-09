import numpy as np
import pytest

from pyplatypus.data import classes_to_onehot, colours_to_classes, onehot_to_colours, unite_masks
from pyplatypus.data.masks import MaskError

BINARY = [(0, 0, 0), (255, 255, 255)]


def test_unite_merges_one_file_per_object():
    """The Data Science Bowl stores one mask file per nucleus; they have to become one."""
    a = np.zeros((4, 4, 3), np.uint8); a[0] = 255
    b = np.zeros((4, 4, 3), np.uint8); b[2] = 255
    united = unite_masks([a, b])
    assert united[0].max() == 255 and united[2].max() == 255
    assert united[1].max() == 0


def test_unite_rejects_mismatched_shapes():
    with pytest.raises(MaskError, match="same size"):
        unite_masks([np.zeros((4, 4, 3)), np.zeros((8, 8, 3))])


def test_colours_become_class_indices():
    mask = np.zeros((4, 4, 3), np.uint8)
    mask[2:] = 255
    classes, unmatched = colours_to_classes(mask, BINARY)
    assert classes.shape == (4, 4)
    assert classes[0, 0] == 0 and classes[3, 3] == 1
    assert unmatched == pytest.approx(0.0)


def test_unmatched_colours_are_reported_not_hidden():
    """A colormap that does not describe the data is the quietest way to train on nothing."""
    mask = np.full((4, 4, 3), 77, np.uint8)
    classes, unmatched = colours_to_classes(mask, BINARY)
    assert unmatched == pytest.approx(1.0)
    assert (classes == 0).all()


def test_onehot_round_trip():
    classes = np.array([[0, 1], [1, 0]])
    onehot = classes_to_onehot(classes, 2)
    assert onehot.shape == (2, 2, 2)
    assert np.array_equal(onehot_to_colours(onehot, BINARY)[0, 1], (255, 255, 255))


def test_class_index_beyond_the_colormap_is_caught():
    with pytest.raises(MaskError, match="only defines 2"):
        classes_to_onehot(np.array([[0, 5]]), 2)


def test_masks_work_in_3d():
    """Nothing in here indexes [:, :], so a volume behaves like an image."""
    volume = np.zeros((4, 4, 4, 3), np.uint8)
    volume[2:] = 255
    classes, _ = colours_to_classes(volume, BINARY)
    assert classes.shape == (4, 4, 4)
    assert classes_to_onehot(classes, 2).shape == (4, 4, 4, 2)
