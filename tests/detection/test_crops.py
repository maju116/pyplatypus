"""`crop_boxes`: a detection cut out of the photograph it was found in.

Every assertion here is about *content* rather than shape. A crop of the right size taken
from the wrong place passes a shape check, trains a downstream classifier on nonsense, and
reports nothing - which is the same failure a mask once had when it was read through a
different path from its image.
"""

import numpy as np
import pytest

from pyplatypus.detection import crop_boxes
from pyplatypus.detection.metrics import DetectionError


@pytest.fixture
def gradient():
    """Every pixel a different value, so a crop's position is readable from its contents."""
    return np.arange(20 * 30, dtype=np.float32).reshape(20, 30, 1) / (20 * 30)


def test_a_crop_is_the_part_of_the_image_the_box_names(gradient):
    crop = crop_boxes(gradient, [[5, 4, 9, 7]])[0]
    np.testing.assert_array_equal(crop, gradient[4:7, 5:9])


def test_several_boxes_come_back_in_the_order_given(gradient):
    crops = crop_boxes(gradient, [[10, 2, 14, 6], [1, 1, 3, 3]])
    np.testing.assert_array_equal(crops[0], gradient[2:6, 10:14])
    np.testing.assert_array_equal(crops[1], gradient[1:3, 1:3])


def test_a_fractional_box_is_rounded_outward_so_the_object_survives(gradient):
    """Rounding to nearest would lose up to a pixel on each side. On a 26-pixel platelet
    that is a twelfth of the object, and a classifier sees a clipped one."""
    crop = crop_boxes(gradient, [[5.6, 4.2, 8.1, 6.9]])[0]
    # floor(5.6)=5 .. ceil(8.1)=9  and  floor(4.2)=4 .. ceil(6.9)=7
    np.testing.assert_array_equal(crop, gradient[4:7, 5:9])


def test_a_box_reaching_past_the_frame_is_cut_at_the_frame(gradient):
    crop = crop_boxes(gradient, [[26, 17, 40, 40]])[0]
    np.testing.assert_array_equal(crop, gradient[17:20, 26:30])


def test_context_grows_with_the_box_rather_than_by_a_fixed_number_of_pixels(gradient):
    """One value has to suit a platelet and a white cell, so it is a fraction of the box."""
    small = crop_boxes(gradient, [[10, 10, 12, 12]], context=0.5)[0]
    large = crop_boxes(gradient, [[10, 6, 16, 12]], context=0.5)[0]
    assert small.shape[:2] == (4, 4)       # 2px box, 1px added each side
    assert large.shape[:2] == (12, 12)     # 6px box, 3px each side
    np.testing.assert_array_equal(small, gradient[9:13, 9:13])
    np.testing.assert_array_equal(large, gradient[3:15, 7:19])


def test_context_of_zero_is_the_default_and_changes_nothing(gradient):
    np.testing.assert_array_equal(crop_boxes(gradient, [[5, 5, 9, 9]])[0],
                                  crop_boxes(gradient, [[5, 5, 9, 9]], context=0.0)[0])


# --- bringing them to one size -----------------------------------------------------------

def test_letterbox_preserves_the_aspect_and_pads_the_rest(gradient):
    """A 4-wide, 16-tall crop into a square: scaled by 12/16, so 3 columns of content and
    9 of padding. Asserted as a count rather than a shape, because the shape is 12x12
    either way and that is exactly what a stretch would also produce."""
    crop = crop_boxes(gradient, [[4, 2, 8, 18]], size=(12, 12), fill=0.5)[0]
    assert crop.shape == (12, 12, 1)
    padded = (crop[:, :, 0] == 0.5).all(axis=0).sum()
    assert padded == 9


def test_stretch_fills_the_frame_and_has_to_be_asked_for(gradient):
    crop = crop_boxes(gradient, [[4, 2, 8, 18]], size=(12, 12), fit="stretch", fill=0.5)[0]
    assert crop.shape == (12, 12, 1)
    assert (crop[:, :, 0] == 0.5).all(axis=0).sum() == 0


def test_every_crop_is_the_same_shape_once_size_is_given_but_still_a_list(gradient):
    """The return type does not depend on an argument: a caller who wants a batch writes
    `np.stack(crops)` and one who does not is never handed an array they must unpack."""
    crops = crop_boxes(gradient, [[1, 1, 3, 9], [10, 2, 20, 4]], size=(8, 8))
    assert isinstance(crops, list)
    assert all(c.shape == (8, 8, 1) for c in crops)
    assert np.stack(crops).shape == (2, 8, 8, 1)


def test_a_greyscale_image_without_a_channel_axis_is_accepted(gradient):
    flat = gradient[..., 0]
    crop = crop_boxes(flat, [[5, 4, 9, 7]])[0]
    assert crop.shape == (3, 4, 1)
    np.testing.assert_array_equal(crop[..., 0], flat[4:7, 5:9])


# --- refusals ----------------------------------------------------------------------------

def test_a_box_with_nothing_inside_the_frame_is_refused_by_index(gradient):
    """Clipping silently would hand back an empty array. A box outside the frame cannot
    have come from `predict` on this image, so something upstream is wrong."""
    with pytest.raises(DetectionError, match="box 1 is"):
        crop_boxes(gradient, [[1, 1, 5, 5], [40, 40, 50, 50]])


def test_an_unknown_fit_is_refused_with_both_names(gradient):
    with pytest.raises(DetectionError, match="'letterbox' or 'stretch'"):
        crop_boxes(gradient, [[1, 1, 5, 5]], size=(8, 8), fit="squash")


def test_negative_context_is_refused_rather_than_shrinking_the_box(gradient):
    with pytest.raises(DetectionError, match="context cannot be negative"):
        crop_boxes(gradient, [[1, 1, 5, 5]], context=-0.1)


def test_an_image_of_the_wrong_rank_is_refused(gradient):
    with pytest.raises(DetectionError, match="2 or 3 dimensional"):
        crop_boxes(gradient[None], [[1, 1, 5, 5]])
