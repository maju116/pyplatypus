"""Letterboxing, and getting a box back onto the pixels it came from."""

import numpy as np
import pytest

from pyplatypus.detection.boxes import (
    Letterbox,
    box_areas,
    clip_boxes,
    drop_degenerate,
)
from pyplatypus.detection.metrics import DetectionError


def test_a_square_image_into_a_square_input_needs_no_padding():
    fit = Letterbox.fit((100, 100), (416, 416))
    assert fit.scale == pytest.approx(4.16)
    assert (fit.pad_x, fit.pad_y) == (0, 0)


def test_a_wide_image_is_padded_top_and_bottom():
    fit = Letterbox.fit((480, 640), (416, 416))
    assert fit.scale == pytest.approx(0.65)
    assert fit.pad_x == pytest.approx(0.0)
    assert fit.pad_y == pytest.approx(52.0)


def test_the_aspect_ratio_survives():
    """The whole point. A 2:1 image stays 2:1 inside the padding."""
    fit = Letterbox.fit((200, 400), (416, 416))
    corners = fit.forward([[0, 0, 400, 200]])[0]
    assert (corners[2] - corners[0]) / (corners[3] - corners[1]) == pytest.approx(2.0)


@pytest.mark.parametrize("source", [(480, 640), (640, 480), (100, 100), (1024, 37)])
def test_forward_and_inverse_are_exact_inverses(source):
    fit = Letterbox.fit(source, (416, 416))
    height, width = source
    boxes = [[0, 0, width, height], [width / 4, height / 4, width / 2, height / 2]]
    assert np.allclose(fit.inverse(fit.forward(boxes)), boxes)


def test_a_box_in_the_padding_comes_back_inside_the_image():
    """A prediction can land in the padding, which is not a place."""
    fit = Letterbox.fit((480, 640), (416, 416))
    inside = fit.inverse([[0, 0, 416, 20]])[0]  # entirely in the top band
    assert inside[1] >= 0 and inside[3] >= 0
    assert inside[3] <= 480


def test_an_impossible_shape_is_refused():
    with pytest.raises(DetectionError, match="cannot have shape"):
        Letterbox.fit((0, 100), (416, 416))
    with pytest.raises(DetectionError, match="cannot have shape"):
        Letterbox.fit((100, 100), (416, 0))


def test_an_image_is_scaled_and_centred():
    fit = Letterbox.fit((480, 640), (416, 416))
    image = np.ones((480, 640, 3), dtype=np.float32)
    out = fit.apply_to_image(image, fill=0.5)

    assert out.shape == (416, 416, 3)
    # The padding is the fill, the middle is the image.
    assert out[0, 0, 0] == pytest.approx(0.5)
    assert out[208, 208, 0] == pytest.approx(1.0)


def test_grey_padding_rather_than_black():
    """Black is a legitimate pixel value in a radiograph, so the default must not be it:
    a model that learns 'the dark band means nothing' has learned about the padding."""
    fit = Letterbox.fit((480, 640), (416, 416))
    out = fit.apply_to_image(np.ones((480, 640, 1), dtype=np.float32))
    assert out[0, 0, 0] == pytest.approx(0.5)


def test_grey_is_refused_where_grey_is_unknown():
    """`pyplatypus#453`: the default is grey on a 0-1 scale and on no other, so an 8-bit
    photograph padded with it gets a nearly black band - under the very docstring that
    argues against black. Refused rather than guessed at, because the value that means
    nothing differs by unit: 127.5 for 8-bit, and for a CT the midpoint of the stored
    range is bone.

    Asked of `apply_to_image` directly as well as through `crop_boxes`, because this is
    the entry point `DetectionDataset.read` uses and the two silences are separate code.
    """
    fit = Letterbox.fit((480, 640), (416, 416))
    with pytest.raises(DetectionError, match="grey on a 0-1 scale"):
        fit.apply_to_image(np.full((480, 640, 3), 200, dtype=np.uint8))

    # And the value given explicitly still means the image's own units.
    out = fit.apply_to_image(np.full((480, 640, 3), 200, dtype=np.uint8), fill=127.5)
    assert out[0, 0, 0] == pytest.approx(127.5)
    assert out[208, 208, 0] == pytest.approx(200.0)


def test_a_boolean_mask_is_already_on_a_zero_to_one_scale():
    """The one integer-like dtype whose scale really is 0-1: True and False, with 0.5
    between them. Refusing it would be a check failing on correct content, and asking
    `np.iinfo` about it raises - so it is answered before the dtype rule, not by it."""
    fit = Letterbox.fit((100, 80), (64, 64))
    out = fit.apply_to_image(np.ones((100, 80), dtype=bool))
    assert out[0, 0, 0] == pytest.approx(0.5)
    assert out[32, 32, 0] == pytest.approx(1.0)


def test_a_letterbox_refuses_an_image_it_was_not_fitted_for():
    """Reusing one is how a box ends up scaled by the wrong factor."""
    fit = Letterbox.fit((480, 640), (416, 416))
    with pytest.raises(DetectionError, match="fit a new one"):
        fit.apply_to_image(np.ones((100, 100, 3), dtype=np.float32))


def test_a_two_dimensional_image_is_treated_as_one_channel():
    fit = Letterbox.fit((100, 100), (64, 64))
    assert fit.apply_to_image(np.ones((100, 100), dtype=np.float32)).shape == (64, 64, 1)


def test_areas_and_clipping():
    assert box_areas([[0, 0, 10, 10], [0, 0, 2, 5]]).tolist() == [100.0, 10.0]
    assert clip_boxes([[-5, -5, 200, 200]], (100, 100)).tolist() == [[0, 0, 100, 100]]


def test_degenerate_boxes_are_dropped_and_counted():
    """A letterbox that shrinks by 8 can take a 4-pixel platelet below one pixel, at which
    point training on it teaches the model that nothing is something."""
    boxes = [[0, 0, 10, 10], [5, 5, 5.2, 5.2], [20, 20, 40, 40]]
    kept, labels, dropped = drop_degenerate(boxes, [0, 1, 2])
    assert dropped == 1
    assert kept.shape == (2, 4)
    assert labels.tolist() == [0, 2]


def test_dropping_keeps_boxes_and_labels_together():
    """The failure this guards is silent: labels that no longer match their boxes train a
    model to call a red cell a platelet, with no error anywhere."""
    boxes = [[0, 0, 0.1, 0.1], [0, 0, 10, 10]]
    kept, labels, _ = drop_degenerate(boxes, [7, 3])
    assert labels.tolist() == [3]
    assert kept[0].tolist() == [0, 0, 10, 10]


def test_mismatched_boxes_and_labels_are_refused():
    with pytest.raises(DetectionError, match="they must agree"):
        drop_degenerate([[0, 0, 1, 1]], [0, 1])


def test_the_whole_image_lands_exactly_on_the_rectangle_the_letterbox_pastes():
    """`pyplatypus#336`, and it is the invariant the class's own docstring asks for.

    The paste rounded the inner size and the offset for the pixel grid while `forward` used
    the unrounded offset and the unrounded `scale`, so image and boxes disagreed by up to a
    pixel. The transform now names what the grid did, and the test is the strongest form of
    the claim: the source image's OWN bounding box, pushed through `forward`, is exactly the
    rectangle `apply_to_image` fills.
    """
    source, target = (300, 417), (416, 416)
    fit = Letterbox.fit(source, target)
    source_h, source_w = source
    inner_h, inner_w = fit.inner_shape

    whole = fit.forward(np.array([[0.0, 0.0, float(source_w), float(source_h)]]))[0]
    assert whole.tolist() == pytest.approx(
        [fit.pad_x, fit.pad_y, fit.pad_x + inner_w, fit.pad_y + inner_h], abs=1e-9
    ), f"the image's own box does not land on the pasted rectangle: {whole.tolist()}"

    # The offsets have to be whole pixels, because the paste can only use whole ones. This
    # is what used to be 58.359... against a paste at 58.
    assert float(fit.pad_x).is_integer() and float(fit.pad_y).is_integer()

    # And the paste really does fill that rectangle, so the comparison above is about the
    # picture rather than about two readings of the same arithmetic.
    image = np.ones((source_h, source_w, 3), dtype=np.float32)
    out = fit.apply_to_image(image, fill=0.0)
    rows = np.where(out[:, :, 0].max(axis=1) > 0.5)[0]
    columns = np.where(out[:, :, 0].max(axis=0) > 0.5)[0]
    assert (rows.min(), rows.max() + 1) == (int(fit.pad_y), int(fit.pad_y) + inner_h)
    assert (columns.min(), columns.max() + 1) == (int(fit.pad_x), int(fit.pad_x) + inner_w)

    # The control: a shape whose padding is already whole, which agreed before the fix as
    # well - so the assertions above cannot be passing because both sides were changed into
    # something else together.
    square = Letterbox.fit((200, 400), (400, 400))
    assert (square.pad_x, square.pad_y) == (0.0, 100.0)
    assert square.axis_scales == (1.0, 1.0)
    assert square.forward(np.array([[0.0, 0.0, 400.0, 200.0]]))[0].tolist() == [
        0.0,
        100.0,
        400.0,
        300.0,
    ]


def test_a_box_still_returns_to_its_own_pixels_through_the_inverse():
    """The round trip is what the stored transform is for, and it has to survive #336."""
    fit = Letterbox.fit((300, 417), (416, 416))
    boxes = np.array(
        [[10.0, 20.0, 110.0, 120.0], [0.0, 0.0, 417.0, 300.0], [200.5, 100.25, 300.0, 299.0]]
    )
    back = fit.inverse(fit.forward(boxes))
    assert back == pytest.approx(boxes, abs=1e-9)


# `pyplatypus#456`: THREE SENTENCES OF THE REFUSAL DESCRIBED THE ARRAY WRONGLY, and every
# one of the three refused correctly. The fixtures that existed were `uint8` and 0-1 floats,
# so none of the three could fail: §4's fixture that never contained the case, in a message
# §4 also says is the place a reader learns what a healthy run looks like.


def test_a_signed_integer_image_is_told_its_own_dtypes_range():
    """`int16`'s scale is -32768 to 32767, and the message said `0 to 32767`.

    The sharpest part is that the refusal's own reason is *the midpoint of a CT's stored
    range is bone*, and a CT in Hounsfield units is exactly an array with negative values -
    so the sentence named a range that excludes every value the case it argues from holds.
    """
    hounsfield = np.full((8, 12, 1), -1000, dtype=np.int16)
    hounsfield[2:5, 3:7, 0] = 400
    with pytest.raises(DetectionError) as refusal:
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(hounsfield)
    message = str(refusal.value)
    assert "int16, whose scale is -32768 to 32767" in message
    # Its own values beside the dtype's, because the two answer different questions.
    assert "holds values -1000 to 400" in message
    # And NO midpoint is offered: for physical units the windowing decides what grey is, so
    # there is nothing honest to suggest. 127.5 belongs to the 8-bit sentence and must not
    # travel to this one.
    assert "127.5" not in message
    assert "window it" in message


def test_an_unsigned_integer_image_still_gets_the_eight_bit_sentence():
    """The control for the test above: the signed branch must not swallow the common case."""
    photograph = np.full((8, 12, 3), 200, dtype=np.uint8)
    with pytest.raises(DetectionError) as refusal:
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(photograph)
    message = str(refusal.value)
    assert "uint8, whose scale is 0 to 255" in message
    assert "127.5" in message
    assert "whose scale is -" not in message


def test_a_non_finite_value_is_refused_by_name_rather_than_as_a_scale():
    """`min()` and `max()` of an array holding a `NaN` are `nan`, so the old message said
    *holding values nan to nan* about an array whose other values are 0.2 - and offered
    *divide by 255*, which is no advice for a `NaN`."""
    patchy = np.full((8, 12, 1), 0.2, dtype=np.float32)
    patchy[3, 4, 0] = np.nan
    with pytest.raises(DetectionError) as refusal:
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(patchy)
    message = str(refusal.value)
    assert "non-finite" in message
    assert "nan to nan" not in message
    # The finite values ARE named, because they are what the reader can act on.
    assert "0.2" in message
    # An infinity is the same complaint, which the finding did not name and the fix covers.
    patchy[3, 4, 0] = np.inf
    with pytest.raises(DetectionError, match="non-finite"):
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(patchy)


def test_an_array_that_is_entirely_non_finite_says_so():
    """The other end of the same branch: with no finite value there is no range to name,
    and a message reporting an empty one would be the defect again."""
    nothing = np.full((8, 12, 1), np.nan, dtype=np.float32)
    with pytest.raises(DetectionError) as refusal:
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(nothing)
    assert "no finite value at all" in str(refusal.value)


def test_a_float_refused_for_overshooting_one_is_told_the_overshoot():
    """`:.4g` rounded 1.000002 to `1`, so the refusal described the array as *holding values
    0 to 1* - a range this function ACCEPTS. The reader was told the image is on the scale
    the refusal says it is not."""
    overshooting = np.linspace(0.0, 1 + 2e-6, 8 * 12, dtype=np.float64).reshape(8, 12, 1)
    with pytest.raises(DetectionError) as refusal:
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(overshooting)
    message = str(refusal.value)
    assert "1.000002" in message
    assert "holding values 0 to 1." in message
    # The control that the boundary itself did not move: one order of magnitude smaller is
    # still the same scale and is accepted.
    accepted = np.linspace(0.0, 1 + 1e-7, 8 * 12, dtype=np.float64).reshape(8, 12, 1)
    assert Letterbox.fit((8, 12), (6, 6)).apply_to_image(accepted).min() >= 0.0


def test_a_float_a_hair_below_zero_is_still_a_zero_to_one_image():
    """The tolerance has two ends and after `0.8.0a82` only one of them was held.

    `pyplatypus#462`, and **the history this docstring gave was wrong**: `pyplatypus#464`'s
    pass measured it, and the lower end was never held at all. The sixteenth pass's "lower
    end" mutation edited BOTH ends in one `sed`, and what caught it was the upper end's own
    test - `tests/detection/test_crops.py::test_float_rounding_at_one_is_not_a_different_scale`,
    which has been there since a81. Measured on three trees: the lower end alone survives at
    a81 and at a82 and fails only here; the upper end alone is caught at a81; the coarse
    upper bound, `high <= 1.9`, survives at a81 and is caught from a82 on.

    So `0.8.0a82` unheld nothing. It added a real hold on the upper end at the coarse scale,
    and this test is the first the lower end has ever had. The commit that shipped it says
    *"one of which a82 unheld itself"* in its subject and cannot be changed; this is the
    correction.

    Why a value below zero is a 0-1 image rather than a different scale: float32's eps at 1.0
    is 1.19e-07, so a pixel that is 0 up to its own rounding can read as -1e-7. Same argument
    as the upper end, in the direction nobody wrote a fixture for.
    """
    just_under = np.linspace(-1e-7, 1.0, 8 * 12, dtype=np.float64).reshape(8, 12, 1)
    padded = Letterbox.fit((8, 12), (6, 6)).apply_to_image(just_under)
    # Row 0 is entirely padding for this shape, which is what makes this an assertion about
    # the pad value rather than about whether 0.5 occurs anywhere in a resized picture.
    assert float(padded[0].min()) == pytest.approx(0.5)
    assert float(padded[0].max()) == pytest.approx(0.5)

    # The control on the other side of the same bound: two orders of magnitude past the
    # tolerance is a different scale and is refused, with its own figure in the message so
    # that the reader is not told a range this function accepts.
    well_under = np.linspace(-1e-5, 1.0, 8 * 12, dtype=np.float64).reshape(8, 12, 1)
    with pytest.raises(DetectionError) as refusal:
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(well_under)
    assert f"{float(well_under.min()):.9g}" in str(refusal.value)


def test_an_unsigned_integer_mask_is_judged_by_its_dtype_rather_than_by_what_it_holds():
    """*0.5 is not the middle of 0-255, whatever one picture happens to hold.*

    `0.8.0a82`'s changelog calls that the truth of the matter, and until `pyplatypus#462` no
    fixture reached it: there is no `uint8` image anywhere in this suite whose values lie in
    0-1. Such an array is a **mask**, which is exactly why the rule is worth pinning rather
    than left to whoever decides the values look more informative than the dtype - mutating
    the integer branch to answer 0.5 whenever an integer array's values are within 0-1 passed
    all 275 tests in `tests/detection` before this test existed.
    """
    mask = np.zeros((8, 12, 1), dtype=np.uint8)
    mask[2:5, 3:7, 0] = 1
    # The case nothing reached, asserted so that a rewritten fixture cannot quietly stop
    # being this case.
    assert (int(mask.min()), int(mask.max())) == (0, 1)

    with pytest.raises(DetectionError) as refusal:
        Letterbox.fit((8, 12), (6, 6)).apply_to_image(mask)
    # The dtype's scale rather than the values', because for an integer array the dtype IS
    # the scale - and the sentence a reader acts on says so.
    assert "uint8, whose scale is 0 to 255" in str(refusal.value)

    # The control, and it is the DISTINCTION rather than a second case: the same mask as
    # `bool` is a 0-1 scale by its own dtype, so silence is answered with 0.5 there. A rule
    # that refused both would be wrong in the other direction, and this is the only place
    # the two sit side by side.
    padded = Letterbox.fit((8, 12), (6, 6)).apply_to_image(mask.astype(bool))
    assert float(padded[0].min()) == pytest.approx(0.5)
