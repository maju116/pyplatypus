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
