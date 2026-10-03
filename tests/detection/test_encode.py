"""Boxes into the tensors a YOLOv3 head predicts, and back.

The test that carries the weight is the round trip: encode a list of boxes, decode the
targets, get the same boxes. An encoder that is its decoder's inverse is right in a way
that no assertion on shapes can establish - the same reason `tile` and `stitch` are tested
against each other on the segmentation side.
"""

import numpy as np
import pytest

from pyplatypus.detection.encode import (
    COCO_ANCHORS,
    STRIDES,
    decode,
    encode,
    grid_shapes,
)
from pyplatypus.detection.metrics import DetectionError

THREE = ((0.2, 0.2), (0.3, 0.15)), ((0.1, 0.1), (0.05, 0.2)), ((0.03, 0.03), (0.02, 0.05))


# --- grids ------------------------------------------------------------------------------

def test_the_classic_grids_for_416():
    assert grid_shapes((416, 416)) == ((13, 13), (26, 26), (52, 52))


def test_a_rectangular_input_gives_rectangular_grids():
    assert grid_shapes((416, 608)) == ((13, 19), (26, 38), (52, 76))


def test_a_size_the_strides_do_not_divide_is_refused_with_the_nearest_ones():
    """Rounding instead would put every box a fraction of a cell out, which trains and
    never says so."""
    with pytest.raises(DetectionError, match="not divisible by 32"):
        grid_shapes((400, 400))
    with pytest.raises(DetectionError, match="384|416"):
        grid_shapes((400, 400))


# --- the round trip ---------------------------------------------------------------------

def as_set(boxes):
    return np.array(sorted(map(tuple, np.round(np.asarray(boxes), 4).tolist())))


@pytest.mark.parametrize("seed", range(6))
def test_encode_and_decode_are_inverses(seed):
    rng = np.random.default_rng(seed)
    boxes, labels = [], []
    for _ in range(int(rng.integers(1, 7))):
        w, h = rng.uniform(20, 120, 2)
        cx = rng.uniform(w / 2, 416 - w / 2)
        cy = rng.uniform(h / 2, 416 - h / 2)
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        labels.append(int(rng.integers(0, 3)))

    encoded = encode(boxes, labels, n_class=3)
    assert encoded.unplaced == 0, "this fixture is meant to be representable"

    out_boxes, _, out_labels = decode(encoded.targets, n_class=3)
    assert np.allclose(as_set(out_boxes), as_set(boxes), atol=1e-4)
    assert sorted(out_labels.tolist()) == sorted(labels)


def test_a_rectangular_input_round_trips_too():
    boxes = [[10, 20, 110, 90], [300, 200, 500, 380]]
    encoded = encode(boxes, [0, 1], input_shape=(416, 608), n_class=3)
    out, _, _ = decode(encoded.targets, input_shape=(416, 608), n_class=3)
    assert np.allclose(as_set(out), as_set(boxes), atol=1e-4)


def test_the_class_survives_the_round_trip():
    encoded = encode([[10, 10, 50, 50]], [2], n_class=3)
    _, _, labels = decode(encoded.targets, n_class=3)
    assert labels.tolist() == [2]


# --- the infinity the old encoding had --------------------------------------------------

def test_a_centre_on_a_cell_boundary_stays_finite():
    """The package this replaces stored `logit(centre - floor(centre))`, which is `-inf`
    when the centre lands on a cell line. Measured on a 416 input at grid 13, that was 37
    of 1200 integer-pixel boxes - roughly one in thirty with a poisoned target."""
    encoded = encode([[0.0, 0.0, 64.0, 64.0]], [0], n_class=3)
    assert all(np.isfinite(target).all() for target in encoded.targets)
    assert encoded.placed == 1

    out, _, _ = decode(encoded.targets, n_class=3)
    assert np.allclose(out, [[0.0, 0.0, 64.0, 64.0]], atol=1e-4)


def test_every_integer_pixel_box_on_a_coarse_grid_encodes_finitely():
    """The general version: sweep the positions that used to produce an infinity."""
    for xmin in range(0, 384, 8):
        for side in (16, 32, 64):
            encoded = encode([[xmin, xmin, xmin + side, xmin + side]], [0], n_class=3)
            assert all(np.isfinite(t).all() for t in encoded.targets), (xmin, side)


def test_a_box_flush_with_the_far_edge_does_not_index_past_the_grid():
    """A centre of exactly 1.0 after normalising would land one cell beyond the last."""
    encoded = encode([[400, 400, 416, 416]], [0], n_class=3)
    assert encoded.placed == 1


# --- assignment -------------------------------------------------------------------------

def test_a_large_object_goes_to_the_coarse_grid_and_a_small_one_to_the_fine():
    """Which is the whole point of three scales, and why an image of nothing but platelets
    trains only the finest grid."""
    big = encode([[0, 0, 300, 300]], [0], anchors=THREE, n_class=1)
    small = encode([[0, 0, 12, 12]], [0], anchors=THREE, n_class=1)

    assert big.targets[0][..., 4].sum() == 1          # coarsest
    assert big.targets[2][..., 4].sum() == 0
    assert small.targets[2][..., 4].sum() == 1        # finest
    assert small.targets[0][..., 4].sum() == 0


def test_each_box_takes_exactly_one_slot():
    encoded = encode([[50, 50, 150, 150]], [0], n_class=3)
    total = sum(target[..., 4].sum() for target in encoded.targets)
    assert total == 1


def test_the_number_of_anchors_per_grid_comes_from_the_anchors():
    """The flexibility the package this replaces had: the head's width is
    `anchors_per_grid * (n_class + 5)`, and both are the caller's."""
    two = encode([[10, 10, 50, 50]], [0], anchors=THREE, n_class=1)
    assert two.targets[0].shape == (13, 13, 2, 6)

    five = tuple(tuple((0.1 + 0.02 * i, 0.1) for i in range(5)) for _ in range(3))
    assert encode([[10, 10, 50, 50]], [0], anchors=five, n_class=1).targets[0].shape == \
        (13, 13, 5, 6)


# --- what cannot be represented ---------------------------------------------------------

def test_two_boxes_in_one_cell_with_one_shape_cannot_both_be_stored():
    """Counted rather than overwritten in silence. On a dataset of touching objects this
    is the difference between 'the model is poor' and 'the target never had half of them'."""
    boxes = [[100, 100, 130, 130], [101, 101, 131, 131], [102, 102, 132, 132]]
    encoded = encode(boxes, [0, 0, 0], n_class=3)
    assert encoded.placed + encoded.unplaced == 3
    assert encoded.unplaced >= 1


def test_separated_boxes_of_the_same_shape_all_fit():
    boxes = [[100, 100, 130, 130], [200, 200, 230, 230], [300, 300, 330, 330]]
    encoded = encode(boxes, [0, 0, 0], n_class=3)
    assert (encoded.placed, encoded.unplaced) == (3, 0)


def test_a_finer_input_loses_fewer_objects_at_the_same_density():
    """The measurement that makes input size a decision rather than a default: at the
    density of a blood smear, a 416 input cannot represent about one object in eleven, and
    a 608 input represents them all."""
    rng = np.random.default_rng(3)
    def pack(shape):
        boxes = []
        for _ in range(45):
            side = rng.uniform(40, 60)
            cx = rng.uniform(side / 2, shape[1] - side / 2)
            cy = rng.uniform(side / 2, shape[0] - side / 2)
            boxes.append([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
        return boxes

    coarse = encode(pack((416, 416)), [0] * 45, input_shape=(416, 416), n_class=1)
    fine = encode(pack((608, 608)), [0] * 45, input_shape=(608, 608), n_class=1)
    assert coarse.unplaced > 0
    assert fine.unplaced < coarse.unplaced


# --- decoding a network's output --------------------------------------------------------

def test_a_raw_output_decodes_through_the_sigmoid():
    """The path the published COCO weights take: offsets and objectness in logit space."""
    rng = np.random.default_rng(1)
    raw = [rng.normal(0, 1, (gh, gw, 3, 85)).astype(np.float32)
           for gh, gw in grid_shapes((416, 416))]
    boxes, scores, labels = decode(raw, n_class=80, objectness=0.9, raw=True)

    assert len(boxes) > 0
    assert np.isfinite(boxes).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert ((labels >= 0) & (labels < 80)).all()


def test_nothing_above_the_threshold_is_an_empty_answer_not_an_error():
    empty = [np.zeros((gh, gw, 3, 8)) for gh, gw in grid_shapes((416, 416))]
    boxes, scores, labels = decode(empty, n_class=3, objectness=0.5)
    assert boxes.shape == (0, 4) and scores.shape == (0,) and labels.shape == (0,)


def test_a_grid_of_the_wrong_shape_says_what_was_expected():
    wrong = [np.zeros((13, 13, 3, 8)), np.zeros((26, 26, 3, 8)), np.zeros((50, 50, 3, 8))]
    with pytest.raises(DetectionError, match=r"expected \(52, 52, 3, 8\)"):
        decode(wrong, n_class=3)


# --- refusals ---------------------------------------------------------------------------

def test_anchors_given_in_pixels_are_refused_with_the_division_named():
    """COCO's are published as pixels at 416, and using them unchanged would make every
    anchor larger than the image."""
    with pytest.raises(DetectionError, match=r"\(116, 90\) / 416"):
        encode([[10, 10, 50, 50]], [0],
               anchors=(((116, 90),), ((30, 61),), ((10, 13),)), n_class=1)


def test_unequal_anchor_counts_are_refused_because_the_head_cannot_have_two_widths():
    with pytest.raises(DetectionError, match="same number of anchors"):
        encode([[10, 10, 50, 50]], [0],
               anchors=(((0.2, 0.2), (0.3, 0.3)), ((0.1, 0.1),), ((0.05, 0.05),)),
               n_class=1)


def test_a_label_outside_the_class_count_is_refused():
    with pytest.raises(DetectionError, match="labels must be in 0..2"):
        encode([[10, 10, 50, 50]], [3], n_class=3)


def test_a_zero_sided_box_is_refused_and_points_at_the_remedy():
    with pytest.raises(DetectionError, match="drop_degenerate"):
        encode([[10, 10, 10, 50]], [0], n_class=3)


def test_mismatched_boxes_and_labels_are_refused():
    with pytest.raises(DetectionError, match="they must agree"):
        encode([[10, 10, 50, 50]], [0, 1], n_class=3)


def test_an_image_with_no_objects_gives_empty_targets_rather_than_nothing():
    encoded = encode([], [], n_class=3)
    assert encoded.shapes == ((13, 13, 3, 8), (26, 26, 3, 8), (52, 52, 3, 8))
    assert encoded.placed == 0
    assert all(target.sum() == 0 for target in encoded.targets)


def test_cocos_anchors_are_fractions_and_ordered_coarse_to_fine():
    """Largest first, matching the coarsest grid - the order the published weights expect.
    Getting this backwards loads cleanly and predicts nonsense."""
    first = np.asarray(COCO_ANCHORS[0])
    last = np.asarray(COCO_ANCHORS[-1])
    assert (first <= 1).all() and (last <= 1).all()
    assert first.prod(axis=1).mean() > last.prod(axis=1).mean()
    assert STRIDES == (32, 16, 8)
