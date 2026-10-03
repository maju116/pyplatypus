"""Anchors fitted to your own boxes.

The test that carries the weight is at the bottom: plant COCO's nine anchors as clusters
of boxes, refit from scratch, and check the result is COCO's anchors **in COCO's published
grouping**. That verifies the k-means++ seeding, the IoU distance, the convergence and the
grouping rule at once, against the one external reference available - and it is a test the
old implementation could not have had, because it grouped by width and would fail it.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from pyplatypus.detection.anchors import (
    anchor_coverage,
    box_shapes,
    fit_shapes,
    generate_anchors,
)
from pyplatypus.detection.encode import COCO_ANCHORS
from pyplatypus.detection.metrics import DetectionError, iou_matrix


@dataclass
class Annotated:
    """Enough of an `Annotation` for these functions, without reading a file."""

    boxes: np.ndarray
    width: int
    height: int


def clustered(centres, *, per_centre=80, shape=(416, 416), spread=0.04, seed=0):
    """Boxes scattered around known (width, height) fractions of `shape`."""
    rng = np.random.default_rng(seed)
    height, width = shape
    boxes = []
    for centre_w, centre_h in centres:
        for _ in range(per_centre):
            box_w = np.clip(centre_w * rng.normal(1, spread), 1e-3, 0.99) * width
            box_h = np.clip(centre_h * rng.normal(1, spread), 1e-3, 0.99) * height
            x = rng.uniform(0, width - box_w)
            y = rng.uniform(0, height - box_h)
            boxes.append([x, y, x + box_w, y + box_h])
    return [Annotated(np.asarray(boxes, dtype=float), width, height)]


def shape_iou(a, b):
    return iou_matrix([[0, 0, a[0], a[1]]], [[0, 0, b[0], b[1]]])[0, 0]


# --- the distance ----------------------------------------------------------------------

def test_euclidean_distance_is_blind_to_scale_and_overlap_is_not():
    """The reason for the distance, as a pair of cases Euclidean cannot separate.

    The first draft of this test asserted something else - that Euclidean would rank a
    300x10 box closer to a 300x300 one than a 310x310 box is - and propped it up with
    `or True` because it would not pass. It would not pass because it was false: Euclidean
    ranks that case correctly. The real failure is scale, and it is sharper.
    """
    small, small_double = (0.02, 0.02), (0.04, 0.04)
    large, large_nudged = (0.50, 0.50), (0.52, 0.52)

    # Identical Euclidean distance.
    assert np.hypot(*np.subtract(small, small_double)) == \
        pytest.approx(np.hypot(*np.subtract(large, large_nudged)))

    # Entirely different as anchors: a fourfold area error against an eight per cent one.
    assert shape_iou(small, small_double) == pytest.approx(0.25, abs=0.01)
    assert shape_iou(large, large_nudged) > 0.9


# --- recovering what was planted --------------------------------------------------------

def test_k_means_recovers_shapes_it_was_given():
    planted = [(0.05, 0.05), (0.10, 0.20), (0.30, 0.15),
               (0.40, 0.40), (0.15, 0.08), (0.60, 0.25)]
    annotations = clustered(planted, per_centre=120, spread=0.06, seed=0)
    shapes = box_shapes(annotations)

    centres, mean_iou, _, converged = fit_shapes(shapes, len(planted), seed=1)
    assert converged
    assert mean_iou > 0.85

    by_area = sorted(map(tuple, centres.tolist()), key=lambda p: -p[0] * p[1])
    expected = sorted(planted, key=lambda p: -p[0] * p[1])
    for recovered, truth in zip(by_area, expected):
        assert shape_iou(recovered, truth) > 0.95, (recovered, truth)


def test_more_anchors_describe_the_data_better():
    """The property that makes `mean_iou` the number to choose the count by, rather than
    taking three per grid because YOLOv3 did."""
    annotations = clustered([(0.06, 0.06)], per_centre=600, spread=0.25, seed=0)
    shapes = box_shapes(annotations)
    scores = [fit_shapes(shapes, k, seed=1)[1] for k in (3, 6, 9, 12)]
    assert scores == sorted(scores), scores


def test_the_same_seed_gives_the_same_anchors():
    annotations = clustered([(0.1, 0.1), (0.3, 0.3)], seed=0)
    shapes = box_shapes(annotations)
    first = fit_shapes(shapes, 4, seed=7)[0]
    second = fit_shapes(shapes, 4, seed=7)[0]
    assert np.allclose(first, second)


def test_a_different_seed_can_differ_but_still_fits():
    annotations = clustered([(0.1, 0.1), (0.3, 0.3), (0.5, 0.2)], seed=0)
    shapes = box_shapes(annotations)
    assert fit_shapes(shapes, 3, seed=1)[1] > 0.8
    assert fit_shapes(shapes, 3, seed=99)[1] > 0.8


# --- the space the shapes are measured in -----------------------------------------------

def test_shapes_are_fractions_of_the_input_not_of_the_source():
    """The old code divided by the source image, which equals the input fraction only when
    the image is stretched to fill. A 100x100 object in a 640x480 image is square after
    letterboxing and a third taller than it is wide under source normalisation."""
    annotations = [Annotated(np.asarray([[100.0, 100.0, 200.0, 200.0]]), 640, 480)]

    letterboxed = box_shapes(annotations, input_shape=(416, 416))[0]
    sourced = box_shapes(annotations, input_shape=(416, 416), letterbox=False)[0]

    assert letterboxed[0] == pytest.approx(letterboxed[1])          # square stays square
    assert sourced[1] / sourced[0] == pytest.approx(640 / 480)      # and this does not
    assert abs(sourced[1] - letterboxed[1]) / letterboxed[1] > 0.3


def test_a_square_source_makes_the_two_agree():
    """Which is why a dataset of one size hid this: the distortion was uniform."""
    annotations = [Annotated(np.asarray([[0.0, 0.0, 100.0, 50.0]]), 416, 416)]
    assert np.allclose(box_shapes(annotations), box_shapes(annotations, letterbox=False))


def test_images_with_no_boxes_contribute_nothing_rather_than_breaking():
    annotations = [Annotated(np.zeros((0, 4)), 640, 480),
                   Annotated(np.asarray([[0.0, 0.0, 64.0, 64.0]]), 640, 480)]
    assert box_shapes(annotations).shape == (1, 2)


# --- the report rather than a plot ------------------------------------------------------

def test_the_result_is_data_and_carries_the_evidence():
    """The old `generate_anchors` printed a class count and drew a scatter plot as side
    effects, and returned a nested list with no indication of whether it had worked."""
    annotations = clustered([(0.05, 0.05), (0.2, 0.2), (0.5, 0.4)], seed=0)
    fit = generate_anchors(annotations, anchors_per_grid=1, scales=3, seed=1)

    assert fit.mean_iou > 0.8
    assert fit.boxes_used == 240
    assert fit.converged is True
    assert len(fit.per_anchor) == 3
    assert {row["grid"] for row in fit.per_anchor} == {0, 1, 2}
    assert all(row["boxes"] > 0 for row in fit.per_anchor)
    # And in pixels, because an anchor in fractions is hard to sanity-check by eye.
    assert all(row["width_pixels"] > 1 for row in fit.per_anchor)


def test_anchors_per_grid_and_scales_are_the_callers(tmp_path):
    annotations = clustered([(0.05, 0.05), (0.15, 0.15), (0.3, 0.3), (0.5, 0.5)], seed=0)
    fit = generate_anchors(annotations, anchors_per_grid=2, scales=2, seed=1)
    assert len(fit.anchors) == 2
    assert all(len(group) == 2 for group in fit.anchors)
    assert fit.flat.shape == (4, 2)


def test_boxes_too_small_after_letterboxing_are_dropped_and_counted():
    annotations = [Annotated(np.asarray([
        [0.0, 0.0, 100.0, 100.0],
        [0.0, 0.0, 0.5, 0.5],        # sub-pixel once the letterbox shrinks it
        [0.0, 0.0, 80.0, 80.0],
    ]), 4000, 4000)]
    fit = generate_anchors(annotations, anchors_per_grid=1, scales=1, seed=0)
    assert fit.boxes_dropped == 1
    assert fit.boxes_used == 2


# --- borrowing someone else's -----------------------------------------------------------

def test_coverage_says_whether_borrowed_anchors_fit():
    """The question to ask before using COCO's on a blood smear, answered in one number
    instead of a training run."""
    annotations = clustered([(0.09, 0.09)], per_centre=600, spread=0.15, seed=0)
    shapes = box_shapes(annotations)

    borrowed = anchor_coverage(shapes, COCO_ANCHORS)
    fitted, _, _, _ = fit_shapes(shapes, 9, seed=1)
    own = anchor_coverage(shapes, fitted)

    assert own["mean_iou"] > borrowed["mean_iou"] + 0.15
    assert borrowed["anchors"] == 9
    assert borrowed["boxes"] == 600


def test_coverage_counts_the_boxes_their_best_anchor_barely_overlaps():
    """Below a half, a box overlaps its own template less than it misses it."""
    shapes = np.array([[0.9, 0.9]] * 10 + [[0.02, 0.02]] * 10)
    report = anchor_coverage(shapes, [[(0.9, 0.9)]])
    assert report["boxes_below_half"] == 10
    assert report["worst_iou"] < 0.01


# --- refusals ---------------------------------------------------------------------------

def test_asking_for_more_anchors_than_there_are_distinct_shapes_is_refused():
    annotations = [Annotated(np.asarray([[0.0, 0.0, 10.0, 10.0]] * 50), 416, 416)]
    with pytest.raises(DetectionError, match="only 1 distinct"):
        generate_anchors(annotations, anchors_per_grid=3, scales=3, seed=0)


def test_no_boxes_at_all_is_refused():
    with pytest.raises(DetectionError, match="no boxes"):
        fit_shapes(np.zeros((0, 2)), 3)


def test_a_zero_sided_box_is_refused_and_points_at_the_remedy():
    with pytest.raises(DetectionError, match="drop_degenerate"):
        fit_shapes(np.array([[0.1, 0.1], [0.0, 0.1]]), 2)


def test_fewer_than_one_anchor_is_refused():
    with pytest.raises(DetectionError, match="at least 1"):
        fit_shapes(np.array([[0.1, 0.1], [0.2, 0.2]]), 0)


# --- the verification that covers the whole thing ---------------------------------------

def test_the_pipeline_reproduces_cocos_published_anchors_and_grouping():
    """Plant COCO's nine anchors as clusters, refit from nothing, and expect COCO's
    anchors back in COCO's groups.

    This is the one external reference available for the whole pipeline, and it is also
    where the grouping rule is decided. Sorting the nine by **area** reproduces the
    published arrangement; sorting by **width** - what the old implementation did - puts
    (30, 61) in the finest grid and (33, 23) in the middle one, swapping them.
    """
    annotations = clustered(
        [pair for group in COCO_ANCHORS for pair in group], per_centre=80, spread=0.04, seed=0
    )
    fit = generate_anchors(annotations, anchors_per_grid=3, scales=3, seed=2)

    assert fit.converged
    assert fit.mean_iou > 0.9
    for grid in range(3):
        recovered = sorted(tuple(round(v * 416) for v in pair) for pair in fit.anchors[grid])
        published = sorted(tuple(round(v * 416) for v in pair) for pair in COCO_ANCHORS[grid])
        for got, want in zip(recovered, published):
            assert abs(got[0] - want[0]) <= 3 and abs(got[1] - want[1]) <= 3, \
                (grid, recovered, published)


def test_sorting_by_width_would_not_reproduce_cocos_grouping():
    """Stated as a test so the reason for the area ordering cannot quietly be reverted."""
    flat = [pair for group in COCO_ANCHORS for pair in group]
    by_width = sorted(flat, key=lambda p: -p[0])
    by_area = sorted(flat, key=lambda p: -(p[0] * p[1]))

    width_groups = [sorted(by_width[i * 3:(i + 1) * 3]) for i in range(3)]
    area_groups = [sorted(by_area[i * 3:(i + 1) * 3]) for i in range(3)]
    published = [sorted(group) for group in COCO_ANCHORS]

    assert area_groups == published
    assert width_groups != published


def test_the_anchors_it_produces_are_accepted_by_the_encoder():
    """The join nobody tests until it fails: anchors are an argument to `encode`, and a
    shape or a scale it rejects makes the whole pipeline useless."""
    from pyplatypus.detection.encode import encode

    annotations = clustered([(0.05, 0.05), (0.2, 0.2), (0.5, 0.4)], seed=0)
    fit = generate_anchors(annotations, anchors_per_grid=3, scales=3, seed=1)

    encoded = encode([[10, 10, 60, 60]], [0], anchors=fit.anchors, n_class=1)
    assert encoded.placed == 1
    assert encoded.targets[0].shape == (13, 13, 3, 6)
