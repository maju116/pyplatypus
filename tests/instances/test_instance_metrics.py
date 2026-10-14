"""The instance metric, against pycocotools and against hand-worked answers.

The hand-worked cases say the code does what it means. The pycocotools cases say the IoU
underneath it is the same quantity everybody else computes - and the numbers below were
produced once, against that reference, and written in here so the suite does not need it
installed. That is the arrangement the detection metrics already have with the same library.

The problems are built so the comparison is worth making: predictions are the truth objects
shifted by a pixel or two, one truth object with nothing predicting it and one prediction with
no truth behind it. Random blobs on a grid barely overlap, and a matrix of zeros agrees with
anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplatypus.instances import (
    CHALLENGE_THRESHOLDS,
    InstanceError,
    average_precision,
    foreground_dice,
    instances_from_mask,
    iou_matrix,
    label_map,
    score_cases,
    summarise_instances,
)

SIZE = 32

#: Computed once with pycocotools 2.x on the problems `problem()` builds, as plain IoU with
#: `iscrowd = 0` so the union is in the denominator. If one of these changes, either the
#: contingency table or the problem generator moved - both are worth knowing about.
PYCOCOTOOLS = {
    1: [
        [0.361445783133, 0.0, 0.0, 0.0],
        [0.018018018018, 0.591549295775, 0.0, 0.0],
        [0.0, 0.0, 0.777777777778, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    2: [
        [0.352941176471, 0.0, 0.0, 0.0],
        [0.0, 0.363636363636, 0.0, 0.0],
        [0.0, 0.0, 0.729166666667, 0.0],
        [0.0, 0.0, 0.0, 0.125],
    ],
    3: [
        [0.64, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
}


def problem(seed: int, n_truth: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """The same generator the reference numbers were computed from, reproduced exactly."""
    rng = np.random.default_rng(seed)
    boxes = []
    truth = np.zeros((SIZE, SIZE), dtype=np.int32)
    for index in range(1, n_truth + 1):
        while True:
            top, left = rng.integers(1, SIZE - 10, size=2)
            height, width = rng.integers(5, 9, size=2)
            if not truth[top : top + height, left : left + width].any():
                break
        truth[top : top + height, left : left + width] = index
        boxes.append((top, left, height, width))

    predicted = np.zeros((SIZE, SIZE), dtype=np.int32)
    label = 0
    for index, (top, left, height, width) in enumerate(boxes):
        if index == n_truth - 1:
            continue
        shift_y, shift_x = rng.integers(-2, 3, size=2)
        grow = rng.integers(-1, 2)
        label += 1
        y, x = max(0, top + shift_y), max(0, left + shift_x)
        predicted[y : y + height + grow, x : x + width + grow] = label
    label += 1
    predicted[SIZE - 7 : SIZE - 2, SIZE - 7 : SIZE - 2] = label
    return truth, predicted


# --- agreement with pycocotools --------------------------------------------------------


@pytest.mark.parametrize("seed", sorted(PYCOCOTOOLS))
def test_iou_agrees_with_pycocotools(seed):
    truth, predicted = problem(seed)
    computed = iou_matrix(truth, predicted)
    np.testing.assert_allclose(computed, np.array(PYCOCOTOOLS[seed]), atol=1e-12, rtol=0)


def test_the_reference_problems_actually_overlap():
    """A guard on the fixture rather than on the code.

    Every number above could be zero and every assertion would still pass, which is the shape
    of check that admits the failure it was written to catch. So: each problem has pairs that
    overlap, and at least one that straddles the lowest threshold.
    """
    for seed, matrix in PYCOCOTOOLS.items():
        values = np.array(matrix)
        assert (values > 0).sum() >= 3, f"seed {seed} has almost no overlap to compare"
        assert values.max() > min(CHALLENGE_THRESHOLDS), (
            f"seed {seed} has no pair above the lowest threshold, so the metric built on this "
            "IoU would score zero whatever the matching did"
        )


# --- the metric, by hand ---------------------------------------------------------------


def test_a_perfect_prediction_scores_one():
    truth = label_map(
        [
            np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]) > 0,
            np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]]) > 0,
        ]
    )
    predicted = instances_from_mask(np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]]))
    assert average_precision(iou_matrix(truth, predicted)) == 1.0


def test_two_touching_objects_merged_into_one_score_zero():
    """The limitation `dsbowl-unet`'s card names, as a number.

    Two nuclei side by side, predicted as one region: each truth object has IoU exactly 0.5
    with it. Nothing clears a threshold, so there are two false negatives and one false
    positive and the image scores zero - which is the point of measuring this baseline rather
    than describing it.
    """
    truth = label_map([np.array([[1, 1, 0, 0]]) > 0, np.array([[0, 0, 1, 1]]) > 0])
    predicted = instances_from_mask(np.array([[1, 1, 1, 1]]))

    iou = iou_matrix(truth, predicted)
    np.testing.assert_allclose(iou, [[0.5], [0.5]])
    assert average_precision(iou) == 0.0


def test_the_half_threshold_is_strict_and_that_is_what_keeps_the_match_unique():
    """With `>=` instead of `>`, the case above would award two true positives for one region.

    That is the whole reason the comparison is strict, and it is invisible in any test whose
    IoUs avoid 0.5 exactly. Asserted on the counting directly: a matrix of two 0.5s must yield
    no match at 0.5, while 0.5 + epsilon yields one for each row - and one prediction cannot
    honestly satisfy two objects.
    """
    exactly_half = np.array([[0.5], [0.5]])
    assert average_precision(exactly_half, thresholds=[0.5]) == 0.0

    # Above half, only one of the two can clear it, because two disjoint regions cannot each
    # cover more than half of one object. Here that is forced by construction.
    one_clears = np.array([[0.6], [0.4]])
    assert average_precision(one_clears, thresholds=[0.5]) == pytest.approx(1 / 2)


def test_a_threshold_below_half_is_refused():
    with pytest.raises(InstanceError, match="below 0.5"):
        average_precision(np.array([[0.4]]), thresholds=[0.45])


def test_the_challenge_thresholds_are_the_ten_the_challenge_swept():
    assert CHALLENGE_THRESHOLDS[0] == 0.5
    assert CHALLENGE_THRESHOLDS[-1] == 0.95
    assert len(CHALLENGE_THRESHOLDS) == 10
    assert average_precision(np.array([[1.0]])) == 1.0


def test_an_image_with_nothing_on_either_side_is_not_scored():
    """`None`, not zero: zero says the prediction was wrong where there was nothing to be
    wrong about, and it would drag a mean down for images that contain nothing."""
    empty = np.zeros((4, 4), dtype=np.int32)
    assert average_precision(iou_matrix(empty, empty)) is None


def test_missing_everything_and_inventing_everything_both_score_zero():
    objects = label_map([np.array([[1, 1, 0, 0]]) > 0])
    nothing = np.zeros((1, 4), dtype=np.int32)

    assert average_precision(iou_matrix(objects, nothing)) == 0.0
    assert average_precision(iou_matrix(nothing, objects)) == 0.0


# --- the representation, and its refusals ----------------------------------------------


def test_overlapping_truth_is_refused_rather_than_resolved():
    """Picking the topmost label would answer the open question wrongly instead of declining
    it, and the open question is the next piece of work in this milestone."""
    with pytest.raises(InstanceError, match="overlaps an earlier one on 1 pixels"):
        label_map([np.array([[1, 1, 0]]) > 0, np.array([[0, 1, 1]]) > 0])


def test_masks_of_different_shapes_are_refused():
    with pytest.raises(InstanceError, match="has shape"):
        label_map([np.zeros((2, 2), bool), np.zeros((2, 3), bool)])


def test_an_empty_list_of_masks_is_refused():
    with pytest.raises(InstanceError, match="no instance masks"):
        label_map([])


def test_comparing_different_grids_is_refused():
    with pytest.raises(InstanceError, match="have to cover the same grid"):
        iou_matrix(np.zeros((2, 2), np.int32), np.zeros((2, 3), np.int32))


# --- the naive post-process ------------------------------------------------------------


def test_four_connectivity_keeps_corner_touching_objects_apart():
    """Under 8-connectivity these are one object. A diagonal join is the cheapest way to
    merge things that are not joined, and this baseline exists to be beaten rather than to
    flatter itself."""
    corner = np.array([[1, 0], [0, 1]])
    assert instances_from_mask(corner).max() == 2


def test_a_minimum_area_drops_small_regions_and_relabels_consecutively():
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[0:3, 0:3] = 1  # 9 pixels
    mask[5, 5] = 1  # 1 pixel
    labels = instances_from_mask(mask, minimum_area=4)

    assert labels.max() == 1, "the single pixel should be gone"
    assert sorted(np.unique(labels)) == [0, 1], "labels have to stay consecutive from 1"


def test_a_volume_is_refused_rather_than_flattened():
    with pytest.raises(InstanceError, match="expected a 2-D mask"):
        instances_from_mask(np.zeros((2, 2, 2)))


# --- reporting -------------------------------------------------------------------------


def test_the_mean_is_over_images_and_the_pooled_number_is_reported_beside_it():
    """Two images, one with a single object found and one with three objects all missed.

    The mean over images is 0.5; pooling the counts gives 1/(1+0+3) = 0.25. They are different
    quantities, the challenge ranked on the first, and reporting one without the other is how a
    number gets quoted as the other.
    """
    one = label_map([np.array([[1, 1, 0, 0]]) > 0])
    three = label_map(
        [
            np.array([[1, 1, 0, 0, 0, 0]]) > 0,
            np.array([[0, 0, 1, 1, 0, 0]]) > 0,
            np.array([[0, 0, 0, 0, 1, 1]]) > 0,
        ]
    )
    merged = instances_from_mask(np.array([[1, 1, 1, 1, 1, 1]]))

    rows = score_cases([("found", one, one), ("merged", three, merged)])
    assert [row["average_precision"] for row in rows] == [1.0, 0.0]

    summary = summarise_instances(rows)
    assert summary["mean_average_precision"] == 0.5
    assert summary["n"] == 2
    assert summary["unscored"] == 0
    assert summary["worst"] == "merged"
    assert (summary["truth_objects"], summary["predicted_objects"]) == (4, 2)


def test_unscored_images_are_counted_and_left_out_of_the_mean():
    empty = np.zeros((2, 2), dtype=np.int32)
    one = label_map([np.array([[1, 1], [0, 0]]) > 0])

    rows = score_cases([("nothing", empty, empty), ("found", one, one)])
    summary = summarise_instances(rows)

    assert summary["n"] == 1
    assert summary["unscored"] == 1
    assert summary["mean_average_precision"] == 1.0


def test_nothing_to_score_at_all_is_refused():
    empty = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(InstanceError, match="no case had anything to score"):
        summarise_instances(score_cases([("nothing", empty, empty)]))


def test_the_foreground_dice_does_not_care_how_the_objects_are_numbered():
    """It is a mask score, so relabelling the truth cannot move it.

    Worth pinning because the input is a *label map* and the obvious wrong implementation -
    comparing the arrays rather than their foregrounds - agrees with this one whenever the
    objects happen to be numbered 1 upwards, which is how every fixture here numbers them.
    """
    predicted = np.array([[True, True], [False, False]])
    assert foreground_dice(predicted, np.array([[1, 1], [0, 0]])) == 1.0
    assert foreground_dice(predicted, np.array([[7, 9], [0, 0]])) == 1.0


def test_the_foreground_dice_is_not_smoothed_and_says_so_by_returning_none():
    """The `+1` the training metrics carry would score an empty prediction on empty truth
    1.0, which is the number an unscorable case must not report. `None` is this module's
    answer for it - the same one `score_cases` gives an `average_precision` - and the
    unsmoothed arithmetic is what makes a near-miss read as a near-miss.
    """
    empty = np.zeros((2, 2), dtype=np.int32)
    assert foreground_dice(np.zeros((2, 2), bool), empty) is None
    # One predicted pixel against 101 true ones: the overlap is 2/102 = 0.0196, where the
    # smoothed form reads 3/103 = 0.0291 - half again as much for the same near-total miss.
    predicted = np.zeros(101, dtype=bool)
    predicted[0] = True
    truth = np.zeros(101, dtype=np.int32)
    truth[0], truth[1:] = 1, 2
    assert foreground_dice(predicted, truth) == pytest.approx(2 / 102)
