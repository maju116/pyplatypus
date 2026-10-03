"""Non-maximum suppression: one object, one box.

A detector predicts from every cell of every grid, so an object arrives as a cluster. The
metric's matching is greedy and claims each truth once, so without suppression each cluster
is one true positive and a handful of false ones - and precision collapses for a reason
that has nothing to do with whether the object was found.
"""

import numpy as np
import pytest

from pyplatypus.detection import non_max_suppression as nms
from pyplatypus.detection.metrics import DetectionError


def test_a_cluster_on_one_object_becomes_one_box():
    boxes = [[10, 10, 50, 50], [12, 12, 52, 52], [11, 9, 49, 51], [200, 200, 240, 240]]
    keep = nms(boxes, [0.9, 0.8, 0.7, 0.6])
    assert keep.tolist() == [0, 3]


def test_the_most_confident_box_survives():
    boxes = [[10, 10, 50, 50], [11, 11, 51, 51]]
    assert nms(boxes, [0.3, 0.9]).tolist() == [1]


def test_boxes_that_do_not_overlap_are_both_kept():
    boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]
    assert sorted(nms(boxes, [0.9, 0.8]).tolist()) == [0, 1]


def test_suppression_is_per_class_because_two_things_can_share_a_place():
    """A platelet sitting on a red cell is two objects at nearly one position, and
    suppressing across classes deletes one of them."""
    boxes = [[10, 10, 50, 50], [12, 12, 52, 52]]
    assert nms(boxes, [0.9, 0.8], [0, 0]).tolist() == [0]
    assert sorted(nms(boxes, [0.9, 0.8], [0, 1]).tolist()) == [0, 1]
    assert nms(boxes, [0.9, 0.8], [0, 1], per_class=False).tolist() == [0]


def test_the_result_is_ordered_by_confidence_across_classes():
    """So a caller taking the first N takes the most confident N, not the most confident
    of class 0."""
    boxes = [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]]
    assert nms(boxes, [0.3, 0.9, 0.6], [1, 0, 1]).tolist() == [1, 2, 0]


def test_the_threshold_decides_how_much_overlap_is_too_much():
    boxes = [[0, 0, 10, 10], [5, 0, 15, 10]]        # IoU 1/3
    assert len(nms(boxes, [0.9, 0.8], iou_threshold=0.3)) == 1
    assert len(nms(boxes, [0.9, 0.8], iou_threshold=0.5)) == 2


def test_a_score_threshold_drops_the_tail_first():
    boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]
    assert nms(boxes, [0.9, 0.2], score_threshold=0.5).tolist() == [0]


def test_a_limit_takes_the_most_confident():
    boxes = [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]]
    assert nms(boxes, [0.1, 0.9, 0.5], limit=2).tolist() == [1, 2]


def test_nothing_in_gives_nothing_out():
    assert nms([], []).shape == (0,)


def test_a_zero_area_box_does_not_divide_by_zero():
    boxes = [[5, 5, 5, 5], [0, 0, 10, 10]]
    assert len(nms(boxes, [0.9, 0.8])) == 2


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_an_impossible_threshold_is_refused(threshold):
    with pytest.raises(DetectionError, match="iou_threshold"):
        nms([[0, 0, 1, 1]], [0.9], iou_threshold=threshold)


def test_mismatched_lengths_are_refused():
    with pytest.raises(DetectionError, match="one score per box"):
        nms([[0, 0, 1, 1], [2, 2, 3, 3]], [0.9])
    with pytest.raises(DetectionError, match="they must agree"):
        nms([[0, 0, 1, 1]], [0.9], [0, 1])


def test_suppression_is_what_makes_precision_meaningful():
    """The join with the metric, shown rather than asserted: the same cluster scored with
    and without suppression."""
    from pyplatypus.detection import detection_report

    truth = [{"boxes": [[10, 10, 50, 50]], "labels": [0]}]
    cluster = np.array([[10, 10, 50, 50], [11, 11, 51, 51], [9, 9, 49, 49],
                        [12, 10, 52, 50]], dtype=float)
    scores = np.array([0.9, 0.85, 0.8, 0.75])

    raw = detection_report(
        [{"boxes": cluster, "scores": scores, "labels": [0] * 4}], truth, labels=["a"]
    )
    keep = nms(cluster, scores, [0] * 4)
    suppressed = detection_report(
        [{"boxes": cluster[keep], "scores": scores[keep],
          "labels": [0] * len(keep)}], truth, labels=["a"]
    )

    assert raw.per_class[0]["false_positives"] == 3
    assert suppressed.per_class[0]["false_positives"] == 0
    assert suppressed.per_class[0]["precision"] == 1.0
