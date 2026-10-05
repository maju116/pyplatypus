"""One row per image: what was found, what was missed, how well it fits.

The question a table of averages cannot answer. These tests hold two things above all:
that a row says what it claims on a case whose answer is known by construction, and that
the rows and the dataset-wide report **cannot disagree**, because they come from the same
matching and one is a decomposition of the other.
"""

import numpy as np
import pytest

from pyplatypus.detection.metrics import (
    DetectionError,
    detection_report,
    image_report,
)

CLASSES = ["cat", "dog"]


def box(x, y, size=10):
    return [x, y, x + size, y + size]


def predicted(boxes, labels, scores):
    return {"boxes": boxes, "labels": labels, "scores": scores}


def truth(boxes, labels):
    return {"boxes": boxes, "labels": labels}


# --------------------------------------------------------------- what a row means

def test_a_found_box_a_missed_one_and_a_spurious_one():
    """Three outcomes in one image, each reaching its own column."""
    preds = [predicted([box(0, 0), box(500, 500)], [0, 0], [0.9, 0.8])]
    truths = [truth([box(0, 0), box(100, 100)], [0, 0])]

    row, = image_report(preds, truths, labels=CLASSES)
    assert row["n_truth"] == 2
    assert row["n_predicted"] == 2
    assert row["matched"] == 1      # the box at the origin
    assert row["missed"] == 1       # the truth at (100, 100), never predicted
    assert row["spurious"] == 1     # the prediction at (500, 500), nothing there
    assert row["mean_matched_iou"] == pytest.approx(1.0)


def test_nothing_matched_gives_None_rather_than_zero():
    """An image where it found nothing and one where it found badly are different.

    A 0.0 would merge them, and the merged number reads as "placed, poorly".
    """
    preds = [predicted([box(900, 900)], [0], [0.9])]
    truths = [truth([box(0, 0)], [0])]

    row, = image_report(preds, truths, labels=CLASSES)
    assert row["matched"] == 0
    assert row["mean_matched_iou"] is None


def test_an_empty_image_is_a_row_and_not_a_gap():
    preds = [predicted(np.zeros((0, 4)), [], [])]
    truths = [truth(np.zeros((0, 4)), [])]

    row, = image_report(preds, truths, labels=CLASSES)
    assert row["n_truth"] == 0 and row["n_predicted"] == 0
    assert row["matched"] == 0 and row["missed"] == 0 and row["spurious"] == 0
    assert row["mean_matched_iou"] is None


def test_a_prediction_never_claims_a_truth_of_another_class():
    """Perfectly placed, wrong label: a miss and a false positive, not a hit."""
    preds = [predicted([box(0, 0)], [1], [0.9])]      # dog
    truths = [truth([box(0, 0)], [0])]                # cat

    row, = image_report(preds, truths, labels=CLASSES)
    assert row["matched"] == 0
    assert row["missed"] == 1
    assert row["spurious"] == 1


def test_a_wrong_class_prediction_does_not_claim_a_truth_when_both_are_present():
    """The case that actually occurs: two classes in one frame, one box on the wrong one.

    The simpler cross-class test above is passed by code that never compares classes at
    all, because with only one class in the image the per-class loop finds nothing to
    match and leaves early. Here class 0 has a truth of its own, so the loop runs - and a
    second class-0 prediction sitting exactly on the class-1 truth must still be spurious.

    This is BCCD's ordinary frame: red and white cells in one picture, overlapping.
    """
    preds = [predicted([box(0, 0), box(100, 100)], [0, 0], [0.9, 0.8])]
    truths = [truth([box(0, 0), box(100, 100)], [0, 1])]

    row, = image_report(preds, truths, labels=CLASSES)
    assert row["matched"] == 1      # only the class-0 pair
    assert row["missed"] == 1       # the class-1 truth, never predicted as class 1
    assert row["spurious"] == 1     # the class-0 box lying on it


def test_a_duplicate_box_is_spurious_rather_than_a_second_hit():
    preds = [predicted([box(0, 0), box(0, 0)], [0, 0], [0.9, 0.8])]
    truths = [truth([box(0, 0)], [0])]

    row, = image_report(preds, truths, labels=CLASSES)
    assert row["matched"] == 1 and row["spurious"] == 1


def test_the_threshold_decides_how_many_count_as_predicted():
    """Counts are read at an operating point, and the point moves them."""
    preds = [predicted([box(0, 0), box(500, 500)], [0, 0], [0.9, 0.2])]
    truths = [truth([box(0, 0)], [0])]

    loose, = image_report(preds, truths, labels=CLASSES, score_threshold=0.1)
    tight, = image_report(preds, truths, labels=CLASSES, score_threshold=0.5)
    assert loose["n_predicted"] == 2 and loose["spurious"] == 1
    assert tight["n_predicted"] == 1 and tight["spurious"] == 0


# --------------------------------------------------------- it agrees with the table

def a_dataset_with_real_matches(seed=11, n_images=25):
    """Predictions derived from the truths, so most of them actually match.

    Built this way on purpose. The first version of this fixture drew both sets at random
    in a 200x200 field, and 10-pixel boxes never reach IoU 0.5 by accident: it produced
    **55 truths, 32 predictions and 0 matches**, so every assertion below compared zero
    with zero and passed whatever the code did. Measured, not guessed - and the same shape
    as the two voxel counts that were both zero earlier in this project.
    """
    rng = np.random.default_rng(seed)
    preds, truths = [], []
    for _ in range(n_images):
        n_t = int(rng.integers(1, 5))
        origins = rng.integers(0, 300, (n_t, 2))
        labels = list(rng.integers(0, 2, n_t))
        truths.append(truth([box(x, y, 20) for x, y in origins], labels))

        kept, p_boxes, p_labels = [], [], []
        for (x, y), label in zip(origins, labels):
            if rng.random() < 0.75:                      # found, slightly off
                p_boxes.append(box(x + rng.integers(0, 4), y + rng.integers(0, 4), 20))
                p_labels.append(label)
                kept.append(True)
        for _ in range(int(rng.integers(0, 3))):         # and some inventions
            p_boxes.append(box(*rng.integers(400, 600, 2), 20))
            p_labels.append(int(rng.integers(0, 2)))
        preds.append(predicted(p_boxes, p_labels,
                               list(rng.uniform(0.55, 1.0, len(p_boxes)))))
    return preds, truths


def test_the_rows_sum_to_the_dataset_report():
    """The assertion that matters: a decomposition, not a second implementation.

    If these two ever disagree, one of them is lying about what matched - and a user
    reading a per-image table to explain a dataset-wide number would be chasing a
    difference that is in the code rather than in the data.
    """
    preds, truths = a_dataset_with_real_matches()
    point = 0.5
    rows = image_report(preds, truths, labels=CLASSES, score_threshold=point)
    table = detection_report(preds, truths, labels=CLASSES, iou_thresholds=(0.5,),
                             score_threshold=point)

    # Before comparing anything: there must be something to compare. Without this the
    # whole test passes on a dataset where nothing matched at all.
    assert sum(r["matched"] for r in rows) > 20
    assert sum(r["spurious"] for r in rows) > 0
    assert sum(r["missed"] for r in rows) > 0

    assert sum(r["n_truth"] for r in rows) == sum(c["n_truth"] for c in table.per_class)
    assert (sum(r["n_predicted"] for r in rows)
            == sum(c["n_predicted"] for c in table.per_class))

    # True positives are not a column of the per-class table, so they are reconstructed
    # from precision, which is the same quantity by another name.
    expected = sum(round(c["precision"] * c["n_predicted"])
                   for c in table.per_class if c["precision"] is not None)
    assert sum(r["matched"] for r in rows) == expected


def test_the_mean_overlap_of_the_rows_is_the_tables():
    """Weighted by how many each image matched, which is what the table averages over."""
    preds, truths = a_dataset_with_real_matches(seed=3, n_images=20)
    rows = image_report(preds, truths, labels=CLASSES)
    table = detection_report(preds, truths, labels=CLASSES, iou_thresholds=(0.5,))

    values = [(r["mean_matched_iou"], r["matched"]) for r in rows if r["matched"]]
    assert len(values) > 10
    assert 0.5 < table.mean_matched_iou < 1.0      # varied, not all perfect
    pooled = sum(v * w for v, w in values) / sum(w for _, w in values)
    assert pooled == pytest.approx(table.mean_matched_iou)


# ------------------------------------------------------------------------ refusals

def test_keys_are_carried_through_so_a_row_can_be_found_again():
    preds = [predicted([box(0, 0)], [0], [0.9]), predicted([], [], [])]
    truths = [truth([box(0, 0)], [0]), truth([box(5, 5)], [0])]

    rows = image_report(preds, truths, labels=CLASSES, keys=["first", "second"])
    assert [r["key"] for r in rows] == ["first", "second"]


def test_a_key_per_image_or_none_at_all():
    preds = [predicted([], [], [])]
    truths = [truth([], [])]
    with pytest.raises(DetectionError, match="one each"):
        image_report(preds, truths, labels=CLASSES, keys=["a", "b"])


def test_predictions_and_truths_must_line_up():
    with pytest.raises(DetectionError, match="line up"):
        image_report([predicted([], [], [])], [], labels=CLASSES)


def test_an_impossible_threshold_is_refused():
    preds = [predicted([], [], [])]
    truths = [truth([], [])]
    with pytest.raises(DetectionError, match="above 0 and at most 1"):
        image_report(preds, truths, labels=CLASSES, iou_threshold=0.0)
