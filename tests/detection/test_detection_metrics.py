"""Detection metrics, checked against hand-worked answers and against pycocotools.

The hand-worked cases are the ones that say the code does what it means. The pycocotools
numbers are the ones that say "mAP" here means what it means everywhere else, and they are
baked in rather than computed, so the agreement is checked on every run by a suite that
does not need pycocotools installed. It was computed once, against the reference, and
`scripts/` is not where that should live - a verification that happened once is not a test.
"""

import numpy as np
import pytest

from pyplatypus.detection import (
    COCO_THRESHOLDS,
    average_precision,
    detection_report,
    iou_matrix,
    match_detections,
)
from pyplatypus.detection.metrics import DetectionError

# --- intersection over union -----------------------------------------------------------


def test_a_box_against_itself_is_one():
    assert iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == pytest.approx(1.0)


def test_disjoint_boxes_are_zero():
    assert iou_matrix([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0.0


def test_half_overlap_is_a_third():
    """Two 100-unit boxes sharing 50: intersection 50, union 150."""
    assert iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)


def test_a_contained_box_is_the_ratio_of_areas():
    assert iou_matrix([[0, 0, 10, 10]], [[0, 0, 5, 10]])[0, 0] == pytest.approx(0.5)


def test_a_zero_area_box_gives_zero_rather_than_dividing_by_zero():
    assert iou_matrix([[5, 5, 5, 5]], [[0, 0, 10, 10]])[0, 0] == 0.0


def test_empty_input_gives_an_empty_matrix():
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)


def test_a_reversed_box_is_refused_with_the_likely_cause():
    with pytest.raises(DetectionError, match="column order"):
        iou_matrix([[10, 10, 0, 0]], [[0, 0, 1, 1]])


# --- matching --------------------------------------------------------------------------


def truth(*boxes):
    return list(boxes)


def test_a_prediction_over_a_truth_is_a_hit():
    _, hit, _, _ignored = match_detections([[0, 0, 10, 10]], [0.9], truth([0, 0, 10, 10]))
    assert hit.tolist() == [True]


def test_a_prediction_below_the_threshold_is_a_miss():
    # IoU 1/3, threshold 0.5.
    _, hit, _, _ignored = match_detections([[5, 0, 15, 10]], [0.9], truth([0, 0, 10, 10]))
    assert hit.tolist() == [False]


def test_a_second_prediction_on_the_same_object_is_a_false_positive():
    """Which is what makes non-maximum suppression worth doing."""
    _, hit, _, _ignored = match_detections(
        [[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], truth([0, 0, 10, 10])
    )
    assert hit.tolist() == [True, False]


def test_the_higher_scoring_prediction_claims_the_truth():
    order, hit, _, _ignored = match_detections(
        [[5, 5, 15, 15], [0, 0, 10, 10]], [0.3, 0.9], truth([0, 0, 10, 10])
    )
    # Sorted by score, so the good box comes first and takes it.
    assert order.tolist() == [1, 0]
    assert hit.tolist() == [True, False]


def test_among_candidates_the_best_overlap_wins():
    """Two truths in reach; the prediction should take the one it fits, not the first."""
    _, hit, _, _ignored = match_detections(
        [[9, 0, 20, 10]], [0.9], truth([0, 0, 10, 10], [10, 0, 20, 10])
    )
    assert hit.tolist() == [True]
    # And the one it took is the better fit: with only the poor truth present, it misses.
    _, poor, _, _ignored = match_detections([[9, 0, 20, 10]], [0.9], truth([0, 0, 10, 10]))
    assert poor.tolist() == [False]


def test_ties_in_score_keep_the_input_order():
    """So a report is reproducible rather than dependent on the sort's internals."""
    order, _, _, _ignored = match_detections(
        [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], [0.5, 0.5, 0.5], truth()
    )
    assert order.tolist() == [0, 1, 2]


def test_one_score_per_box_is_required():
    with pytest.raises(DetectionError, match="one score per box"):
        match_detections([[0, 0, 1, 1], [2, 2, 3, 3]], [0.9], truth())


# --- average precision, by hand --------------------------------------------------------


@pytest.mark.parametrize(
    "hit, n_truth, expected",
    [
        ([True], 1, 1.0),  # found it, nothing else
        ([False], 1, 0.0),  # missed it
        ([True, False], 1, 1.0),  # found it, then a duplicate: recall never grows
        ([False, True], 1, 0.5),  # a false positive outranks the hit
        ([True, True], 2, 1.0),  # both found, top two
        ([True, False], 2, 0.5),  # one of two, recall caps at a half
    ],
)
def test_average_precision_matches_the_arithmetic(hit, n_truth, expected):
    assert average_precision(hit, n_truth) == pytest.approx(expected)


def test_the_three_conventions_disagree_and_that_is_why_they_are_named():
    """One truth of two found, then a false positive. Worked by hand:

    all : recall steps to 0.5 at precision 1.0, so the area is 0.5
    11  : six of the eleven levels (0 to 0.5) sit at precision 1.0, so 6/11
    """
    hit, n_truth = [True, False], 2
    assert average_precision(hit, n_truth, interpolation="all") == pytest.approx(0.5)
    assert average_precision(hit, n_truth, interpolation="11") == pytest.approx(6 / 11)
    assert average_precision(hit, n_truth, interpolation="101") == pytest.approx(51 / 101)


def test_a_class_with_no_truth_has_no_average_precision():
    """None rather than zero: a class the data never contained should not drag a mean
    down, and it did not score badly - it was not scored."""
    assert average_precision([False, False], 0) is None


def test_an_unknown_interpolation_is_refused():
    with pytest.raises(DetectionError, match="'all', '101' or '11'"):
        average_precision([True], 1, interpolation="voc")


# --- the whole report ------------------------------------------------------------------


def one_image(boxes, labels, scores=None):
    entry = {"boxes": boxes, "labels": labels}
    if scores is not None:
        entry["scores"] = scores
    return entry


def test_a_perfect_detector_scores_one():
    truths = [one_image([[0, 0, 10, 10]], [0]), one_image([[5, 5, 20, 20]], [1])]
    preds = [one_image([[0, 0, 10, 10]], [0], [0.9]), one_image([[5, 5, 20, 20]], [1], [0.8])]
    report = detection_report(preds, truths, labels=["a", "b"])
    assert report.mean_average_precision == pytest.approx(1.0)
    assert report.classes_without_truth == []


def test_a_detector_that_finds_nothing_scores_zero():
    truths = [one_image([[0, 0, 10, 10]], [0])]
    preds = [one_image([], [], [])]
    report = detection_report(preds, truths, labels=["a"])
    assert report.mean_average_precision == pytest.approx(0.0)
    assert report.per_class[0]["false_negatives"] == 1


def test_predictions_are_pooled_across_images_rather_than_averaged_per_image():
    """The documented decision, demonstrated rather than asserted.

    Two images: one with a single object found confidently, one with nine objects of which
    one is found at low confidence. Averaging the two images' APs gives about 0.56. The
    pooled curve - one ranking over the dataset - gives 0.2, because nine of the ten
    objects were missed. The pooled number is the one that describes the detector.
    """
    truths = [
        one_image([[0, 0, 10, 10]], [0]),
        one_image([[20 * i, 0, 20 * i + 10, 10] for i in range(9)], [0] * 9),
    ]
    preds = [
        one_image([[0, 0, 10, 10]], [0], [0.99]),
        one_image([[0, 0, 10, 10]], [0], [0.10]),
    ]
    report = detection_report(preds, truths, labels=["a"])
    pooled = report.mean_average_precision

    per_image = np.mean(
        [
            detection_report([preds[i]], [truths[i]], labels=["a"]).mean_average_precision
            for i in (0, 1)
        ]
    )
    assert pooled == pytest.approx(0.2)
    assert per_image == pytest.approx(0.5555, abs=1e-3)
    assert pooled < per_image


def test_a_declared_class_the_data_never_contained_is_reported_not_scored():
    truths = [one_image([[0, 0, 10, 10]], [0])]
    preds = [one_image([[0, 0, 10, 10]], [0], [0.9])]
    report = detection_report(preds, truths, labels=["a", "never_seen"])

    assert report.classes_without_truth == [1]
    assert report.per_class[1]["average_precision"] is None
    # The mean is over the class that had truth, so it is not halved by the absent one.
    assert report.mean_average_precision == pytest.approx(1.0)


HARD = [[0, 0, 10, 10], [50, 50, 60, 60]]


def one_hard_one_not(boxes=None):
    """Two truths of one class, the second marked `difficult`."""
    return {"boxes": HARD if boxes is None else boxes, "labels": [0, 0], "difficult": [False, True]}


def test_difficult_objects_are_left_out_as_pascal_voc_leaves_them_out():
    preds = [one_image([[0, 0, 10, 10]], [0], [0.9])]
    report = detection_report(preds, [one_hard_one_not()], labels=["a"])
    assert report.per_class[0]["n_truth"] == 1
    assert report.mean_average_precision == pytest.approx(1.0)


def test_finding_a_difficult_object_is_not_punished_for_finding_it():
    """The case the test above cannot contain, which is why it missed `pyplatypus#364`.

    It predicts only the ordinary object, so `n_truth == 1` and an average precision of 1
    are true whether a difficult truth is set aside or deleted. The difference appears the
    moment a detection lands on the difficult one - and before `#364` that detection had
    nothing to match and was scored as a false positive, so a model that found the hard
    object did WORSE than one that missed it.
    """
    # The difficult object detected, and ranked first, which is the costliest place on the
    # curve for a false positive to sit.
    preds = [one_image([[50, 50, 60, 60], [0, 0, 10, 10]], [0, 0], [0.9, 0.8])]
    report = detection_report(preds, [one_hard_one_not()], labels=["a"])
    row = report.per_class[0]
    assert report.mean_average_precision == pytest.approx(1.0)
    assert (row["true_positives"], row["false_positives"], row["ignored"]) == (1, 0, 1)
    # `n_predicted` still counts it, so the arithmetic closes through `ignored` rather
    # than by hiding the prediction.
    assert row["n_predicted"] == row["true_positives"] + row["false_positives"] + row["ignored"]
    assert row["precision"] == pytest.approx(1.0)

    # The control that says this is about the flag and not about two boxes: with nothing
    # marked difficult, the same prediction is an ordinary pair of hits.
    plain = {"boxes": HARD, "labels": [0, 0], "difficult": [False, False]}
    ordinary = detection_report(preds, [plain], labels=["a"]).per_class[0]
    assert (ordinary["true_positives"], ordinary["ignored"], ordinary["n_truth"]) == (2, 0, 2)


def test_a_duplicate_of_an_ORDINARY_truth_is_still_a_false_positive():
    """The other control: setting difficult truths aside must not soften the usual rule."""
    preds = [one_image([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0], [0.9, 0.8])]
    row = detection_report(preds, [one_hard_one_not()], labels=["a"]).per_class[0]
    assert (row["true_positives"], row["false_positives"], row["ignored"]) == (1, 1, 0)


def test_the_best_overlap_decides_and_the_difficult_flag_is_asked_second():
    """`voc_eval`'s order: a detection fitting a difficult truth better is set aside.

    The prediction overlaps the ordinary truth at 0.6 and the difficult one at 0.9, so a
    reading that looked for any admissible ordinary truth first would score it.
    """
    truth = {
        "boxes": [[0, 0, 100, 60], [0, 0, 100, 90]],
        "labels": [0, 0],
        "difficult": [False, True],
    }
    preds = [one_image([[0, 0, 100, 100]], [0], [0.9])]
    row = detection_report(preds, [truth], labels=["a"]).per_class[0]
    assert (row["true_positives"], row["false_positives"], row["ignored"]) == (0, 0, 1)

    _, hit, _, ignored = match_detections(
        [[0, 0, 100, 100]], [0.9], [[0, 0, 100, 60]], difficult_boxes=[[0, 0, 100, 90]]
    )
    assert not hit[0] and ignored[0]


def test_a_tie_goes_to_the_truth_that_scores():
    """Equal overlap, so a difficult object must not absorb a detection that fits both."""
    box = [[0, 0, 10, 10]]
    _, hit, _, ignored = match_detections(box, [0.9], box, difficult_boxes=box)
    assert hit[0] and not ignored[0]


def test_a_difficult_truth_absorbs_every_detection_of_it_rather_than_one():
    """VOC never marks a difficult truth as detected: the region is not being judged.

    So two guesses at one difficult object are both set aside, where a truth that scores
    would have taken the first and made a false positive of the second.
    """
    both = [[0, 0, 10, 10], [1, 1, 11, 11]]
    _, hit, _, ignored = match_detections(both, [0.9, 0.8], [], difficult_boxes=[[0, 0, 10, 10]])
    assert not hit.any() and ignored.all()


def test_the_conventions_used_travel_with_the_result():
    truths = [one_image([[0, 0, 10, 10]], [0])]
    preds = [one_image([[0, 0, 10, 10]], [0], [0.9])]
    report = detection_report(
        preds, truths, labels=["a"], iou_thresholds=(0.5, 0.75), interpolation="101"
    )
    assert report.iou_thresholds == (0.5, 0.75)
    assert report.interpolation == "101"


def test_rows_carry_an_overall_line_for_r():
    truths = [one_image([[0, 0, 10, 10]], [0])]
    preds = [one_image([[0, 0, 10, 10]], [0], [0.9])]
    rows = detection_report(preds, truths, labels=["a"]).as_rows()
    assert rows[-1]["label"] == "all"
    assert rows[-1]["average_precision"] == pytest.approx(1.0)


def test_mismatched_image_counts_are_refused():
    with pytest.raises(DetectionError, match="they must line up"):
        detection_report([one_image([], [], [])], [], labels=["a"])


def test_predictions_without_scores_are_refused_with_the_reason():
    with pytest.raises(DetectionError, match="no ordering"):
        detection_report(
            [one_image([[0, 0, 1, 1]], [0])], [one_image([[0, 0, 1, 1]], [0])], labels=["a"]
        )


@pytest.mark.parametrize("thresholds", [(), (0.0,), (1.5,), (-0.5,)])
def test_impossible_iou_thresholds_are_refused(thresholds):
    truths = [one_image([[0, 0, 10, 10]], [0])]
    preds = [one_image([[0, 0, 10, 10]], [0], [0.9])]
    with pytest.raises(DetectionError):
        detection_report(preds, truths, labels=["a"], iou_thresholds=thresholds)


# --- agreement with pycocotools --------------------------------------------------------


def random_problem(seed: int, n_images: int = 8, n_class: int = 3) -> dict:
    """A reproducible detection problem: some predictions are jittered truths, some are
    inventions, scores are arbitrary. Mixed enough that the precision-recall curve has
    steps in it rather than being trivial."""
    rng = np.random.default_rng(seed)
    truths, preds = [], []
    for _ in range(n_images):
        k = int(rng.integers(0, 6))
        boxes, labels = [], []
        for _ in range(k):
            x, y = rng.uniform(0, 80, 2)
            w, h = rng.uniform(10, 40, 2)
            boxes.append([x, y, x + w, y + h])
            labels.append(int(rng.integers(0, n_class)))
        truths.append({"boxes": boxes, "labels": labels})

        m = int(rng.integers(0, 8))
        pboxes, plabels, scores = [], [], []
        for _ in range(m):
            if boxes and rng.random() < 0.6:
                base = boxes[int(rng.integers(0, len(boxes)))]
                jitter = rng.normal(0, 5, 4)
                pboxes.append(
                    [
                        base[0] + jitter[0],
                        base[1] + jitter[1],
                        base[2] + jitter[2],
                        base[3] + jitter[3],
                    ]
                )
            else:
                x, y = rng.uniform(0, 80, 2)
                w, h = rng.uniform(10, 40, 2)
                pboxes.append([x, y, x + w, y + h])
            plabels.append(int(rng.integers(0, n_class)))
            scores.append(float(rng.uniform(0.05, 0.99)))
        pboxes = [
            [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])] for b in pboxes
        ]
        preds.append({"boxes": pboxes, "labels": plabels, "scores": scores})
    return {"truths": truths, "predictions": preds, "n_class": n_class}


#: Computed once with pycocotools 2.x as the reference, on the problems `random_problem`
#: produces. Baked in so this agreement is checked on every run, by a suite that does not
#: need pycocotools installed. If one of these changes, the matching or the interpolation
#: changed, and a detection number that disagrees with COCO is a number nobody can compare.
#:
#: Taken from `repr()` of the reference values, not from a formatted table. The first
#: version of this constant was copied out of a `%.10f` printout with the remaining digits
#: invented, and two of the five then disagreed at 4e-11 - which reads exactly like a real
#: difference in the matching, and is not. Transcribe the number, never the rendering.
COCOTOOLS_REFERENCE = {
    #  seed: (mAP@0.5,            mAP@[.50:.95])
    1: (0.08952145214521452, 0.026592409240924087),
    2: (0.07755775577557757, 0.018316831683168316),
    3: (0.06445073078736445, 0.018727015558698722),
    4: (0.04895489548954895, 0.014081408140814078),
    5: (0.12061920477762061, 0.030355178374980352),
}


@pytest.mark.parametrize("seed", sorted(COCOTOOLS_REFERENCE))
def test_mean_average_precision_agrees_with_pycocotools(seed):
    case = random_problem(seed)
    labels = [str(index) for index in range(case["n_class"])]
    expected_half, expected_coco = COCOTOOLS_REFERENCE[seed]

    half = detection_report(
        case["predictions"],
        case["truths"],
        labels=labels,
        iou_thresholds=(0.5,),
        interpolation="101",
    )
    coco = detection_report(
        case["predictions"],
        case["truths"],
        labels=labels,
        iou_thresholds=COCO_THRESHOLDS,
        interpolation="101",
    )

    assert half.mean_average_precision == pytest.approx(expected_half, abs=1e-12)
    assert coco.mean_average_precision == pytest.approx(expected_coco, abs=1e-12)


def test_the_reference_problems_are_not_trivial():
    """A regression test against a reference is worth nothing if the problems are all
    empty or all perfect, so this asserts they have substance."""
    for seed in COCOTOOLS_REFERENCE:
        case = random_problem(seed)
        truths = sum(len(entry["labels"]) for entry in case["truths"])
        preds = sum(len(entry["labels"]) for entry in case["predictions"])
        assert truths >= 5, (seed, truths)
        assert preds >= 5, (seed, preds)
        score = detection_report(
            case["predictions"],
            case["truths"],
            labels=[str(i) for i in range(case["n_class"])],
            interpolation="101",
        ).mean_average_precision
        assert 0 < score < 1, (seed, score)


# --- how well the matched boxes actually fit --------------------------------------------


def test_matching_reports_the_overlap_it_matched_at():
    """The threshold throws this away: a box matching at 0.52 and one at 0.97 are both a
    hit, and only one of them is well placed."""
    _, hit, overlap, _ignored = match_detections([[0, 0, 10, 10]], [0.9], truth([0, 0, 10, 10]))
    assert hit.tolist() == [True]
    assert overlap[0] == pytest.approx(1.0)

    # A looser fit: 0.5 of the way across, IoU 1/3 - below the default threshold, so no hit
    # and no overlap recorded.
    _, hit, overlap, _ignored = match_detections([[5, 0, 15, 10]], [0.9], truth([0, 0, 10, 10]))
    assert hit.tolist() == [False]
    assert overlap[0] == 0.0


def test_the_report_says_how_well_the_matches_fit():
    """Which is the number the AP figures only imply. A detector scoring 0.85 at IoU 0.5
    and 0.50 averaged over 0.50-0.95 is finding its objects and placing them loosely."""
    truths = [one_image([[0, 0, 100, 100]], [0])]
    tight = [one_image([[0, 0, 100, 100]], [0], [0.9])]
    loose = [one_image([[0, 0, 70, 100]], [0], [0.9])]  # IoU 0.7

    assert detection_report(tight, truths, labels=["a"]).mean_matched_iou == pytest.approx(1.0)
    assert detection_report(loose, truths, labels=["a"]).mean_matched_iou == pytest.approx(
        0.7, abs=0.01
    )


def test_the_matched_overlap_is_weighted_by_how_many_matched():
    """A class with two matches should not weigh the same as one with eight hundred."""
    truths = [
        one_image([[0, 0, 100, 100]], [0]),
        one_image([[0, 0, 100, 100], [200, 0, 300, 100], [400, 0, 500, 100]], [1, 1, 1]),
    ]
    preds = [
        one_image([[0, 0, 60, 100]], [0], [0.9]),  # class 0: IoU 0.6
        one_image(
            [[0, 0, 100, 100], [200, 0, 300, 100], [400, 0, 500, 100]], [1, 1, 1], [0.9, 0.9, 0.9]
        ),  # class 1: IoU 1.0
    ]
    report = detection_report(preds, truths, labels=["a", "b"])
    by_class = {row["label"]: row["mean_matched_iou"] for row in report.per_class}
    assert by_class["a"] == pytest.approx(0.6, abs=0.01)
    assert by_class["b"] == pytest.approx(1.0)
    # One match at 0.6 and three at 1.0 weighted gives 0.9, not the 0.8 a mean of means
    # would give.
    assert report.mean_matched_iou == pytest.approx(0.9, abs=0.01)


def test_nothing_matched_means_the_overlap_is_unknown_not_zero():
    truths = [one_image([[0, 0, 100, 100]], [0])]
    preds = [one_image([[500, 500, 600, 600]], [0], [0.9])]
    report = detection_report(preds, truths, labels=["a"])
    assert report.per_class[0]["mean_matched_iou"] is None
    assert report.mean_matched_iou is None


def test_each_IOU_THRESHOLD_scores_its_own_ranking_when_a_difficult_truth_absorbs_a_prediction():
    """`pyplatypus#368`: one ordering was built at the lowest threshold and used at all of them.

    Absorption is `absorbed >= iou_threshold`, so a prediction covering a difficult object
    at IoU 0.8 leaves the ranking at 0.50 to 0.80 and stays in it at 0.85 to 0.95. The
    scores were pooled only at the first threshold, so at the higher ones the flag array
    was longer than the permutation indexing it - no error, and a curve built from the
    wrong predictions. `map_50_95` is where a user meets it.
    """
    difficult = {"boxes": [[0, 0, 10, 10]], "labels": [0], "difficult": [True]}
    on_it = {"boxes": [[0, 0, 10, 8]], "labels": [0], "scores": [0.9]}  # IoU 0.8
    ordinary = {"boxes": [[100, 100, 110, 110]], "labels": [0], "difficult": [False]}
    found = {"boxes": [[100, 100, 110, 110]], "labels": [0], "scores": [0.8]}
    predictions, truths = [on_it, found], [difficult, ordinary]

    thresholds = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
    one_at_a_time = [
        detection_report(
            predictions, truths, labels=["a"], iou_thresholds=(threshold,)
        ).mean_average_precision
        for threshold in thresholds
    ]
    # The absorption holds up to 0.80 and stops: 1.0 while the prediction is set aside,
    # 0.5 once it is judged and ranks first as a false positive.
    assert one_at_a_time == [1.0] * 7 + [0.5] * 3

    together = detection_report(
        predictions, truths, labels=["a"], iou_thresholds=thresholds
    ).mean_average_precision
    assert together == pytest.approx(sum(one_at_a_time) / len(one_at_a_time))
    assert together == pytest.approx(0.85)

    # Two thresholds are enough, and the pair is what the defect answered 0.5 to.
    pair = detection_report(
        predictions, truths, labels=["a"], iou_thresholds=(0.5, 0.85)
    ).mean_average_precision
    assert pair == pytest.approx(0.75)

    # The control: with nothing marked difficult no prediction ever leaves a ranking, so
    # one ordering serves every threshold and the two routes agreed even before the fix.
    plain = [dict(difficult, difficult=[False]), ordinary]
    assert detection_report(
        predictions, plain, labels=["a"], iou_thresholds=thresholds
    ).mean_average_precision == pytest.approx(
        sum(
            detection_report(
                predictions, plain, labels=["a"], iou_thresholds=(threshold,)
            ).mean_average_precision
            for threshold in thresholds
        )
        / len(thresholds)
    )


def test_a_duplicate_may_claim_the_NEXT_unclaimed_truth_which_is_COCOS_rule_not_VOCS():
    """The decision `pyplatypus#369` writes down, so that changing it fails a test by name.

    `voc_eval` takes the argmax over *all* of an image's truths: a detection whose best
    overlap is a truth another prediction already claimed is a false positive there, full
    stop. Here it may claim the next unclaimed truth above the threshold instead, which is
    what COCO does and what `map_50_95` is defined against. Both are defensible; only one
    of them is this package's, and the docstring cited the other one for a while.
    """
    first = [0.0, 0, 10, 10]
    second = [3.0, 0, 13, 10]  # IoU 0.538 with `first`
    claiming = [0.0, 0, 10, 10]  # exactly `first`
    duplicate = [1.0, 0, 11, 10]  # 0.818 with `first`, 0.667 with `second`

    order, hit, overlap, ignored = match_detections(
        [claiming, duplicate], [0.9, 0.8], [first, second], iou_threshold=0.5
    )
    assert list(order) == [0, 1]
    # Both are true positives: the second one's best overlap is the claimed truth, and it
    # matches the unclaimed one instead rather than being refused.
    assert list(hit) == [True, True]
    assert overlap[1] == pytest.approx(0.6667, abs=1e-4)
    assert not ignored.any()

    # The same scene under VOC's rule would score the duplicate as a false positive, so
    # this is the one place the two conventions give different numbers with nothing marked
    # difficult. The control is a duplicate with no second truth to fall back on.
    _, alone, _, _ = match_detections([claiming, duplicate], [0.9, 0.8], [first], iou_threshold=0.5)
    assert list(alone) == [True, False]
