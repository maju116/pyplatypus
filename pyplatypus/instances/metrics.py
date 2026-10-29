"""Average precision over instances, the way the Data Science Bowl scored it.

**This is not the detection module's mAP and the difference is not cosmetic.** Four
decisions separate them, and two numbers both honestly called "average precision" differ by
tens of points on identical predictions, so each is named here rather than assumed.

**The formula is the challenge's: `TP / (TP + FP + FN)` at each IoU threshold, averaged over
the thresholds.** It is not a precision-recall curve and nothing is interpolated, because
nothing is ranked - an instance post-process produces regions, not scored proposals. The
detection module pools predictions across images and integrates a curve; doing that here
would need a confidence per instance that no post-process in this package produces.

**Thresholds run 0.5 to 0.95 in steps of 0.05, and anything below 0.5 is refused.** At an
IoU strictly above 0.5 a prediction can satisfy at most one truth object and a truth object
at most one prediction: each of two disjoint regions clearing 0.5 would have to cover more
than half of the same third region, and two halves do not fit in one whole. So counting
matches needs no assignment algorithm and the count is exactly the one a greedy matcher would
reach. **The strictness is load-bearing** - the comparison is `iou > threshold`, not `>=`,
and with `>=` the threshold 0.5 the challenge itself uses would admit two predictions each
covering exactly half of one object. Below 0.5 no comparison saves it: one prediction can
match several objects and TP is silently inflated, so it is refused rather than quietly
turned into a different metric.

**The mean is over images, not over objects.** That is the leaderboard's quantity: an image
of 2 nuclei weighs as much as one of 375, and this dataset holds both. A ratio of pooled
counts would be a defensible number and a different one, and `summarise_instances` reports
the challenge's and says so rather than reporting both - which it claimed to do, and did not,
until `pyplatypus#326`.

**Instances are a label map, and overlapping truth is refused.** One integer per pixel,
which is all this dataset needs and all that can be scored without deciding what an overlap
means. Making the mask layer hold more than one thing per pixel is open work -
`pyplatypus#158`, with `pyplatypus#91` - and a metric that quietly took the topmost label
would answer the question wrongly rather than decline it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyplatypus.errors import PlatypusError

#: 0.5 to 0.95 in steps of 0.05, which is what the 2018 Data Science Bowl swept.
CHALLENGE_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


class InstanceError(PlatypusError):
    kind = "instance_error"


def label_map(masks: Sequence[np.ndarray]) -> np.ndarray:
    """One integer per pixel from a list of per-object masks, refusing overlap.

    The Data Science Bowl stores one PNG per nucleus, which is the representation this reads;
    `unite_masks()` is the other thing to do with those files and it takes the element-wise
    maximum, so it is where instance identity is thrown away.

    Overlap is refused rather than resolved. Two objects claiming a pixel is either a dataset
    this metric cannot score or a representation question that is open on purpose
    (`pyplatypus#158`); either way the honest answer is not to pick one.
    """
    if not masks:
        raise InstanceError("no instance masks to label")
    shape = masks[0].shape
    labels = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        if mask.shape != shape:
            raise InstanceError(
                f"instance {index - 1} has shape {mask.shape} but instance 0 has {shape}; "
                "every mask for one image has to cover the same grid"
            )
        taken = (labels > 0) & (mask > 0)
        if taken.any():
            raise InstanceError(
                f"instance {index - 1} overlaps an earlier one on {int(taken.sum())} pixels. "
                "This metric scores a label map, which holds one object per pixel; a mask "
                "layer that holds more than one is open work (pyplatypus#158)"
            )
        labels[mask > 0] = index
    return labels


def object_labels(labels: np.ndarray) -> np.ndarray:
    """The object numbers a map carries, ascending - which is not `1..max`.

    One definition, because the rows of `iou_matrix` are in this order and anything that
    turns a row back into an object number has to use the same list. `overlay_instance_
    agreement` did it with `row + 1` and was correct only while the numbering was
    consecutive; when `pyplatypus#325` made the matrix's rows the values present, that line
    started colouring the wrong objects on a gapped map - and the R suite's own `gapped`
    fixture, compared against this drawing pixel for pixel, is what caught it.

    So this is a second copy avoided rather than added: `measure.instance_sizes` says the
    same thing in prose - *"a model that found objects 1, 2 and 4 found three objects"* - and
    `rplatypus`'s matcher says it in R.
    """
    return np.unique(labels[labels > 0]).astype(np.int64)


def iou_matrix(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Pairwise IoU between every truth label and every predicted label.

    Computed from a contingency table rather than pair by pair: at 375 instances in one
    image - which this dataset has - the pairwise form is 140,000 full-image comparisons and
    this is one pass. Background is label 0 and is not an object on either side.
    """
    if truth.shape != predicted.shape:
        raise InstanceError(
            f"truth is {truth.shape} and the prediction is {predicted.shape}; "
            "they have to cover the same grid to be compared"
        )
    # **An object is a number the map actually carries, not every number up to the
    # maximum.** Indexing by `max()` made a missing number into an object of zero area,
    # which matches nothing and is counted as a false negative or a false positive: two
    # 3x3 objects numbered 1 and 4, scored against themselves, came out at an average
    # precision of 0.333 rather than 1.0. `pyplatypus#325`.
    #
    # `measure.instance_sizes` has said the right thing the whole time - *"a model that
    # found objects 1, 2 and 4 found three objects"* - and so has `rplatypus`'s own
    # matcher, which takes `sort(unique(...))` of each side. This is the second place in
    # this project where the engine held the wrong copy of a rule R held correctly.
    #
    # A gap is not somebody else's mistake either: `instances_from_parts` produces one at
    # its own default, because a seed outside the foreground loses every pixel and
    # `_drop_small` renumbers only when `minimum_area > 0`.
    truth_values = object_labels(truth)
    predicted_values = object_labels(predicted)
    n_truth, n_predicted = len(truth_values), len(predicted_values)
    if n_truth == 0 or n_predicted == 0:
        return np.zeros((n_truth, n_predicted), dtype=np.float64)

    # The values are mapped onto 1..n so the single-pass count below still works: a lookup
    # array indexed by the label itself, which costs the maximum label in memory and keeps
    # the whole function one pass over the image. Reindexing is what makes the rows and
    # columns mean *objects*; the bincount trick is unchanged.
    truth_index = np.zeros(int(truth_values[-1]) + 1, dtype=np.int64)
    truth_index[truth_values] = np.arange(1, n_truth + 1)
    predicted_index = np.zeros(int(predicted_values[-1]) + 1, dtype=np.int64)
    predicted_index[predicted_values] = np.arange(1, n_predicted + 1)

    truth = truth_index[np.clip(truth.astype(np.int64), 0, len(truth_index) - 1)]
    predicted = predicted_index[np.clip(predicted.astype(np.int64), 0, len(predicted_index) - 1)]

    # One bin per (truth, predicted) pair, background included, so the row and column sums
    # below are the object areas without a second pass over the image.
    flat = truth.ravel().astype(np.int64) * (n_predicted + 1) + predicted.ravel().astype(np.int64)
    table = np.bincount(flat, minlength=(n_truth + 1) * (n_predicted + 1)).reshape(
        n_truth + 1, n_predicted + 1
    )
    intersection = table[1:, 1:].astype(np.float64)
    truth_area = table[1:, :].sum(axis=1, keepdims=True).astype(np.float64)
    predicted_area = table[:, 1:].sum(axis=0, keepdims=True).astype(np.float64)
    union = truth_area + predicted_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def average_precision(
    iou: np.ndarray, thresholds: Sequence[float] = CHALLENGE_THRESHOLDS
) -> float | None:
    """The challenge's per-image score from an IoU matrix, or `None` when there is nothing to score.

    `None` rather than zero when the image holds no truth objects and nothing was predicted:
    zero would say the prediction was wrong where there was nothing to be wrong about, and it
    would drag a mean down for images that contain nothing. The detection module reports a
    class with no ground truth the same way and for the same reason.
    """
    for threshold in thresholds:
        if threshold < 0.5:
            raise InstanceError(
                f"threshold {threshold} is below 0.5, where one prediction can satisfy two "
                "truth objects and the match is no longer one to one. From 0.5 up, counting "
                "matches is exact because the comparison is strict; below it this would need "
                "an assignment algorithm and would not be the challenge's number"
            )
    n_truth, n_predicted = iou.shape
    if n_truth == 0 and n_predicted == 0:
        return None

    scores = []
    for threshold in thresholds:
        hit = iou > threshold
        true_positive = int(hit.any(axis=1).sum()) if n_truth else 0
        false_negative = n_truth - true_positive
        false_positive = n_predicted - true_positive
        scores.append(true_positive / (true_positive + false_positive + false_negative))
    return float(np.mean(scores))


def score_cases(
    cases: Sequence[tuple[str, np.ndarray, np.ndarray]],
    thresholds: Sequence[float] = CHALLENGE_THRESHOLDS,
) -> list[dict]:
    """One row per case: the score, and the counts that explain it.

    The counts are at the lowest threshold, which is where a prediction looks its best - so a
    row with many objects found and a low score is one whose masks are roughly right and
    precisely wrong, and the two failures read differently.
    """
    rows = []
    for name, truth, predicted in cases:
        iou = iou_matrix(truth, predicted)
        lowest = min(thresholds)
        hit = iou > lowest
        matched = int(hit.any(axis=1).sum()) if iou.size else 0
        rows.append(
            {
                "case": name,
                "average_precision": average_precision(iou, thresholds),
                "truth": int(iou.shape[0]),
                "predicted": int(iou.shape[1]),
                f"matched_at_{lowest}": matched,
            }
        )
    return rows


def foreground_dice(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    """Binary overlap of anything the model called an object against anything that is one.

    The mask's own number in an instance table, and the first thing to read in one: if the
    foreground moved, the instance score moved for a reason no decoding can explain. That
    role needs one quantity, and the training metric `dice` is not one - it reduces over the
    class channels a head has, three for `rim` (background, interior, rim band) against two
    for `flows`, so on the Data Science Bowl split the two models read **0.09 apart** where
    their masks are **0.0062 apart and in the other direction** (`pyplatypus#222`).

    Three decisions, each written down because each is worth more than some of the effects
    this task measures:

    - **Both sides are binarised and no channel is weighted.** `predicted` is a foreground
      as `read_prediction` returns it, which is the one place the channel layout is read;
      `truth` is a label map, so `truth > 0` is its foreground however the objects are
      numbered.
    - **Unsmoothed.** The `+1` the training metrics carry makes an empty prediction on empty
      truth score 1.0. Here that case has nothing to score and says so by returning `None`,
      which is this module's convention for it - `score_cases` leaves an unscorable case's
      `average_precision` at `None` and `summarise_instances` filters on exactly that.
    - **Per image.** The caller means over images, which is how `average_precision` is pooled
      in the same row - one pooling inside one row, so a reader comparing two columns is
      comparing the same kind of average.

      **This bullet used to argue from a pooling difference that no longer exists**, and the
      argument carried a number measured before the fix: it said the training metric pools
      over *batches* with each weighed equally, at 0.0088 on that split. `Trainer._run_epoch`
      weights by **examples** since `maju116/rplatypus#196`, and
      `test_an_epoch_line_is_the_mean_over_examples_whatever_the_batching` holds it there. So
      the figure is gone rather than re-measured - §3C says a number in prose needs a test
      holding it or no number - and what is left is the reason that is still true.
      `pyplatypus#327`.

    Args:
        predicted: the predicted foreground, boolean or anything truthy per pixel.
        truth: a label map, objects numbered from 1 and 0 the background.

    Returns:
        The Dice, or `None` if neither side has any foreground - which is a case with
        nothing to score rather than a perfect one.

    >>> truth = np.array([[0, 1], [1, 0]])
    >>> foreground_dice(np.array([[False, True], [True, False]]), truth)
    1.0
    >>> foreground_dice(np.array([[True, True], [False, False]]), truth)
    0.5
    >>> foreground_dice(np.array([[True, False], [False, True]]), truth)
    0.0
    >>> foreground_dice(np.zeros((2, 2), bool), np.zeros((2, 2), int)) is None
    True
    >>> foreground_dice(np.ones((2, 2), bool), np.zeros((2, 2), int))
    0.0
    """
    called = np.asarray(predicted).astype(bool)
    actual = np.asarray(truth) > 0
    total = int(called.sum()) + int(actual.sum())
    if not total:
        return None
    return float(2 * int(np.logical_and(called, actual).sum()) / total)


def summarise_instances(rows: Sequence[dict]) -> dict:
    """The leaderboard's number, and the one it is often confused with.

    `mean_average_precision` is the mean over images, which is what the challenge ranked on.
    A **pooled** figure - the same formula over the summed counts, so every object weighs the
    same rather than every image - is a different quantity, and this dataset makes the gap
    visible: it holds images of 2 objects and images of 375.

    **This function does not return the pooled one, and said it did until
    `pyplatypus#326`.** It cannot be computed from what it is given either: `score_cases`
    hands it one `matched_at_<lowest>` key, and a pooled average precision needs the TP, FP
    and FN at *every* threshold. So the promise is removed rather than honoured here - the
    figure is a feature with a public return shape to change, and it is not the leaderboard's
    number, which is what this function is for.
    """
    scored = [row for row in rows if row["average_precision"] is not None]
    if not scored:
        raise InstanceError("no case had anything to score")
    truth = sum(row["truth"] for row in scored)
    predicted = sum(row["predicted"] for row in scored)
    return {
        "n": len(scored),
        "unscored": len(rows) - len(scored),
        "mean_average_precision": float(np.mean([row["average_precision"] for row in scored])),
        "median": float(np.median([row["average_precision"] for row in scored])),
        "worst": min(scored, key=lambda row: row["average_precision"])["case"],
        "truth_objects": truth,
        "predicted_objects": predicted,
    }
