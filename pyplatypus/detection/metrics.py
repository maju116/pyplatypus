"""Precision, recall and average precision for boxes.

Four decisions are made here that implementations differ on, so each is named rather than
assumed. Two numbers both honestly called "mAP" can differ by several points on identical
predictions, which is why `detection_report` returns the convention it used alongside the
score.

**Boxes are `(xmin, ymin, xmax, ymax)` in continuous coordinates**, so a box's area is
`(xmax - xmin) * (ymax - ymin)`. Pascal VOC's XML stores pixel indices and its own
evaluation adds 1 to each side, which raises the IoU of small boxes - a 10-pixel box gains
21% of its area. The reader that imports VOC annotations is where that convention is
resolved; by the time boxes reach this module they are continuous.

**Matching is greedy by descending score.** Each prediction takes the highest-IoU
unmatched truth above the threshold; a truth can be claimed once. A second prediction on
the same object is a false positive, which is what makes duplicate boxes cost something
and non-maximum suppression worth doing.

**Predictions are pooled across images before AP is computed**, per class, never averaged
per image. The precision-recall curve is a ranking over every prediction in the dataset,
and a mean of per-image APs is a different quantity: images with one object dominate it,
and an image with no objects of a class has no AP to contribute. This is the same reason
the segmentation metrics sum TP, FP and FN before applying their formula - a ratio of sums
is not the mean of ratios.

**A class with no ground truth has no average precision.** Not zero, which would drag mAP
down for a class the data never contained, and not one. It is reported as `None` and left
out of the mean, and `detection_report` says how many classes that happened to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pyplatypus.errors import PlatypusError


class DetectionError(PlatypusError):
    kind = "detection_error"


Interpolation = Literal["all", "101", "11"]

#: COCO's thresholds: AP averaged over IoU from 0.50 to 0.95 in steps of 0.05.
COCO_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass(frozen=True)
class DetectionMetrics:
    """What a detector scored, and under which conventions.

    `interpolation` and `iou_thresholds` are part of the result and not a footnote: the
    same predictions score differently under each, so a number without them cannot be
    compared to anybody else's.
    """

    mean_average_precision: float | None
    per_class: list[dict[str, Any]]
    iou_thresholds: tuple[float, ...]
    interpolation: Interpolation
    classes_without_truth: list[int] = field(default_factory=list)

    def as_rows(self) -> list[dict[str, Any]]:
        """One row per class plus an overall row, which is the shape R wants."""
        rows = list(self.per_class)
        rows.append({
            "label": "all",
            "class": None,
            "average_precision": self.mean_average_precision,
            "n_truth": sum(row["n_truth"] for row in self.per_class),
            "n_predicted": sum(row["n_predicted"] for row in self.per_class),
            "true_positives": sum(row["true_positives"] for row in self.per_class),
            "false_positives": sum(row["false_positives"] for row in self.per_class),
            "false_negatives": sum(row["false_negatives"] for row in self.per_class),
            "precision": None,
            "recall": None,
        })
        return rows


def _as_boxes(boxes: Any, *, what: str) -> np.ndarray:
    """An (n, 4) float array, or a readable refusal."""
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        return np.zeros((0, 4), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 4:
        raise DetectionError(
            f"{what} must be an (n, 4) array of (xmin, ymin, xmax, ymax); got shape "
            f"{np.asarray(boxes).shape}"
        )
    bad = (array[:, 2] < array[:, 0]) | (array[:, 3] < array[:, 1])
    if bad.any():
        first = array[bad][0]
        raise DetectionError(
            f"{what} contains a box whose maximum is below its minimum: "
            f"{tuple(first)}. Boxes are (xmin, ymin, xmax, ymax), so a reversed pair is "
            f"usually a column order that does not match."
        )
    return array


def iou_matrix(a: Any, b: Any) -> np.ndarray:
    """Pairwise intersection over union, shape `(len(a), len(b))`.

    Zero-area boxes give an IoU of 0 rather than a division error. A box of zero area is
    not refused here because an annotation can legitimately be degenerate and refusing
    mid-evaluation would stop a run over one bad file; `detection_report` counts them.
    """
    first = _as_boxes(a, what="the first argument")
    second = _as_boxes(b, what="the second argument")
    if first.size == 0 or second.size == 0:
        return np.zeros((len(first), len(second)), dtype=float)

    left = np.maximum(first[:, None, 0], second[None, :, 0])
    top = np.maximum(first[:, None, 1], second[None, :, 1])
    right = np.minimum(first[:, None, 2], second[None, :, 2])
    bottom = np.minimum(first[:, None, 3], second[None, :, 3])

    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = ((first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1]))[:, None]
    area_b = ((second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1]))[None, :]
    union = area_a + area_b - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float),
                     where=union > 0)


def match_detections(pred_boxes: Any, pred_scores: Any, truth_boxes: Any, *,
                     iou_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Which predictions hit a truth, in descending order of score.

    Returns `(order, hit)`: the indices that sort the predictions by score, and a boolean
    per prediction in that order. A truth already claimed by a higher-scoring prediction
    cannot be claimed again, so a duplicate box is a false positive.
    """
    predictions = _as_boxes(pred_boxes, what="pred_boxes")
    truths = _as_boxes(truth_boxes, what="truth_boxes")
    scores = np.asarray(pred_scores, dtype=float).ravel()
    if len(scores) != len(predictions):
        raise DetectionError(
            f"{len(predictions)} predicted boxes but {len(scores)} scores; there must be "
            f"one score per box"
        )

    # Descending score. `-scores` with a stable sort keeps the input order among ties,
    # which makes the result reproducible rather than dependent on the sort's internals.
    order = np.argsort(-scores, kind="stable")
    hit = np.zeros(len(order), dtype=bool)
    if len(truths) == 0 or len(predictions) == 0:
        return order, hit

    overlaps = iou_matrix(predictions[order], truths)
    taken = np.zeros(len(truths), dtype=bool)
    for position in range(len(order)):
        candidates = np.where(~taken & (overlaps[position] >= iou_threshold))[0]
        if candidates.size == 0:
            continue
        best = candidates[np.argmax(overlaps[position, candidates])]
        taken[best] = True
        hit[position] = True
    return order, hit


def average_precision(hit: Any, n_truth: int, *,
                      interpolation: Interpolation = "all") -> float | None:
    """Area under the precision-recall curve, for one class.

    `hit` is the boolean sequence from `match_detections`, in descending score order; the
    curve is traced by walking down it. `None` when the class has no ground truth, because
    an average precision over nothing is unknown rather than zero.

    Three conventions, and they are not interchangeable:

    - `"all"` - every point on the curve, with precision made monotonically decreasing
      first. Pascal VOC from 2010 onwards. The exact area.
    - `"101"` - precision at 101 evenly spaced recall levels. What COCO reports.
    - `"11"` - precision at 11 levels, Pascal VOC 2007. Reads highest of the three and is
      here because papers from that era quote it.
    """
    if interpolation not in ("all", "101", "11"):
        raise DetectionError(
            f"interpolation must be 'all', '101' or '11'; got {interpolation!r}"
        )
    if n_truth <= 0:
        return None

    flags = np.asarray(hit, dtype=bool).ravel()
    if flags.size == 0:
        return 0.0

    true_positives = np.cumsum(flags)
    false_positives = np.cumsum(~flags)
    recall = true_positives / n_truth
    precision = true_positives / np.maximum(true_positives + false_positives, 1)

    # Monotonic from the right: the best precision achievable at this recall or beyond.
    # Without it the curve saw-tooths and the area depends on where the teeth fall.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    if interpolation == "all":
        # Area as a sum of rectangles over the recall steps, with a leading 0.
        steps = np.diff(np.concatenate(([0.0], recall)))
        return float(np.sum(steps * envelope))

    levels = np.linspace(0, 1, 101) if interpolation == "101" else np.linspace(0, 1, 11)
    # For each level, the precision at the first recall that reaches it; 0 if none does.
    indices = np.searchsorted(recall, levels, side="left")
    sampled = np.where(indices < len(envelope), envelope[np.minimum(indices, len(envelope) - 1)], 0.0)
    return float(np.mean(sampled))


def detection_report(predictions: Sequence[dict[str, Any]],
                     truths: Sequence[dict[str, Any]], *,
                     labels: Sequence[str] | None = None,
                     iou_thresholds: Sequence[float] = (0.5,),
                     interpolation: Interpolation = "all",
                     score_threshold: float = 0.0) -> DetectionMetrics:
    """Score a whole dataset's detections.

    `predictions` and `truths` are one entry per image, in the same order, each a mapping
    with `boxes` and `labels`, and predictions also with `scores`. An image with nothing
    predicted, or nothing to find, is an entry with empty arrays rather than a gap - a
    missing image would silently shrink the denominator.

    `score_threshold` drops low-confidence predictions before anything else. Leave it at 0
    for average precision, which is defined over the whole ranking: raising it there throws
    away the tail of the curve and inflates the score. It is useful for the precision and
    recall columns, which are single operating points and need one.
    """
    if len(predictions) != len(truths):
        raise DetectionError(
            f"{len(predictions)} images of predictions against {len(truths)} of truth; "
            f"they must line up, with an empty entry for an image that has neither"
        )
    thresholds = tuple(float(t) for t in iou_thresholds)
    if not thresholds:
        raise DetectionError("iou_thresholds is empty; give at least one, e.g. (0.5,)")
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise DetectionError(
                f"an IoU threshold must be above 0 and at most 1; got {threshold}"
            )

    classes = _classes_present(predictions, truths, labels)
    rows: list[dict[str, Any]] = []
    without_truth: list[int] = []

    for index in classes:
        pooled_hits: dict[float, list[np.ndarray]] = {t: [] for t in thresholds}
        pooled_scores: list[np.ndarray] = []
        n_truth = 0
        n_predicted = 0

        for predicted, truth in zip(predictions, truths, strict=True):
            p_boxes, p_scores = _of_class_predicted(predicted, index, score_threshold)
            t_boxes = _of_class_truth(truth, index)
            n_truth += len(t_boxes)
            n_predicted += len(p_boxes)
            if len(p_boxes) == 0:
                continue
            for threshold in thresholds:
                order, hit = match_detections(p_boxes, p_scores, t_boxes,
                                              iou_threshold=threshold)
                pooled_hits[threshold].append(hit)
                if threshold == thresholds[0]:
                    pooled_scores.append(np.asarray(p_scores, dtype=float)[order])

        # Pooled, then re-sorted by score across images: the curve is one ranking over the
        # dataset, so per-image order would make it depend on which image came first.
        scores = np.concatenate(pooled_scores) if pooled_scores else np.zeros(0)
        across = np.argsort(-scores, kind="stable") if scores.size else np.zeros(0, int)

        per_threshold: list[float | None] = []
        for threshold in thresholds:
            flags = (np.concatenate(pooled_hits[threshold]) if pooled_hits[threshold]
                     else np.zeros(0, dtype=bool))
            per_threshold.append(average_precision(flags[across] if flags.size else flags,
                                                   n_truth, interpolation=interpolation))

        known = [value for value in per_threshold if value is not None]
        averaged = float(np.mean(known)) if known else None
        if n_truth == 0:
            without_truth.append(index)

        first = (np.concatenate(pooled_hits[thresholds[0]])
                 if pooled_hits[thresholds[0]] else np.zeros(0, dtype=bool))
        true_positives = int(first.sum())
        false_positives = int((~first).sum())
        false_negatives = int(max(n_truth - true_positives, 0))

        rows.append({
            "label": labels[index] if labels is not None and index < len(labels) else str(index),
            "class": index,
            "average_precision": averaged,
            "n_truth": n_truth,
            "n_predicted": n_predicted,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": (true_positives / n_predicted) if n_predicted else None,
            "recall": (true_positives / n_truth) if n_truth else None,
        })

    scored = [row["average_precision"] for row in rows
              if row["average_precision"] is not None]
    return DetectionMetrics(
        mean_average_precision=float(np.mean(scored)) if scored else None,
        per_class=rows,
        iou_thresholds=thresholds,
        interpolation=interpolation,
        classes_without_truth=without_truth,
    )


def _classes_present(predictions, truths, labels) -> list[int]:
    """Every class the data mentions, or every class the labels declare.

    When labels are given they decide, so a class the model never predicted and the data
    never contained still appears in the report - which is how someone notices that a
    class is missing rather than merely scoring badly.
    """
    if labels is not None:
        return list(range(len(labels)))
    seen: set[int] = set()
    for entry in list(predictions) + list(truths):
        for value in np.asarray(entry.get("labels", []), dtype=int).ravel():
            seen.add(int(value))
    return sorted(seen)


def _of_class_predicted(entry: dict[str, Any], index: int, score_threshold: float):
    boxes = _as_boxes(entry.get("boxes", []), what="predicted boxes")
    labels = np.asarray(entry.get("labels", []), dtype=int).ravel()
    scores = np.asarray(entry.get("scores", []), dtype=float).ravel()
    if len(scores) == 0 and len(boxes) > 0:
        raise DetectionError(
            "predictions need a `scores` entry: average precision is an ordering of "
            "predictions by confidence, and without scores there is no ordering"
        )
    if not (len(boxes) == len(labels) == len(scores)):
        raise DetectionError(
            f"an image's predictions have {len(boxes)} boxes, {len(labels)} labels and "
            f"{len(scores)} scores; the three must agree"
        )
    keep = (labels == index) & (scores >= score_threshold)
    return boxes[keep], scores[keep]


def _of_class_truth(entry: dict[str, Any], index: int) -> np.ndarray:
    boxes = _as_boxes(entry.get("boxes", []), what="truth boxes")
    labels = np.asarray(entry.get("labels", []), dtype=int).ravel()
    if len(boxes) != len(labels):
        raise DetectionError(
            f"an image's truth has {len(boxes)} boxes and {len(labels)} labels; they must "
            f"agree"
        )
    # Pascal VOC marks some objects `difficult` and its own evaluation leaves them out.
    # Honoured when present, because counting them makes a model look worse than the
    # benchmark it is being compared against.
    difficult = np.asarray(entry.get("difficult", np.zeros(len(boxes))), dtype=bool).ravel()
    if len(difficult) != len(boxes):
        raise DetectionError(
            f"`difficult` has {len(difficult)} entries for {len(boxes)} boxes"
        )
    return boxes[(labels == index) & ~difficult]
