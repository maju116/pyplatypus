"""Anchors fitted to your own boxes, rather than COCO's borrowed.

This is the piece of the old package with no equivalent anywhere in R, and the part of it
that was most clearly right: k-means over box shapes with **intersection over union as the
distance**, not Euclidean.

Euclidean distance on (width, height) is blind to scale, and scale is the whole question
an anchor answers. `(0.02, 0.02)` against `(0.04, 0.04)` and `(0.50, 0.50)` against
`(0.52, 0.52)` sit at exactly the same Euclidean distance, 0.0283 - and the first is a
fourfold error in area, IoU 0.25, while the second is eight per cent, IoU 0.93. A metric
that cannot tell those apart will spend its centres on the large shapes, where absolute
differences are large and the matching is forgiving, and leave the small ones sharing one.
IoU distance measures what the assignment later measures.

Three things changed in carrying it over, and each was checkable.

**Shapes are measured in the space the encoder uses.** The old code divided a box by its
source image's dimensions, which equals the fraction of the network's input only when the
image is stretched to fill it - which is what the old code did. With letterboxing the two
differ on the padded axis: a 100x100 object in a 640x480 image is square after
letterboxing and **33% taller than it is wide** under source normalisation. For a dataset
of one image size the distortion is uniform and the model compensates; it stops being
harmless the moment image sizes are mixed or COCO weights are loaded.

**Anchors are grouped into scales by area, not by width.** Verified against the one
external reference available: COCO's anchors are published in their groups, and sorting
the nine by area reproduces that grouping exactly while sorting by width does not - it
puts (30, 61) in the finest grid and (33, 23) in the middle one, swapping them.

**Nothing is printed or plotted.** The old `generate_anchors` printed a class count and
drew a scatter plot as side effects, and returned a nested list. A function whose output
is another function's argument should return data; `AnchorFit` carries the numbers that
say whether the anchors are any good, which the old one never reported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyplatypus.detection.boxes import Letterbox
from pyplatypus.detection.metrics import DetectionError, iou_matrix


@dataclass(frozen=True)
class AnchorFit:
    """Anchors, and the evidence that they fit.

    `mean_iou` is the quantity k-means with IoU distance maximises: the average overlap
    between a box and the anchor it was assigned. It is the number that says whether these
    anchors describe this data - around 0.5 is poor, above 0.6 is usual for a well-matched
    set, and comparing it across `anchors_per_grid` is how the count is chosen rather than
    assumed. The old implementation reported nothing at all.
    """

    anchors: tuple[tuple[tuple[float, float], ...], ...]
    mean_iou: float
    iterations: int
    converged: bool
    boxes_used: int
    boxes_dropped: int
    per_anchor: list[dict[str, Any]]

    @property
    def flat(self) -> np.ndarray:
        """All anchors as one `(scales * per_grid, 2)` array, in the grouped order."""
        return np.asarray([pair for group in self.anchors for pair in group], dtype=float)

    def as_rows(self) -> list[dict[str, Any]]:
        """One row per anchor, which is the shape R wants."""
        return list(self.per_anchor)


def _shapes_and_labels(
    annotations: Sequence[Any], input_shape: tuple[int, int], letterbox: bool, want_labels: bool
) -> tuple[np.ndarray, np.ndarray]:
    """One implementation, because two would drift.

    `box_shapes` and `shape_table` are two views of this. A second copy of the letterbox
    arithmetic is the kind of duplication that reads as harmless and ends with a plot
    showing boxes in different places from where the anchors were fitted to them - which
    looks like a bad fit rather than like a bug.

    `want_labels` rather than always reading them: `box_shapes` never needed a class and
    requiring one now would tighten a contract for no reason - anything standing in for an
    annotation with boxes and a frame would stop working, which is what happened when the
    first version of this did read them unconditionally.
    """
    shapes, labels = [], []
    for annotation in annotations:
        boxes = np.asarray(annotation.boxes, dtype=float).reshape(-1, 4)
        if boxes.size == 0:
            continue
        if letterbox:
            fit = Letterbox.fit((annotation.height, annotation.width), input_shape)
            boxes = fit.forward(boxes)
            width, height = input_shape[1], input_shape[0]
        else:
            width, height = annotation.width, annotation.height
        shapes.append(
            np.stack(
                [(boxes[:, 2] - boxes[:, 0]) / width, (boxes[:, 3] - boxes[:, 1]) / height], axis=1
            )
        )
        if want_labels:
            labels.append(np.asarray(annotation.labels, dtype=int).reshape(-1))
    if not shapes:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=int)
    return (np.concatenate(shapes), np.concatenate(labels) if labels else np.zeros(0, dtype=int))


def box_shapes(
    annotations: Sequence[Any], *, input_shape: tuple[int, int] = (416, 416), letterbox: bool = True
) -> np.ndarray:
    """Every annotated box as a (width, height) fraction of the network's input.

    `letterbox=False` reproduces the old behaviour - dividing by the source image - and is
    here only so the difference can be measured rather than argued about.
    """
    return _shapes_and_labels(annotations, input_shape, letterbox, want_labels=False)[0]


def shape_table(
    annotations: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    input_shape: tuple[int, int] = (416, 416),
    letterbox: bool = True,
) -> dict[str, list]:
    """The same shapes, with the class each box belongs to. One row per box.

    What `box_shapes` drops, and what makes the picture worth looking at: a cloud of
    widths and heights says how varied the objects are, and the same cloud coloured by
    class says whether a class has anchors near it at all.

    Measured on BCCD, whose three classes have median sides of 133, 69 and 26 pixels at a
    416 input: anchors fitted to all three together cover each of them at 0.88, 0.85 and
    0.83, so fitting to the mixture does not abandon the smallest class. COCO's nine cover
    the same three at 0.64, 0.70 and 0.70 - *evenly* worse rather than blind to one, which
    is the useful thing to know. They are not aimed elsewhere; they span 10 to 373 pixels
    a side because COCO holds objects of every size, and most of that range describes
    nothing here.

    Plain lists rather than arrays: this crosses into R as a data frame.
    """
    shapes, indices = _shapes_and_labels(annotations, input_shape, letterbox, want_labels=True)
    named = [labels[i] if labels is not None and 0 <= i < len(labels) else str(i) for i in indices]
    return {
        "width": shapes[:, 0].tolist(),
        "height": shapes[:, 1].tolist(),
        "label": indices.tolist(),
        "name": named,
    }


def fit_shapes(
    shapes,
    count: int,
    *,
    iterations: int = 100,
    seed: int = 0,
    centroid: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, float, int, bool]:
    """k-means over box shapes with IoU distance. Returns centres, mean IoU, steps, converged.

    k-means++ for the initial centres, which matters here: a plain random start on a
    long-tailed size distribution regularly puts two centres in the same cluster of small
    objects and leaves the large ones unrepresented, and the result looks plausible.
    """
    data = np.asarray(shapes, dtype=float).reshape(-1, 2)
    if data.size == 0:
        raise DetectionError("there are no boxes to fit anchors to")
    if np.any(data <= 0):
        raise DetectionError(
            "a box with a zero side cannot shape an anchor; `drop_degenerate` removes them"
        )
    if count < 1:
        raise DetectionError(f"the number of anchors must be at least 1; got {count}")
    unique = np.unique(data, axis=0)
    if len(unique) < count:
        raise DetectionError(
            f"{count} anchors were asked for but the data has only {len(unique)} distinct "
            f"box shapes. Ask for fewer, or check that the annotations are what you think."
        )

    rng = np.random.default_rng(seed)
    centres = _plus_plus(data, count, rng)
    take_centroid = centroid or (lambda block: block.mean(axis=0))

    assignment = None
    step = 0
    converged = False
    for step in range(1, iterations + 1):
        overlaps = _shape_iou(data, centres)
        nearest = np.argmax(overlaps, axis=1)
        if assignment is not None and np.array_equal(nearest, assignment):
            converged = True
            break
        assignment = nearest
        for index in range(count):
            block = data[nearest == index]
            # An empty cluster is re-seeded from the worst-matched box rather than left,
            # because an anchor nothing chose is a wasted slot in every head.
            if len(block) == 0:
                worst = np.argmin(overlaps.max(axis=1))
                centres[index] = data[worst]
            else:
                centres[index] = take_centroid(block)

    overlaps = _shape_iou(data, centres)
    best = overlaps.max(axis=1)
    return centres, float(best.mean()), step, converged


def generate_anchors(
    annotations: Sequence[Any],
    *,
    anchors_per_grid: int = 3,
    scales: int = 3,
    input_shape: tuple[int, int] = (416, 416),
    iterations: int = 100,
    seed: int = 0,
    centroid: Callable[[np.ndarray], np.ndarray] | None = None,
    letterbox: bool = True,
    minimum_side: float = 1.0,
) -> AnchorFit:
    """Anchors for a set of annotations, grouped coarsest grid first.

    `anchors_per_grid * scales` anchors are fitted at once and then split by area, which is
    YOLOv3's own arrangement: one k-means over every box, and the grids are a partition of
    the result rather than three separate problems.
    """
    shapes = box_shapes(annotations, input_shape=input_shape, letterbox=letterbox)
    height, width = int(input_shape[0]), int(input_shape[1])
    # A box the letterbox shrank below a pixel is not a shape to fit an anchor to.
    keep = (shapes[:, 0] * width >= minimum_side) & (shapes[:, 1] * height >= minimum_side)
    dropped = int((~keep).sum())
    shapes = shapes[keep]

    total = anchors_per_grid * scales
    centres, mean_iou, steps, converged = fit_shapes(
        shapes, total, iterations=iterations, seed=seed, centroid=centroid
    )

    # Largest area first, so the coarsest grid gets the biggest anchors. Verified against
    # COCO's published grouping, which this reproduces and a width ordering does not.
    order = np.argsort(-(centres[:, 0] * centres[:, 1]))
    ordered = centres[order]

    overlaps = _shape_iou(shapes, ordered)
    chosen = np.argmax(overlaps, axis=1)
    per_anchor = []
    for index in range(total):
        mine = overlaps[chosen == index, index]
        per_anchor.append(
            {
                "grid": index // anchors_per_grid,
                "slot": index % anchors_per_grid,
                "width": float(ordered[index, 0]),
                "height": float(ordered[index, 1]),
                "width_pixels": float(ordered[index, 0] * width),
                "height_pixels": float(ordered[index, 1] * height),
                "boxes": int((chosen == index).sum()),
                "mean_iou": float(mine.mean()) if mine.size else None,
            }
        )

    grouped = tuple(
        tuple(
            (float(w), float(h))
            for w, h in ordered[g * anchors_per_grid : (g + 1) * anchors_per_grid]
        )
        for g in range(scales)
    )
    return AnchorFit(
        anchors=grouped,
        mean_iou=mean_iou,
        iterations=steps,
        converged=converged,
        boxes_used=len(shapes),
        boxes_dropped=dropped,
        per_anchor=per_anchor,
    )


def anchor_coverage(shapes, anchors) -> dict[str, Any]:
    """How well a set of anchors already in hand describes some boxes.

    Which is the question to ask before borrowing COCO's: an anchor set fitted to
    photographs of cars and people has no reason to describe blood cells, and this says so
    in one number instead of a training run.
    """
    data = np.asarray(shapes, dtype=float).reshape(-1, 2)
    flat = (
        np.asarray([pair for group in anchors for pair in group], dtype=float)
        if np.ndim(anchors) == 3
        else np.asarray(anchors, dtype=float).reshape(-1, 2)
    )
    if data.size == 0:
        raise DetectionError("there are no boxes to measure coverage over")
    overlaps = _shape_iou(data, flat)
    best = overlaps.max(axis=1)
    return {
        "anchors": len(flat),
        "boxes": len(data),
        "mean_iou": float(best.mean()),
        "median_iou": float(np.median(best)),
        "worst_iou": float(best.min()),
        # A box below a half overlaps its own best anchor less than it misses it, which at
        # the usual matching threshold is an object the model is being asked to find with
        # a template that does not fit it.
        "boxes_below_half": int((best < 0.5).sum()),
    }


def _shape_iou(shapes: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """IoU between shapes and centres, both treated as boxes at the origin."""
    zeros = np.zeros((len(shapes), 2))
    boxes = np.concatenate([zeros, shapes], axis=1)
    anchor_boxes = np.concatenate([np.zeros((len(centres), 2)), centres], axis=1)
    return iou_matrix(boxes, anchor_boxes)


def _plus_plus(data: np.ndarray, count: int, rng) -> np.ndarray:
    """k-means++ seeding, with IoU distance in place of Euclidean."""
    centres = np.empty((count, 2), dtype=float)
    centres[0] = data[rng.integers(0, len(data))]
    for index in range(1, count):
        distance = 1.0 - _shape_iou(data, centres[:index]).max(axis=1)
        weights = distance**2
        total = weights.sum()
        if total <= 0:
            # Every remaining box already sits on a centre, so there is nothing further
            # away to pick; take any distinct shape rather than loop.
            remaining = [
                row for row in data if not any(np.allclose(row, c) for c in centres[:index])
            ]
            centres[index] = remaining[0] if remaining else data[0]
            continue
        centres[index] = data[rng.choice(len(data), p=weights / total)]
    return centres
