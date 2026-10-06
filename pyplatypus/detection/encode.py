"""A list of boxes into the tensors a YOLOv3 head predicts, and back out again.

Three grids, coarse to fine. For a 416 input they are 13, 26 and 52 cells across - strides
32, 16 and 8 - and each cell carries `anchors_per_grid` predictions of `5 + n_class`
numbers. `anchors_per_grid` comes from the anchors given, not from a constant, which is the
flexibility the package this replaces had and most YOLOv3 code does not.

**Anchors are fractions of the input**, as COCO's are: `(116, 90) / 416`. That makes them
independent of the input size, so the same anchors describe the same shapes at 416 and at
608. Largest first, matching the coarsest grid, because that is the order the published
COCO weights were trained in and the head order they expect.

**Each truth box goes to exactly one anchor**, the one whose shape it matches best across
all scales - compared as shapes with both boxes moved to the origin, since an anchor has
no position. That is YOLOv3's assignment, and it is why an image of nothing but platelets
trains only the finest grid.

### The infinity in the old encoding

The package this replaces stored the target as the raw pre-activation values:
`t_x = logit(center_x - floor(center_x))`. That fraction is in `[0, 1)`, and it is **zero
whenever a box's centre lands on a cell boundary** - at which point `logit(0)` is `-inf`.
Measured on a 416 input at grid 13: **37 of 1200** integer-pixel boxes do exactly that, so
roughly one box in thirty poisoned its target, and what the loss then did with an infinity
depended on the framework.

Here the target holds the offset itself, in `[0, 1)`, and the loss applies the sigmoid.
Nothing is inverted, so nothing can be infinite. Width and height stay as `log(side /
anchor)`, which is finite for any box with a positive side - and `drop_degenerate` in
`boxes.py` is what removes the ones without.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyplatypus.detection.metrics import DetectionError, iou_matrix

#: COCO's anchors, as fractions of a 416 input, coarsest grid first.
COCO_ANCHORS: tuple[tuple[tuple[float, float], ...], ...] = (
    ((116 / 416, 90 / 416), (156 / 416, 198 / 416), (373 / 416, 326 / 416)),
    ((30 / 416, 61 / 416), (62 / 416, 45 / 416), (59 / 416, 119 / 416)),
    ((10 / 416, 13 / 416), (16 / 416, 30 / 416), (33 / 416, 23 / 416)),
)

#: What each grid divides the input by, coarsest first.
STRIDES: tuple[int, ...] = (32, 16, 8)


@dataclass(frozen=True)
class Encoding:
    """The targets for one image, plus what could not be placed.

    `unplaced` is not a diagnostic afterthought. Two truths whose centres fall in the same
    cell and whose best anchor is the same one cannot both be represented - the second
    overwrites the first - and a dataset of densely packed objects does this constantly.
    BCCD's red cells touch and overlap by design, so the number is the difference between
    "the model is poor" and "the target never contained half the cells".
    """

    targets: tuple[np.ndarray, ...]
    placed: int
    unplaced: int

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(target.shape for target in self.targets)


def grid_shapes(input_shape: tuple[int, int], scales: int = 3,
                strides: tuple[int, ...] = STRIDES) -> tuple[tuple[int, int], ...]:
    """The cell counts for an input size, coarsest first.

    Refuses a size the strides do not divide, rather than rounding: a grid that does not
    tile the input puts every box a fraction of a cell out, which trains and never says so.
    """
    height, width = int(input_shape[0]), int(input_shape[1])
    out = []
    for stride in strides[:scales]:
        if height % stride or width % stride:
            raise DetectionError(
                f"an input of {(height, width)} is not divisible by {stride}, which the "
                f"coarsest-to-finest strides {strides[:scales]} require. YOLOv3 needs a "
                f"size divisible by {max(strides[:scales])}; {_nearest(height, max(strides[:scales]))}"
                f" and {_nearest(width, max(strides[:scales]))} are the nearest."
            )
        out.append((height // stride, width // stride))
    return tuple(out)


def encode(boxes, labels, *, anchors=COCO_ANCHORS, input_shape: tuple[int, int] = (416, 416),
           n_class: int = 80, strides: tuple[int, ...] = STRIDES) -> Encoding:
    """Boxes in input-pixel coordinates to one target array per grid.

    Each target is `(grid_h, grid_w, anchors_per_grid, 5 + n_class)`: the within-cell
    offsets, the log-ratios to the anchor, an objectness of 1, and a one-hot class.
    """
    flat_anchors, per_grid = _flatten_anchors(anchors)
    shapes = grid_shapes(input_shape, scales=len(anchors), strides=strides)
    height, width = int(input_shape[0]), int(input_shape[1])

    targets = [np.zeros((gh, gw, per_grid, 5 + n_class), dtype=np.float32)
               for gh, gw in shapes]

    array = np.asarray(boxes, dtype=float).reshape(-1, 4) if np.size(boxes) else \
        np.zeros((0, 4), dtype=float)
    tags = np.asarray(labels, dtype=int).ravel()
    if len(array) != len(tags):
        raise DetectionError(f"{len(array)} boxes and {len(tags)} labels; they must agree")
    if len(array) and (tags.min() < 0 or tags.max() >= n_class):
        raise DetectionError(
            f"labels must be in 0..{n_class - 1} for n_class={n_class}; got "
            f"{tags.min()}..{tags.max()}"
        )
    if array.size == 0:
        return Encoding(tuple(targets), placed=0, unplaced=0)

    # Normalised centre and side, so anchors in fractions can be compared directly.
    centre_x = (array[:, 0] + array[:, 2]) / 2 / width
    centre_y = (array[:, 1] + array[:, 3]) / 2 / height
    box_w = (array[:, 2] - array[:, 0]) / width
    box_h = (array[:, 3] - array[:, 1]) / height
    if np.any(box_w <= 0) or np.any(box_h <= 0):
        raise DetectionError(
            "a box with a zero or negative side cannot be encoded - its log-ratio to an "
            "anchor is undefined. `drop_degenerate` removes them, with a count."
        )

    # Shape-only IoU: both the box and each anchor at the origin, because an anchor has no
    # position and matching on position would assign by where an object is, not what it is.
    shifted = np.stack([np.zeros_like(box_w), np.zeros_like(box_h), box_w, box_h], axis=1)
    anchor_boxes = np.concatenate(
        [np.zeros((len(flat_anchors), 2)), flat_anchors], axis=1
    )
    best = np.argmax(iou_matrix(shifted, anchor_boxes), axis=1)

    placed = unplaced = 0
    for index, anchor_index in enumerate(best):
        grid = anchor_index // per_grid
        slot = anchor_index % per_grid
        grid_h, grid_w = shapes[grid]

        # The cell the centre falls in. `min` guards a centre of exactly 1.0, which a box
        # flush with the right or bottom edge produces and which would index past the grid.
        column = min(int(centre_x[index] * grid_w), grid_w - 1)
        row = min(int(centre_y[index] * grid_h), grid_h - 1)

        if targets[grid][row, column, slot, 4] == 1:
            # Already taken by an earlier box of the same shape in the same cell. Counted
            # rather than overwritten silently.
            unplaced += 1
            continue

        anchor_w, anchor_h = flat_anchors[anchor_index]
        targets[grid][row, column, slot, 0] = centre_x[index] * grid_w - column
        targets[grid][row, column, slot, 1] = centre_y[index] * grid_h - row
        targets[grid][row, column, slot, 2] = np.log(box_w[index] / anchor_w)
        targets[grid][row, column, slot, 3] = np.log(box_h[index] / anchor_h)
        targets[grid][row, column, slot, 4] = 1.0
        targets[grid][row, column, slot, 5 + tags[index]] = 1.0
        placed += 1

    return Encoding(tuple(targets), placed=placed, unplaced=unplaced)


def decode(targets, *, anchors=COCO_ANCHORS, input_shape: tuple[int, int] = (416, 416),
           n_class: int = 80, objectness: float = 0.5, strides: tuple[int, ...] = STRIDES,
           raw: bool = False):
    """Target or prediction arrays back to boxes, scores and labels in input pixels.

    `raw=True` treats the arrays as a network's output - offsets and objectness still in
    logit space, class scores too - and applies the sigmoid. `raw=False` reads them as
    `encode` wrote them, which is what makes the two exact inverses and is the only test
    of an encoder worth having.
    """
    flat_anchors, per_grid = _flatten_anchors(anchors)
    shapes = grid_shapes(input_shape, scales=len(anchors), strides=strides)
    height, width = int(input_shape[0]), int(input_shape[1])
    if len(targets) != len(shapes):
        raise DetectionError(
            f"{len(targets)} grids given but the anchors describe {len(shapes)}"
        )

    boxes, scores, labels = [], [], []
    for grid, array in enumerate(targets):
        data = np.asarray(array, dtype=float)
        grid_h, grid_w = shapes[grid]
        expected = (grid_h, grid_w, per_grid, 5 + n_class)
        if data.shape != expected:
            raise DetectionError(
                f"grid {grid} has shape {data.shape}, expected {expected} for an input of "
                f"{(height, width)} with {per_grid} anchors and {n_class} classes"
            )

        confidence = _sigmoid(data[..., 4]) if raw else data[..., 4]
        rows, columns, slots = np.where(confidence >= objectness)
        if rows.size == 0:
            continue

        offset_x = data[rows, columns, slots, 0]
        offset_y = data[rows, columns, slots, 1]
        if raw:
            offset_x, offset_y = _sigmoid(offset_x), _sigmoid(offset_y)
        anchor = flat_anchors[grid * per_grid + slots]

        centre_x = (columns + offset_x) / grid_w * width
        centre_y = (rows + offset_y) / grid_h * height
        box_w = np.exp(data[rows, columns, slots, 2]) * anchor[:, 0] * width
        box_h = np.exp(data[rows, columns, slots, 3]) * anchor[:, 1] * height

        class_scores = data[rows, columns, slots, 5:]
        if raw:
            class_scores = _sigmoid(class_scores)

        boxes.append(np.stack([centre_x - box_w / 2, centre_y - box_h / 2,
                               centre_x + box_w / 2, centre_y + box_h / 2], axis=1))
        labels.append(np.argmax(class_scores, axis=1))
        scores.append(confidence[rows, columns, slots] *
                      np.max(class_scores, axis=1) if raw
                      else confidence[rows, columns, slots])

    if not boxes:
        return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    return (np.concatenate(boxes), np.concatenate(scores),
            np.concatenate(labels).astype(int))


def _flatten_anchors(anchors) -> tuple[np.ndarray, int]:
    """Anchors as one `(scales * per_grid, 2)` array, with the count per grid.

    Flattened in the given order, so index `grid * per_grid + slot` identifies one anchor
    - which is how a single best-match index says both which head and which slot.

    Grouped sequences and grouped arrays are both accepted, and the flattened form is
    refused by name - see `_refuse_flattened`.

    Nothing here may test `anchors` for truth. An ndarray answers `if not anchors` with
    `ValueError: the truth value of an array is ambiguous`, so the guard below used to
    report numpy's complaint about emptiness in place of its own message about empty
    anchors - the one case it exists to explain.
    """
    groups = list(anchors)
    if not groups:
        raise DetectionError("anchors is empty; YOLOv3 needs one group per output grid")
    _refuse_flattened(anchors)
    counts = {len(group) for group in groups}
    if len(counts) != 1:
        raise DetectionError(
            f"every grid needs the same number of anchors, because the head's width is "
            f"anchors_per_grid * (n_class + 5); got {[len(g) for g in groups]}"
        )
    flat = np.asarray([pair for group in groups for pair in group], dtype=float)
    if flat.ndim != 2 or flat.shape[1] != 2:
        raise DetectionError(
            f"each anchor is a (width, height) pair as a fraction of the input; got an "
            f"array of shape {flat.shape}"
        )
    if np.any(flat <= 0):
        raise DetectionError("an anchor's width and height must both be above 0")
    if np.any(flat > 1):
        raise DetectionError(
            "anchors are fractions of the input, so they are at most 1. COCO's are given "
            "as pixels at 416 and must be divided by it: (116, 90) / 416."
        )
    return flat, counts.pop()


def _refuse_flattened(anchors) -> None:
    """Refuse an (N, 2) array of pairs, which is the flattened form rather than the grouped.

    `AnchorFit.flat` is exactly this shape and is the obvious thing to hand `encode` after
    fitting anchors, so the mistake is one the package invites. A flat array has thrown
    away how the anchors divide between the output grids, which is half of what this
    function returns; dividing by three would be right for YOLOv3 and wrong for any model
    with a different number of heads, and guessing it is how an anchor ends up assigned to
    the wrong stride with nothing to show it.
    """
    try:
        array = np.asarray(anchors, dtype=float)
    except (TypeError, ValueError):
        return  # ragged or not numeric; the per-group checks say so more precisely
    if array.ndim == 2 and array.shape[1] == 2:
        raise DetectionError(
            f"anchors came as a flat {array.shape} array of (width, height) pairs, but they "
            f"are grouped one group per output grid. A flat array cannot say how they "
            f"divide between the grids. If this is an AnchorFit, pass `.anchors` rather "
            f"than `.flat`; otherwise group them, e.g. "
            f"(((0.03, 0.03), ...), ((0.08, 0.07), ...), ((0.28, 0.21), ...))."
        )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _nearest(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)
