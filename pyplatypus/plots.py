"""Pictures of what a model did.

The engine could compute a mask and not show it, which made the claim that this is an
ecosystem rather than a library false on this side: none of the example scripts produced a
figure, so a Python user trained a model, got arrays back, and wrote their own matplotlib.

**The names are the R package's**, and this is the one place the project's naming rule
inverts. The rule is that the engine's name wins, because it is the layer that cannot be
renamed later without changing the configuration format - but these five do not exist in the
engine, so there is nothing for it to win with, and R had them first.

Two of the five need no plotting library at all: `overlay_mask` and `overlay_agreement`
composite one array onto another and hand back a picture, which is what the R versions do.
The three `plot_*` return a matplotlib `Figure` - never `plt.show()`, never a file, because
the caller decides where it goes and that is also the only version testable without a
display. The R equivalents return a ggplot object for the same reason.

Every colour, the alpha and the label format come from `pyplatypus.style`, so this module
makes no decisions of its own.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from pyplatypus.data.masks import onehot_to_colours
from pyplatypus.errors import PlatypusError
from pyplatypus.style import (
    AGREEMENT_COLOURS,
    BOX_COLOURS,
    BOX_LABEL_FORMAT,
    BOX_MIN_SCORE,
    OVERLAY_ALPHA,
)

__all__ = [
    "overlay_agreement",
    "overlay_mask",
    "plot_anchors",
    "plot_boxes",
    "plot_masks",
]


def _rgb(colour: str) -> np.ndarray:
    """A `#rrggbb` string as three 0-255 numbers."""
    return np.array(matplotlib.colors.to_rgb(colour)) * 255.0


def _as_picture(image: np.ndarray) -> np.ndarray:
    """An image as float 0-255, three channels, however it arrived.

    Images reach here either as the 0-1 floats a loader produced or as the 0-255 integers a
    file held, and guessing from the dtype alone is wrong for a float array that happens to
    hold 0-255. The range decides: a float image whose maximum is at most 1 is scaled.
    """
    picture = np.asarray(image, dtype=np.float64)
    if picture.ndim == 2:
        picture = picture[:, :, None]
    if picture.shape[-1] == 1:
        picture = np.repeat(picture, 3, axis=-1)
    if picture.shape[-1] != 3:
        raise PlatypusError(
            f"an image needs 1 or 3 channels to be drawn, got {picture.shape[-1]}; "
            "a prediction with one channel per class is a mask rather than an image"
        )
    if np.issubdtype(np.asarray(image).dtype, np.floating) and picture.max() <= 1.0:
        picture = picture * 255.0
    return picture


def _blend(picture: np.ndarray, tint: np.ndarray, where: np.ndarray, alpha: float) -> None:
    """Mix `tint` into `picture` where `where`, in place."""
    for channel in range(3):
        layer = picture[:, :, channel]
        layer[where] = (1.0 - alpha) * layer[where] + alpha * tint[channel]


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: list[tuple[int, int, int]],
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """An image with a mask tinted onto it.

    Args:
        image: one image, `height x width x channels`, 0-1 floats or 0-255 integers.
        mask: a prediction for it - one-hot or probabilities, `height x width x classes` -
            or an array of class indices.
        colormap: one colour per class, darkest first, as the specification gives it.
        alpha: how much of the tint shows. The default keeps the tissue legible underneath,
            which is the point of an overlay.

    Returns:
        A `height x width x 3` array of 0-255 integers, which is what the R
        `overlay_mask()` returns too.

    Raises:
        PlatypusError: if the image cannot be drawn, or the mask does not cover it.

    >>> import numpy as np
    >>> image = np.full((4, 4, 3), 0.5)
    >>> mask = np.zeros((4, 4, 2)); mask[1:3, 1:3, 1] = 1
    >>> shown = overlay_mask(image, mask, [(0, 0, 0), (255, 0, 0)])
    >>> shown.shape, shown.dtype
    ((4, 4, 3), dtype('uint8'))
    >>> bool((shown[1, 1] != shown[0, 0]).any())
    True
    """
    picture = _as_picture(image)
    classes = np.asarray(mask)
    if classes.ndim == 3 and classes.shape[-1] > 1:
        coloured = onehot_to_colours(classes, colormap)
    else:
        indices = classes.reshape(classes.shape[:2])
        coloured = np.asarray(colormap, dtype=np.uint8)[np.clip(indices, 0, len(colormap) - 1)]
    if coloured.shape[:2] != picture.shape[:2]:
        raise PlatypusError(
            f"the mask is {coloured.shape[:2]} and the image is {picture.shape[:2]}; "
            "a mask drawn over an image it was not computed from lands in the wrong place"
        )

    # Class 0 is background and is not tinted: tinting it would cover the whole image, which
    # is the same thing as showing nothing.
    first = np.asarray(colormap[0], dtype=np.float64)
    foreground = (coloured.astype(np.float64) != first).any(axis=-1)
    for channel in range(3):
        layer = picture[:, :, channel]
        tint = coloured[:, :, channel].astype(np.float64)
        layer[foreground] = (1.0 - alpha) * layer[foreground] + alpha * tint[foreground]
    return np.clip(np.round(picture), 0, 255).astype(np.uint8)


def overlay_agreement(
    image: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    colours: dict[str, str] | None = None,
) -> np.ndarray:
    """An image showing what was found, what was missed and what was invented.

    Three colours rather than one, because a missed lesion and a false alarm cost different
    things and a single-colour overlay hides which of the two is being looked at.

    Args:
        image: one image, as `overlay_mask` takes it.
        prediction: class indices or a one-hot array; anything above class 0 counts as
            foreground.
        truth: the same, for the annotation.
        alpha: how much of the tint shows.
        colours: keyed `hit`, `missed`, `false_alarm`. Defaults to the engine's, which the
            R package reads too, so the two cannot disagree about what red means.

    Returns:
        A `height x width x 3` array of 0-255 integers.

    Raises:
        PlatypusError: if the shapes do not correspond, or a colour is missing.

    >>> import numpy as np
    >>> image = np.full((4, 4, 3), 0.5)
    >>> predicted = np.zeros((4, 4), dtype=int); predicted[1:3, 1:3] = 1
    >>> actual = np.zeros((4, 4), dtype=int); actual[2:4, 1:3] = 1
    >>> shown = overlay_agreement(image, predicted, actual)
    >>> shown.shape
    (4, 4, 3)
    >>> bool((shown[1, 1] != shown[3, 1]).any())   # invented, and missed
    True

    `any` rather than `all`: amber and red share their blue channel - both end `3c` - so no
    overlay can separate those two there, and a check that said `all` would be asserting
    something the palette cannot deliver.
    """
    chosen = AGREEMENT_COLOURS if colours is None else colours
    missing = {"hit", "missed", "false_alarm"} - set(chosen)
    if missing:
        raise PlatypusError(
            f"colours is missing {sorted(missing)}; all three are needed, because the "
            "point of this picture is telling the three apart"
        )

    picture = _as_picture(image)
    predicted = _foreground(prediction)
    actual = _foreground(truth)
    if predicted.shape != actual.shape:
        raise PlatypusError(
            f"the prediction is {predicted.shape} and the truth is {actual.shape}; "
            "agreement between two arrays of different shapes is not defined"
        )
    if predicted.shape != picture.shape[:2]:
        raise PlatypusError(f"the masks are {predicted.shape} and the image is {picture.shape[:2]}")

    for part, where in (
        ("hit", predicted & actual),
        ("missed", ~predicted & actual),
        ("false_alarm", predicted & ~actual),
    ):
        _blend(picture, _rgb(chosen[part]), where, alpha)
    return np.clip(np.round(picture), 0, 255).astype(np.uint8)


def _foreground(mask: np.ndarray) -> np.ndarray:
    """Anything above class 0, whether the mask is indices or one-hot."""
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] > 1:
        return array.argmax(axis=-1) > 0
    return array.reshape(array.shape[:2]) > 0


def _stack(images: np.ndarray) -> np.ndarray:
    """One image or many, always returned as many."""
    array = np.asarray(images)
    if array.ndim in (2, 3) and (array.ndim == 2 or array.shape[-1] in (1, 3)):
        return array[None, ...]
    return array


def plot_masks(
    images: np.ndarray,
    prediction: np.ndarray | None = None,
    truth: np.ndarray | None = None,
    colormap: list[tuple[int, int, int]] | None = None,
    which: list[int] | None = None,
    alpha: float = OVERLAY_ALPHA,
    labels: list[str] | None = None,
) -> Figure:
    """Images, their masks and where the two disagree, one row per image.

    The columns are the ones the R `plot_masks()` draws, in the same order and for the same
    reason: the image, the truth, the prediction, and the agreement only when both are
    given, because agreement between a prediction and nothing is not defined.

    Args:
        images: one image or a stack, `[image] x height x width x channels`.
        prediction: one-hot or probabilities per image, or class indices.
        truth: the annotation, same shapes.
        colormap: one colour per class, darkest first. Required to draw a mask.
        which: which images to draw. The default is the first four, because a montage of
            forty is not a figure anybody reads.
        alpha: how much of the tint shows.
        labels: a row label each. The default numbers them.

    Returns:
        A matplotlib `Figure`. Never shown and never saved - the caller decides.

    Raises:
        PlatypusError: if a mask is given without a colormap, or the counts disagree.

    >>> import numpy as np
    >>> image = np.full((1, 8, 8, 3), 0.5)
    >>> mask = np.zeros((1, 8, 8, 2)); mask[0, 2:5, 2:5, 1] = 1
    >>> figure = plot_masks(image, prediction=mask, truth=mask, colormap=[(0, 0, 0), (255, 0, 0)])
    >>> [axis.get_title() for axis in figure.axes]
    ['image', 'truth', 'prediction', 'agreement']
    """
    stack = _stack(images)
    if which is None:
        which = list(range(min(4, len(stack))))
    if (prediction is not None or truth is not None) and colormap is None:
        raise PlatypusError(
            "drawing a mask needs `colormap`, the one the specification gave - without it "
            "there is no way to know which colour a class is"
        )

    columns = ["image"]
    if truth is not None:
        columns.append("truth")
    if prediction is not None:
        columns.append("prediction")
    if truth is not None and prediction is not None:
        columns.append("agreement")

    figure = Figure(figsize=(3.2 * len(columns), 3.2 * len(which)), layout="constrained")
    axes = figure.subplots(len(which), len(columns), squeeze=False)
    for row, index in enumerate(which):
        drawn = {
            "image": _as_picture(stack[index]).astype(np.uint8),
            "truth": (
                None
                if truth is None
                else overlay_mask(stack[index], _stack(truth)[index], colormap, alpha)
            ),
            "prediction": (
                None
                if prediction is None
                else overlay_mask(stack[index], _stack(prediction)[index], colormap, alpha)
            ),
            "agreement": (
                None
                if truth is None or prediction is None
                else overlay_agreement(
                    stack[index], _stack(prediction)[index], _stack(truth)[index], alpha
                )
            ),
        }
        for column, name in enumerate(columns):
            axis = axes[row][column]
            axis.imshow(drawn[name])
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(name)
            if column == 0:
                label = labels[row] if labels else f"image {index + 1}"
                axis.set_ylabel(label)
    return figure


def plot_boxes(
    images: np.ndarray,
    boxes: list[dict[str, Any]],
    truth: list[dict[str, Any]] | None = None,
    which: list[int] | None = None,
    min_score: float = BOX_MIN_SCORE,
    colours: dict[str, str] | None = None,
    labels: list[str] | None = None,
) -> Figure:
    """Detections drawn on the images they were found in, one row per image.

    Args:
        images: one image or a stack.
        boxes: what `DetectionEngine.predict()` returned - one record per image, with
            `boxes`, `labels` and `scores` in that image's own pixels.
        truth: the annotation in the same shape, drawn in a second colour.
        which: which images to draw; the default is the first four.
        min_score: boxes below this are not drawn. **Not** the engine's `operating_point`,
            which sits near zero so average precision can integrate a whole ranking.
        colours: keyed `prediction` and `truth`; defaults to the engine's.
        labels: a row label each.

    Returns:
        A matplotlib `Figure`.

    Raises:
        PlatypusError: if there is not one record per image.

    >>> import numpy as np
    >>> found = [{"boxes": [[1, 1, 5, 5]], "labels": ["cell"], "scores": [0.91]}]
    >>> figure = plot_boxes(np.full((1, 8, 8, 3), 0.5), found)
    >>> len(figure.axes[0].patches), figure.axes[0].texts[0].get_text()
    (1, 'cell 0.91')
    """
    stack = _stack(images)
    chosen = BOX_COLOURS if colours is None else colours
    if len(boxes) != len(stack):
        raise PlatypusError(
            f"{len(boxes)} record(s) for {len(stack)} image(s); `predict()` returns one "
            "per image and they have to stay paired, because a box drawn on the wrong "
            "frame is wrong in a way no score reports"
        )
    if which is None:
        which = list(range(min(4, len(stack))))

    figure = Figure(figsize=(4.0, 4.0 * len(which)), layout="constrained")
    axes = figure.subplots(len(which), 1, squeeze=False)
    for row, index in enumerate(which):
        axis = axes[row][0]
        axis.imshow(_as_picture(stack[index]).astype(np.uint8))
        axis.set_xticks([])
        axis.set_yticks([])
        if truth is not None:
            _draw_boxes(axis, truth[index], chosen["truth"], None)
        _draw_boxes(axis, boxes[index], chosen["prediction"], min_score)
        axis.set_ylabel(labels[row] if labels else f"image {index + 1}")
    return figure


def _draw_boxes(axis, record: dict[str, Any], colour: str, min_score: float | None) -> None:
    """One record's boxes onto one axis, labelled where there is a score to label with."""
    found = np.asarray(record.get("boxes", []), dtype=float).reshape(-1, 4)
    names = list(record.get("labels", []))
    scores = list(record.get("scores", []))
    for position, box in enumerate(found):
        score = scores[position] if position < len(scores) else None
        if min_score is not None and score is not None and score < min_score:
            continue
        x_min, y_min, x_max, y_max = box
        axis.add_patch(
            matplotlib.patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=colour,
                linewidth=1.4,
            )
        )
        if score is not None and position < len(names):
            axis.text(
                x_min,
                y_min - 2,
                BOX_LABEL_FORMAT.format(label=names[position], score=score),
                color=colour,
                fontsize=7,
            )


def plot_anchors(
    anchors: list[list[tuple[float, float]]],
    boxes: np.ndarray | None = None,
    log: bool = True,
) -> Figure:
    """Fitted anchors against the box shapes they were fitted to.

    Args:
        anchors: one group per grid, each a list of `(width, height)` pairs as fractions of
            the input.
        boxes: the shapes themselves, `n x 2` of widths and heights, drawn underneath.
        log: both axes on a log scale. Detection datasets span a wide range of sizes - a
            BCCD platelet is about two fifths the side of a red cell - and on linear axes
            the smallest class collapses into the corner.

    Returns:
        A matplotlib `Figure`.

    >>> figure = plot_anchors([[(0.1, 0.1), (0.2, 0.3)], [(0.4, 0.5)]])
    >>> len(figure.axes[0].collections), figure.axes[0].get_yscale()
    (2, 'log')
    """
    figure = Figure(figsize=(5.0, 5.0), layout="constrained")
    axis = figure.subplots()
    if boxes is not None:
        shapes = np.asarray(boxes, dtype=float).reshape(-1, 2)
        axis.scatter(shapes[:, 0], shapes[:, 1], s=6, alpha=0.3, color="#999999", label="boxes")
    for group, pairs in enumerate(anchors):
        wide = np.asarray(pairs, dtype=float).reshape(-1, 2)
        axis.scatter(wide[:, 0], wide[:, 1], s=70, marker="x", label=f"grid {group + 1}")
    if log:
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlabel("width, fraction of the input")
    axis.set_ylabel("height, fraction of the input")
    axis.legend(loc="upper left", fontsize=8)
    return figure
