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

from pyplatypus.data.masks import MaskError, onehot_to_colours
from pyplatypus.errors import PlatypusError
from pyplatypus.style import (
    AGREEMENT_COLOURS,
    BOX_COLOURS,
    BOX_LABEL_FORMAT,
    BOX_MIN_SCORE,
    CLASS_COLOURS,
    OVERLAY_ALPHA,
)

__all__ = [
    "flow_to_colours",
    "overlay_agreement",
    "overlay_instance_agreement",
    "overlay_instances",
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


def _classes_to_colours(mask: np.ndarray, colormap: list[tuple[int, int, int]]) -> np.ndarray:
    """A mask as the picture its colormap says it is, whichever way it arrived.

    Private. R exports this under this name, and the public counterpart here would want a
    `Masks and volumes` group that has no other member - a sidebar section whose description
    is about reading, joining and writing masks, holding one function that does none of
    those. Worth deciding with that group rather than as a by-product of a figure.

    One-hot, probabilities or class indices, all to `(h, w, 3)` uint8. The two ways in used
    to read the same mistake differently - a one-hot mask with more channels than the
    colormap has colours was refused by name, while an index mask was clipped, so classes 2
    and 3 of a four-class mask were drawn in class 1's colour and nothing said so.

    Args:
        mask: `height x width` indices, or `height x width x class` one-hot or probabilities.
        colormap: one colour per class, darkest first.

    Returns:
        The mask in colour, `height x width x 3` uint8.

    Raises:
        MaskError: if the mask holds a class the colormap has no colour for.

    >>> import numpy as np
    >>> indices = np.array([[0, 1], [1, 0]])
    >>> _classes_to_colours(indices, [(0, 0, 0), (255, 255, 255)])[0, 1].tolist()
    [255, 255, 255]
    """
    classes = np.asarray(mask)
    if classes.ndim == 3 and classes.shape[-1] > 1:
        return onehot_to_colours(classes, colormap)
    indices = classes.reshape(classes.shape[:2])
    highest = int(indices.max(initial=0))
    if highest >= len(colormap):
        raise MaskError(
            f"the mask holds class {highest} but the colormap defines {len(colormap)} "
            "classes; drawing it would show one class in another's colour"
        )
    return np.asarray(colormap, dtype=np.uint8)[indices]


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
    coloured = _classes_to_colours(mask, colormap)
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


# The list R's `plot_masks()` takes as its default, under the same reasoning: a segmentation
# figure is usually binary, and making every call carry the same five tokens buys nothing. It
# is safe as a default only because a mask holding a class the colormap has no colour for is
# now refused rather than drawn in a neighbour's colour.
BINARY_COLORMAP: list[tuple[int, int, int]] = [(0, 0, 0), (255, 255, 255)]


def _take_slice(array: np.ndarray | None, where: int | str, *, channels: bool):
    """One plane out of a stack of volumes, along the last spatial axis.

    Which axis that is cannot be guessed from the shape - a stack of volumes with channels and
    a stack of volumes without them differ only in rank - so the caller states it, the same
    way `save_volumes` states its contract per rank rather than inspecting the array.
    """
    if array is None:
        return None
    volume = np.asarray(array)
    axis = -2 if channels else -1
    depth = volume.shape[axis]
    position = depth // 2 if where == "middle" else int(where)
    if not 0 <= position < depth:
        raise PlatypusError(f"slice {where} is outside 0..{depth - 1}")
    return np.take(volume, position, axis=axis)


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
    slice: int | str | None = None,
) -> Figure:
    """Images, their masks and where the two disagree, one row per image.

    The columns are the ones the R `plot_masks()` draws, in the same order and for the same
    reason: the image, the truth, the prediction, and the agreement only when both are
    given, because agreement between a prediction and nothing is not defined.

    Args:
        images: one image or a stack, `[image] x height x width x channels`.
        prediction: one-hot or probabilities per image, or class indices.
        truth: the annotation, same shapes.
        colormap: one colour per class, darkest first. The default is black and white, the
            two-class case, which is also what R defaults to.
        which: which images to draw. The default is the first four, because a montage of
            forty is not a figure anybody reads.
        alpha: how much of the tint shows.
        labels: a row label each. The default numbers them.
        slice: which plane of a volume to draw, or `"middle"`. Required for volumes and
            refused for images, because a volume shown as one picture is either a lie or a
            projection nobody asked for. Volumes arrive as a stack, rank 5.

    Returns:
        A matplotlib `Figure`. Never shown and never saved - the caller decides.

    Raises:
        PlatypusError: if a volume is given without a slice, a slice without a volume, or a
            mask whose classes the colormap has no colours for.

    >>> import numpy as np
    >>> image = np.full((1, 8, 8, 3), 0.5)
    >>> mask = np.zeros((1, 8, 8, 2)); mask[0, 2:5, 2:5, 1] = 1
    >>> figure = plot_masks(image, prediction=mask, truth=mask, colormap=[(0, 0, 0), (255, 0, 0)])
    >>> [axis.get_title() for axis in figure.axes]
    ['image', 'truth', 'prediction', 'agreement']
    """
    if np.asarray(images).ndim == 5:
        if slice is None:
            raise PlatypusError(
                f"these are volumes {np.asarray(images).shape}; choose a slice to draw - "
                '`slice="middle"` to start - because a volume shown as one picture is '
                "either a lie or a projection nobody asked for"
            )
        # A mask carries a channel axis when it is one-hot and not when it holds indices, so
        # the plane sits at a different axis in each. Stated per rank rather than guessed.
        images = _take_slice(images, slice, channels=True)
        if prediction is not None:
            prediction = _take_slice(prediction, slice, channels=np.asarray(prediction).ndim == 5)
        if truth is not None:
            truth = _take_slice(truth, slice, channels=np.asarray(truth).ndim == 5)
    elif slice is not None:
        raise PlatypusError(
            "`slice` applies to volumes, and these are images; a stack of volumes has rank 5"
        )

    stack = _stack(images)
    if which is None:
        which = list(range(min(4, len(stack))))
    if colormap is None:
        colormap = BINARY_COLORMAP

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
            # The mask in its own colours rather than tinted over the image, which is what
            # R's `plot_masks()` draws - and R's figures are the published ones, in a README
            # and three articles. The image is underneath in the agreement panel, where the
            # question is where the boundary falls; here the question is what shape the mask
            # is, and tissue showing through makes two shapes harder to compare.
            "truth": (
                None if truth is None else _classes_to_colours(_stack(truth)[index], colormap)
            ),
            "prediction": (
                None
                if prediction is None
                else _classes_to_colours(_stack(prediction)[index], colormap)
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

    # Sized from the picture's own proportions rather than square. A detection frame is
    # usually wider than it is tall - BCCD is 4:3 - and a square panel leaves the labels
    # fighting each other over a picture that has been squeezed to fit it.
    height, width = _as_picture(stack[which[0]]).shape[:2]
    figure = Figure(
        figsize=(6.4, 6.4 * height / width * len(which)),
        layout="constrained",
    )
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

    if truth is not None:
        # Only with both. Two colours need saying which is which; one colour, where every
        # box is a prediction, would be a legend with one entry restating the caption.
        handles = [
            matplotlib.patches.Patch(edgecolor=chosen[name], facecolor="none", label=name)
            for name in ("prediction", "truth")
        ]
        axes[0][0].legend(handles=handles, loc="upper right", fontsize=8)
    return figure


def _draw_boxes(axis, record: dict[str, Any], colour: str, min_score: float | None) -> None:
    """One record's boxes onto one axis, labelled where there is a score to label with."""
    found = np.asarray(record.get("boxes", []), dtype=float).reshape(-1, 4)
    # `predict` puts the class *indices* in `labels` and the names in `names`, so reading
    # `labels` alone labelled every box of a real prediction with an integer. `names` first,
    # then `labels` for a record somebody built by hand with strings in it.
    names = list(record.get("names", record.get("labels", [])))
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
    anchors: Any,
    boxes: np.ndarray | None = None,
    log: bool = False,
    *,
    model: str | None = None,
    split: str = "train",
) -> Figure:
    """Fitted anchors against the box shapes they were fitted to.

    Two ways in, the same two R has, where `plot_anchors()` takes a fit. Give a fitted
    `DetectionEngine` and it asks the engine for both clouds in one call, colouring the
    boxes by class; or give the anchors and the shapes yourself.

    Handing it the engine is the safer of the two and not merely the shorter: widths from
    one place and anchors from another is how a figure comes to show boxes in different
    places from where the anchors were fitted to them, which looks like a bad fit and is a
    bug. `box_shapes` returns both as fractions of the **letterboxed** input, which is the
    convention the anchors were fitted in - taking the source image instead is what had a
    coverage of 0.67 quoted for two years against a true 0.65.

    Args:
        anchors: a fitted `DetectionEngine`, or one group of `(width, height)` pairs per
            grid, as fractions of the input.
        boxes: the shapes themselves, `n x 2` of widths and heights, drawn underneath.
            Refused together with an engine, which already has them.
        log: both axes on a log scale, off by default as in R. Detection datasets span a
            wide range of sizes - a BCCD platelet is about two fifths the side of a red
            cell - and a log scale spreads the classes a linear one crowds into the corner.
        model: which detector, when an engine is given. The default is the first.
        split: which split's boxes, when an engine is given. Worth drawing for a split the
            anchors were *not* fitted on: it is the only way to see a class the anchors have
            nothing near.

    Returns:
        A matplotlib `Figure`.

    Raises:
        PlatypusError: if both an engine and `boxes` are given.

    >>> figure = plot_anchors([[(0.1, 0.1), (0.2, 0.3)], [(0.4, 0.5)]])
    >>> len(figure.axes[0].collections), figure.axes[0].get_yscale()
    (1, 'linear')
    """
    subtitle = None
    # label, shapes, colour. The colour travels with the cloud rather than being taken from
    # its position later, because a class absent from this split must not shift the colour
    # of every class after it.
    cloud: list[tuple[str, np.ndarray, str]] = []
    if hasattr(anchors, "box_shapes"):
        if boxes is not None:
            raise PlatypusError(
                "an engine carries its own boxes, so giving `boxes` as well leaves two "
                "answers to the same question; drop it, or pass the anchors yourself"
            )
        report = anchors.box_shapes(model, split)
        anchors = report["anchors"]
        shapes = np.column_stack(
            [np.asarray(report["boxes"]["width"]), np.asarray(report["boxes"]["height"])]
        )
        names = np.asarray(report["boxes"]["name"])
        # Keyed on the specification's class order, not on which classes this split happens
        # to contain. Keyed on what is present, a class missing from validation shifts every
        # class after it to another colour - so the same cell is green in one figure and
        # purple in the next, from one model, with nothing to say so.
        known = report["classes"] or list(dict.fromkeys(names.tolist()))
        for index, name in enumerate(known):
            present = shapes[names == name]
            if len(present):
                cloud.append((str(name), present, CLASS_COLOURS[index % len(CLASS_COLOURS)]))
        subtitle = (
            f"{len(shapes)} boxes in '{split}', {sum(len(g) for g in anchors)} anchors "
            + (
                "fitted to the training boxes"
                if report["anchors_were_fitted"]
                else "given in the specification"
            )
            + f", as fractions of a {report['input_shape'][0]} x {report['input_shape'][1]} input"
        )
    elif boxes is not None:
        cloud.append(("boxes", np.asarray(boxes, dtype=float).reshape(-1, 2), "#999999"))

    figure = Figure(figsize=(5.0, 5.0), layout="constrained")
    axis = figure.subplots()
    for label, group, colour in cloud:
        axis.scatter(group[:, 0], group[:, 1], s=6, alpha=0.35, color=colour, label=label)
    # One appearance for every anchor, hollow and black, as R draws them - and not one
    # colour per grid, which was the first version here: the grids took matplotlib's default
    # cycle, so an orange anchor sat invisibly inside an orange class cloud. Legibility
    # against any cloud is worth more than which stride an anchor belongs to, and that is
    # what `anchor_coverage` answers anyway.
    every = np.concatenate([np.asarray(pairs, dtype=float).reshape(-1, 2) for pairs in anchors])
    axis.scatter(
        every[:, 0],
        every[:, 1],
        s=70,
        marker="D",
        facecolors="none",
        edgecolors="black",
        linewidths=0.9,
        label="anchors",
    )
    if log:
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlabel("width, fraction of the input")
    axis.set_ylabel("height, fraction of the input")
    if subtitle is not None:
        axis.set_title(subtitle, fontsize=8)
    axis.legend(loc="upper left", fontsize=8)
    return figure


# --------------------------------------------------------------------------- instances
#
# A classes figure and an instances figure are different problems, and the difference is
# what makes these functions possible without a ninth colour. `CLASS_COLOURS` stops at
# eight because a figure with more than eight **classes** cannot be read by colour however
# it is painted - nobody can hold nine legend entries. But nobody reads object 214 by its
# colour either: what a person needs from an instance map is *where one object ends and the
# next begins*, which is **adjacency, not identity**. So a colour may repeat freely as long
# as it never repeats between neighbours, and eight is then not a ceiling but a surplus.
#
# **Measured over all 670 Data Science Bowl cases, 524 of which hold touching objects and
# 5436 adjacent pairs between them: an exact colouring needs at most 3 colours** - 146
# cases need one, 412 need two, 112 need three - against a maximum of 6 neighbours for any
# single object. So the eight already in `CLASS_COLOURS` are more than twice what the
# hardest of these pictures asks for, and `#214`'s open question answers itself: the exact
# route is affordable, so take it.
#
# The cheap alternative on that issue was a cycling fill "where adjacency collisions are
# made rare rather than impossible". Measured on the same pairs: colouring by object value
# modulo eight paints **550 of 5436 adjacent pairs in one colour, 10.12%**. One touching
# pair in ten drawn as one object is not rare, and in a figure whose whole subject is
# whether two objects are two, it is the one error that matters.


def _adjacent_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Which objects touch, as pairs of label values.

    Eight-connected on purpose, and the opposite of the choice `instances_from_mask` makes.
    That one labels under 4-connectivity so a diagonal join cannot merge two objects into
    one, which errs towards finding more objects - the kind direction for a yardstick. Here
    the risk runs the other way: two objects meeting only at a corner are *visually*
    adjacent, and giving them one colour would draw them as one. So a diagonal counts.
    """
    pairs: set[tuple[int, int]] = set()
    height, width = labels.shape
    for row_shift, column_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        first = labels[
            max(row_shift, 0) : height - max(-row_shift, 0) or None,
            max(column_shift, 0) : width - max(-column_shift, 0) or None,
        ]
        second = labels[
            max(-row_shift, 0) : height - max(row_shift, 0) or None,
            max(-column_shift, 0) : width - max(column_shift, 0) or None,
        ]
        touching = (first > 0) & (second > 0) & (first != second)
        for one, other in zip(first[touching].ravel(), second[touching].ravel(), strict=True):
            pairs.add((min(int(one), int(other)), max(int(one), int(other))))
    return pairs


def _colour_index(labels: np.ndarray, palette_size: int) -> dict[int, int]:
    """One palette slot per object such that no two touching objects share one.

    Greedy in order of descending degree - Welsh-Powell - which is not optimal in general
    and is exact here for a reason the measurement above gives: the graph is planar, so four
    colours always suffice, and these graphs are so sparse (0.37 neighbours per object on
    average) that three were enough for every one of 670 real pictures.

    If a picture ever did need more slots than the palette has, the colouring stops being a
    guarantee, so the slot wraps and that is **said** rather than hidden - `overlay_instances`
    reports it. Silently drawing two neighbours alike is the one failure this is for.
    """
    values = [int(value) for value in np.unique(labels[labels > 0])]
    neighbours: dict[int, set[int]] = {value: set() for value in values}
    for one, other in _adjacent_pairs(labels):
        neighbours[one].add(other)
        neighbours[other].add(one)

    chosen: dict[int, int] = {}
    for value in sorted(values, key=lambda v: len(neighbours[v]), reverse=True):
        taken = {chosen[other] for other in neighbours[value] if other in chosen}
        slot = 0
        while slot in taken:
            slot += 1
        chosen[value] = slot % palette_size if slot >= palette_size else slot
    return chosen


def _outline(where: np.ndarray) -> np.ndarray:
    """The pixels of a region that have a non-region pixel among their four neighbours."""
    inside = np.ones_like(where)
    inside[:-1, :] &= where[1:, :]
    inside[1:, :] &= where[:-1, :]
    inside[:, :-1] &= where[:, 1:]
    inside[:, 1:] &= where[:, :-1]
    return where & ~inside


def overlay_instances(
    image: np.ndarray,
    labels: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    outline: bool = False,
    colours: list[str] | None = None,
) -> np.ndarray:
    """An instance label map over its image, coloured so no two touching objects match.

    The rule is adjacency rather than identity, and it is stated here rather than left to
    luck: a colour repeats across the picture and never between neighbours. See the comment
    above `_adjacent_pairs` for the measurement that made the exact colouring the cheap
    option rather than the expensive one.

    Args:
        image: one image, as `overlay_mask` takes it.
        labels: an instance label map, one integer per object and 0 for background.
        alpha: how much of the tint shows. Ignored when `outline` is true, where a stroke
            is drawn at full strength - a translucent outline is a smudge.
        outline: draw one stroke per object instead of filling it, which is what a
            pathologist is used to and what leaves the tissue entirely visible.
        colours: the palette to take slots from. Defaults to the engine's `CLASS_COLOURS`,
            which the R package reads too.

    Returns:
        A `height x width x 3` array of 0-255 integers.

    Raises:
        PlatypusError: if the shapes do not correspond, or the palette is empty.

    >>> import numpy as np
    >>> image = np.full((5, 9, 3), 0.5)
    >>> labels = np.zeros((5, 9), dtype=int)
    >>> labels[1:4, 1:4] = 1
    >>> labels[1:4, 4:7] = 2      # touching object 1 along a seam
    >>> shown = overlay_instances(image, labels)
    >>> shown.shape
    (5, 9, 3)
    >>> bool((shown[2, 2] != shown[2, 5]).any())   # two touching objects, two colours
    True
    """
    palette = CLASS_COLOURS if colours is None else colours
    if not palette:
        raise PlatypusError("colours is empty; there is nothing to draw the objects in")

    picture = _as_picture(image)
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise PlatypusError(
            f"labels is {labels.shape}; an instance map is one integer per pixel, and a "
            "one-hot array cannot carry an object identity"
        )
    if labels.shape != picture.shape[:2]:
        raise PlatypusError(f"the labels are {labels.shape} and the image is {picture.shape[:2]}")

    for value, slot in _colour_index(labels, len(palette)).items():
        where = labels == value
        _blend(
            picture,
            _rgb(palette[slot]),
            _outline(where) if outline else where,
            1.0 if outline else alpha,
        )
    return np.clip(np.round(picture), 0, 255).astype(np.uint8)


def overlay_instance_agreement(
    image: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.5,
    alpha: float = OVERLAY_ALPHA,
    colours: dict[str, str] | None = None,
) -> np.ndarray:
    """Found, missed and invented **objects**, matched the way the metric matches them.

    `overlay_agreement` answers the same three-way question about pixels, and on an instance
    result that is the wrong unit: a merged pair and a split object are both nearly perfect
    pixelwise and both wrong, which is exactly what an average precision of 0.5587 will not
    tell anybody. This matches whole objects at one IoU threshold, using `iou_matrix` - the
    same function the metric uses - so **the picture and the number cannot disagree**.

    Args:
        image: one image, as `overlay_mask` takes it.
        prediction: an instance label map from a model.
        truth: an instance label map from the annotation.
        threshold: the IoU above which a predicted object counts as a true one, strictly
            greater as in `average_precision`. 0.5 is the loosest of the challenge's ten
            thresholds, so this is the kindest reading, and below 0.5 is refused for the
            reason the metric refuses it.
        alpha: how much of the tint shows.
        colours: keyed `hit`, `missed`, `false_alarm`, as `overlay_agreement` takes them.

    Returns:
        A `height x width x 3` array of 0-255 integers.

    Raises:
        PlatypusError: if the shapes do not correspond, a colour is missing, or the
            threshold is below 0.5.

    >>> import numpy as np
    >>> image = np.full((5, 11, 3), 0.5)
    >>> truth = np.zeros((5, 11), dtype=int)
    >>> truth[1:4, 1:4] = 1
    >>> truth[1:4, 5:8] = 2
    >>> predicted = np.zeros((5, 11), dtype=int)
    >>> predicted[1:4, 1:4] = 1          # one found
    >>> predicted[1:4, 8:11] = 2         # one invented, and object 2 missed
    >>> shown = overlay_instance_agreement(image, predicted, truth)
    >>> hit, missed, invented = shown[2, 2], shown[2, 6], shown[2, 9]
    >>> bool((hit != missed).any() and (missed != invented).any())
    True
    """
    from pyplatypus.instances import iou_matrix

    if threshold < 0.5:
        raise PlatypusError(
            f"threshold {threshold} is below 0.5, where one prediction can satisfy two truth "
            "objects and the match is no longer one to one. `average_precision` refuses the "
            "same thing for the same reason, and a figure drawn under a rule the metric "
            "rejects is a figure that disagrees with the number it illustrates"
        )

    chosen = AGREEMENT_COLOURS if colours is None else colours
    missing = {"hit", "missed", "false_alarm"} - set(chosen)
    if missing:
        raise PlatypusError(
            f"colours is missing {sorted(missing)}; all three are needed, because the "
            "point of this picture is telling the three apart"
        )

    picture = _as_picture(image)
    predicted = np.asarray(prediction)
    actual = np.asarray(truth)
    for name, array in (("prediction", predicted), ("truth", actual)):
        if array.ndim != 2:
            raise PlatypusError(
                f"{name} is {array.shape}; both have to be instance label maps, one integer "
                "per pixel"
            )
    if predicted.shape != actual.shape:
        raise PlatypusError(
            f"the prediction is {predicted.shape} and the truth is {actual.shape}; "
            "agreement between two arrays of different shapes is not defined"
        )
    if predicted.shape != picture.shape[:2]:
        raise PlatypusError(f"the maps are {predicted.shape} and the image is {picture.shape[:2]}")

    overlaps = iou_matrix(actual, predicted)
    matched_truth: set[int] = set()
    matched_prediction: set[int] = set()
    if overlaps.size:
        # Greedy on descending IoU: one prediction per true object and no object counted
        # twice, which is what the challenge metric's own counting does at a threshold.
        order = np.argsort(overlaps, axis=None)[::-1]
        for flat in order:
            row, column = np.unravel_index(flat, overlaps.shape)
            # Strictly greater, which is `average_precision`'s own comparison and not a
            # rounding preference: at exactly 0.5 two predictions each covering half an
            # object would both be admitted, so `>=` would draw a merged pair as one hit and
            # one miss while the metric counts two misses. This line said `<` first and a
            # test caught the difference, which is the whole reason the figure takes its
            # matching from the metric rather than from a reading of it.
            if overlaps[row, column] <= threshold:
                break
            # One prediction per true object. At any threshold this function accepts this
            # **cannot bind** - two disjoint predictions cannot both exceed 0.5 against one
            # object, which is exactly why the metric refuses thresholds below 0.5 - so it is
            # here as the statement of the invariant rather than as working code. Said
            # plainly because a mutation that removes it changes nothing, and a guard nothing
            # can test is a guard that has to explain itself.
            if int(row) + 1 in matched_truth or int(column) + 1 in matched_prediction:
                continue
            matched_truth.add(int(row) + 1)
            matched_prediction.add(int(column) + 1)

    hit = np.isin(actual, sorted(matched_truth)) if matched_truth else np.zeros_like(actual, bool)
    missed = (actual > 0) & ~hit
    invented = (predicted > 0) & ~np.isin(
        predicted, sorted(matched_prediction) if matched_prediction else [-1]
    )
    for part, where in (("hit", hit), ("missed", missed), ("false_alarm", invented)):
        _blend(picture, _rgb(chosen[part]), where, alpha)
    return np.clip(np.round(picture), 0, 255).astype(np.uint8)


def flow_to_colours(flow: np.ndarray) -> np.ndarray:
    """A two-channel vector field as a picture: hue is the direction, value is the length.

    The convention optical flow has used for twenty years, and it is a convention rather
    than a palette decision - which is why it is not in `pyplatypus.style`. Nothing about it
    is a choice this project gets to make differently from everyone else's figures, and a
    reader who has seen one flow visualisation can read this one.

    Without it, debugging a `flows` separation (`#212`) or `hv` (`#213`) means reading arrays.
    A field that is right everywhere except reversed looks obviously reversed here and
    identical in any loss.

    Args:
        flow: `(2, height, width)`, channel 0 the row component and channel 1 the column
            component - the layout `flows_from_instances` returns.

    Returns:
        A `height x width x 3` array of 0-255 integers. A zero vector is black, which is
        also what the background is, and that is honest: the background has no direction.

    Raises:
        PlatypusError: if the array is not a two-channel field.

    >>> import numpy as np
    >>> field = np.zeros((2, 2, 3))
    >>> field[0, 0, 0] = 1.0          # pointing down
    >>> field[1, 0, 1] = 1.0          # pointing right
    >>> shown = flow_to_colours(field)
    >>> shown.shape
    (2, 3, 3)
    >>> bool((shown[0, 0] != shown[0, 1]).any())   # two directions, two hues
    True
    >>> tuple(int(channel) for channel in shown[1, 2])   # no vector, no colour
    (0, 0, 0)
    """
    field = np.asarray(flow, dtype=np.float64)
    if field.ndim != 3 or field.shape[0] != 2:
        raise PlatypusError(
            f"flow is {field.shape}; a vector field here is (2, height, width), channel 0 "
            "the row component and channel 1 the column component"
        )

    length = np.hypot(field[0], field[1])
    longest = length.max()
    hue = (np.arctan2(field[1], field[0]) / (2 * np.pi)) % 1.0
    value = length / longest if longest > 0 else np.zeros_like(length)
    hsv = np.stack([hue, np.ones_like(hue), value], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
