"""Millilitres, square millimetres and millimetres, out of a mask and the size of a voxel.

A voxel count is not a measurement. The same lesion is 4,000 voxels on one scanner and
12,000 on another, the same fetal skull is 900 pixels of circumference at one depth setting
and 1,400 at the next, and neither number goes in a report. Multiplied by the size of a
voxel they become millilitres and millimetres, which compare between scans, between
patients, and against whatever a clinician measured by hand.

**Every one of these answers existed before this module and none of them lived here.**
`mask_volume` and `instance_sizes` were the R package's, implemented in R; `voxel_ml` was
computed in `inst/python/platypus_shim.py`, which is Python shipped by the *R* package and
importable by no user of this one; and the head circumference was in `examples/`, which the
wheel does not contain. So a `pip install pyplatypus` user asking how big a finding is got
a voxel count and a line of documentation to copy. `maju116/pyplatypus#270`.

The division that line of documentation does is correct, and copying it is the hazard:
`volumes.qmd` divides a voxel count by 1000 and says why that is allowed - at
`target_spacing: [1, 1, 1]` a voxel is exactly one cubic millimetre - three sections before
the same page says resampling is offered rather than imposed. Dropping the resampling leaves
the arithmetic looking identical and the number wrong by the volume of a voxel, with nothing
to say so. A function that takes `spacing` cannot fail that way, which is the whole argument
for one being here.

**Spacing multiplies the coordinates, not the answer.** For a volume or an area the two are
the same thing, because a count scales by the product of the sides either way. For a
*length* they are not: an ellipse's perimeter is not separable in its axes, so there is no
number to multiply a pixel-space circumference by unless the pixels are square. The ellipse
below therefore scales the member coordinates before taking their second moments, which is
right for both cases and reduces to a single multiplication when the two sides are equal.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyplatypus.errors import PlatypusError


class MeasureError(PlatypusError):
    kind = "measure_error"


def _spacing(spacing: Any, rank: int, *, name: str = "spacing") -> tuple[float, ...]:
    """`spacing` as `rank` positive floats in millimetres, or a refusal naming the rank.

    One validation for the whole module, because the mistake it catches is the same every
    time: a spacing of the wrong length does not measure the thing badly, it measures a
    different shape of thing. Two numbers against a volume would report an area.
    """
    try:
        sides = tuple(float(side) for side in np.atleast_1d(np.asarray(spacing, dtype=float)))
    except (TypeError, ValueError):
        raise MeasureError(
            f"`{name}` must be {rank} numbers in millimetres; got {spacing!r}"
        ) from None
    if len(sides) != rank:
        raise MeasureError(
            f"`{name}` must be {rank} numbers in millimetres, one per dimension of the array "
            f"passed; got {len(sides)}. Giving a different count would measure a different "
            f"shape of thing than the one passed."
        )
    if any(not math.isfinite(side) or side <= 0.0 for side in sides):
        raise MeasureError(f"`{name}` must be positive and finite; got {sides}")
    return sides


def _rank(array: np.ndarray, allowed: tuple[int, ...], *, name: str) -> int:
    """The array's rank, refused unless it is one this measurement is defined for."""
    if array.ndim not in allowed:
        wanted = " or ".join(str(number) for number in allowed)
        raise MeasureError(
            f"`{name}` must have {wanted} dimensions; got {array.shape}. For a stack, measure "
            f"one at a time - which is how `predict` returns them."
        )
    return array.ndim


def mask_volume(mask: np.ndarray, spacing: Any, class_index: int = 1) -> float:
    """How much of a volume one class occupies, in millilitres.

    Millilitres rather than cubic millimetres, because a millilitre is a cubic centimetre
    and that is the unit a radiology report is written in.

    Args:
        mask: One volume of class indices, as `Engine.predict` returns after an `argmax` -
            three dimensions, not a stack.
        spacing: Voxel size in millimetres, three numbers, in the array's own axis order.
            `pyplatypus.data.volumes.volume_spacing` reads it out of the file.
        class_index: The class to measure, **matched against the array's own values**
            rather than used as an index into it. For a mask from `argmax` that value is
            the channel number, counted from 0, so the default of 1 is the foreground of a
            binary segmentation. It is what makes the R package's forward trivial: R
            collapses with `which.max` and counts its classes from 1, so R passes its own
            numbering and no translation happens anywhere.

    Returns:
        The volume in millilitres.

    Raises:
        MeasureError: If `mask` is not one volume, or `spacing` is not three positive
            numbers.

    >>> mask = np.zeros((10, 10, 4), dtype=int)
    >>> mask[2:6, 2:6, 1:3] = 1
    >>> round(mask_volume(mask, spacing=(2, 2, 5)), 3)   # 32 voxels of 20 mm^3
    0.64
    """
    mask = np.asarray(mask)
    _rank(mask, (3,), name="mask")
    sides = _spacing(spacing, 3)
    voxels = int(np.count_nonzero(mask == class_index))
    return voxels * float(np.prod(sides)) / 1000.0


def instance_sizes(labels: np.ndarray, spacing: Any = None) -> list[dict[str, Any]]:
    """How big each object is, one row per object.

    The unit is decided by the rank and the key is named after it rather than being called
    `size` with a note somewhere: `area_mm2` for an image, `volume_ml` for a volume. A
    column whose unit lives in the documentation is a column that reaches a paper as a bare
    number. Without `spacing` there is no such key at all, rather than a count pretending to
    be a measurement.

    Args:
        labels: One instance label map, one integer per object and 0 for background. Two
            dimensions for an image, three for a volume.
        spacing: Voxel size in millimetres, one number per dimension. Unset means the rows
            count voxels and say nothing about size.

    Returns:
        One dict per object present, in increasing object number, with `object`, `voxels`
        and - when `spacing` is given - `area_mm2` or `volume_ml`. An object number absent
        from the map gets no row, because a model that found objects 1, 2 and 4 found three
        objects.

    Raises:
        MeasureError: If `labels` is not one map, or `spacing` has the wrong length.

    >>> labels = np.zeros((8, 8), dtype=int)
    >>> labels[1:4, 1:4] = 1
    >>> labels[5:7, 5:8] = 2
    >>> instance_sizes(labels, spacing=(0.5, 0.5))[0]    # 9 pixels of 0.25 mm^2
    {'object': 1, 'voxels': 9, 'area_mm2': 2.25}
    """
    labels = np.asarray(labels)
    rank = _rank(labels, (2, 3), name="labels")
    present, counts = np.unique(labels[labels > 0], return_counts=True)
    rows: list[dict[str, Any]] = [
        {"object": int(value), "voxels": int(count)}
        for value, count in zip(present.tolist(), counts.tolist(), strict=True)
    ]
    if spacing is None:
        return rows

    per_voxel = float(np.prod(_spacing(spacing, rank)))
    for row in rows:
        if rank == 2:
            row["area_mm2"] = row["voxels"] * per_voxel
        else:
            row["volume_ml"] = row["voxels"] * per_voxel / 1000.0
    return rows


def instance_shapes(labels: np.ndarray, spacing: Any = None) -> list[dict[str, Any]]:
    """How stretched each object is, one row per object.

    `instance_sizes` answers how big and this answers what shape, out of the same label map
    and in one pass over it. The semi-axes are the ones `ellipse_axes` computes for a whole
    mask - the second moments of the member coordinates, so nothing is fitted iteratively -
    and the generalisation to a volume is the constant: for a uniform solid the variance
    along an axis is `semi_axis^2 / (rank + 2)`, which is 4 for an image and **5** for a
    volume. Measured on an ellipsoid of semi-axes 50, 30 and 20, reusing the image's 2
    answers 44.74, 26.80 and 17.87 - 10.67% low - against 0.13% for `sqrt(5)`; and the
    control in the other direction, an 80 by 40 ellipse, is 0.17% with 2 and 11.87% with
    `sqrt(5)`.

    **Why this is not `instance_sizes(shape=True)`**: R keeps `instance_sizes` as its own
    arithmetic - a count times a product - and would have to forward a shape, so the one
    function would be half-local and half-forwarded. `rplatypus`'s own test reads the body of
    every function whose help page has a runnable example and asserts it does not reach the
    bridge, because `R CMD check` runs examples and CRAN has no Python. A flag would fail
    that test; a second name joins `ellipse_measure` on its other side.

    Args:
        labels: One instance label map, one integer per object and 0 for background. Two
            dimensions for an image, three for a volume.
        spacing: Voxel size in millimetres, one number per dimension in the array's own axis
            order. Unset reports the sampling grid instead, and the keys say so.

    Returns:
        One dict per object present, in increasing object number, with `object` and the
        semi-axes largest first - `major_mm` and `minor_mm` for an image, with `middle_mm`
        between them for a volume, named `_px` instead when `spacing` was not given. An
        image also gets `eccentricity`, `sqrt(1 - (minor/major)^2)`, which is the number
        shape analysis reports and comes out of the same eigenvalues. A volume does not:
        an ellipsoid has no single eccentricity, and the ratio of two columns is the
        caller's division rather than a name this package invents.

        An object with fewer than `rank + 1` voxels has no second moment and gets zeros,
        the same answer `ellipse_axes` gives for fewer than three pixels.

    Raises:
        MeasureError: If `labels` is not one map, or `spacing` has the wrong length.

    Note:
        **`eccentricity` is unitless and it is not spacing-free**, which is the one trap in
        this row. On anisotropic voxels it describes whichever space it was computed in: a
        circle of 80 pixels on `spacing=(2, 1)` is physically an ellipse, and the grid says
        `0.0000` where the millimetres say `0.8660` - the closed form `sqrt(1 - (1/2)^2)`.
        It reverses too: an 80 by 40 pixel ellipse on `spacing=(1, 2)` is physically almost
        a circle, `0.8667` on the grid against `0.0685` in millimetres. So the `_px` suffix
        on the neighbouring keys is what says which number this is, and a test holds both
        of those figures.

    >>> labels = np.zeros((41, 81), dtype=int)
    >>> rows, columns = np.ogrid[:41, :81]
    >>> labels[((rows - 20) / 10) ** 2 + ((columns - 20) / 18) ** 2 <= 1] = 1
    >>> labels[(rows - 20) ** 2 + (columns - 60) ** 2 <= 9 ** 2] = 2
    >>> rows_out = instance_shapes(labels, spacing=(0.5, 0.5))
    >>> [row["object"] for row in rows_out]
    [1, 2]
    >>> round(rows_out[1]["eccentricity"], 3)       # a circle, whatever the spacing
    0.0
    >>> round(rows_out[0]["major_mm"], 2)           # 18 pixels of 0.5 mm, discretised
    9.03
    """
    labels = np.asarray(labels)
    rank = _rank(labels, (2, 3), name="labels")
    sides = _spacing(spacing, rank) if spacing is not None else (1.0,) * rank
    unit = "mm" if spacing is not None else "px"
    names = ("major", "minor") if rank == 2 else ("major", "middle", "minor")

    # One pass: the foreground coordinates are sorted by label once and each object is a
    # contiguous block, so this is a sort of the mask rather than one full-array scan per
    # object. The three-line loop a caller writes instead - `ellipse_axes(labels == k)` per
    # object - rescans the whole array every time. Measured on 512 x 512 maps of synthetic
    # ellipses: 50.9 ms against 15.2 at 49 objects, 517.0 against 44.9 at 498, and 1988.7
    # against 143.1 at 1939, which is 3.3x, 11.5x and 13.9x. The two agree to 0.00e+00 on
    # every object at every size, and that exactness is what makes them comparable at all -
    # the naive loop is this function's test control for exactly that reason.
    index = np.nonzero(labels)
    if index[0].size == 0:
        return []
    values = labels[index]
    order = np.argsort(values, kind="stable")
    values = values[order]
    coordinates = np.stack([axis[order] * side for axis, side in zip(index, sides, strict=True)])
    coordinates = coordinates.astype(np.float64)

    present, starts, counts = np.unique(values, return_index=True, return_counts=True)
    factor = math.sqrt(rank + 2)
    rows: list[dict[str, Any]] = []
    for value, start, count in zip(present.tolist(), starts.tolist(), counts.tolist(), strict=True):
        row: dict[str, Any] = {"object": int(value)}
        if count < rank + 1:
            axes = (0.0,) * rank
        else:
            block = coordinates[:, start : start + count]
            spread = np.clip(np.linalg.eigvalsh(np.cov(block)), 0.0, None)
            axes = tuple(factor * math.sqrt(value_) for value_ in reversed(spread.tolist()))
        for name, axis in zip(names, axes, strict=True):
            row[f"{name}_{unit}"] = float(axis)
        if rank == 2:
            major, minor = axes
            row["eccentricity"] = (
                math.sqrt(max(0.0, 1.0 - (minor / major) ** 2)) if major > 0.0 else 0.0
            )
        rows.append(row)
    return rows


def largest_filled_object(mask: np.ndarray) -> np.ndarray:
    """The biggest connected piece of a mask, with its holes filled.

    The repair a shape measurement needs before it can mean anything, and it is applied to
    the truth as well as to the prediction so that both sides of a comparison went through
    the same arithmetic. A prediction is not an ellipse: it can arrive in two pieces or with
    a hole in the middle, and the second moments of either are not the second moments of the
    object - a hole pushes the variance outwards and a stray island drags the centre.

    Args:
        mask: Anything with a foreground; read as a boolean through `> 0`.

    Returns:
        A boolean array of the same shape, with one connected region.

    >>> mask = np.zeros((9, 9), dtype=bool)
    >>> mask[1:8, 1:8] = True
    >>> mask[4, 4] = False       # a hole
    >>> mask[0, 8] = True        # and a stray pixel
    >>> int(largest_filled_object(mask).sum())
    49
    """
    from scipy.ndimage import binary_fill_holes, label

    foreground = np.asarray(mask) > 0
    pieces, count = label(foreground)
    if count > 1:
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        foreground = pieces == int(np.argmax(sizes))
    return np.asarray(binary_fill_holes(foreground), dtype=bool)


def ellipse_axes(mask: np.ndarray, spacing: Any = None) -> tuple[float, float]:
    """The semi-axes of the ellipse a region implies, major first.

    For a uniform filled ellipse the covariance of the member coordinates has eigenvalues
    `a^2 / 4` and `b^2 / 4`, so the axes come out of an eigendecomposition of a 2x2 matrix
    and nothing is fitted iteratively.

    It measures the region it is given and repairs nothing, which matters for a prediction:
    pass it through `largest_filled_object` first, or use `ellipse_circumference`, which
    does.

    Args:
        mask: One image; read as a boolean through `> 0`.
        spacing: Pixel size in millimetres, two numbers in the array's own axis order -
            `(row, column)`. Unset returns pixels.

    Returns:
        `(major, minor)` semi-axes, in millimetres when `spacing` was given and in pixels
        otherwise. `(0.0, 0.0)` for fewer than three foreground pixels, which has no
        second moment.

    Raises:
        MeasureError: If `mask` is not one image, or `spacing` is not two positive numbers.

    >>> mask = np.zeros((41, 41), dtype=bool)
    >>> rows, columns = np.ogrid[:41, :41]
    >>> mask[((rows - 20) / 10) ** 2 + ((columns - 20) / 20) ** 2 <= 1] = True
    >>> tuple(round(axis) for axis in ellipse_axes(mask))
    (20, 10)
    """
    mask = np.asarray(mask) > 0
    _rank(mask, (2,), name="mask")
    sides = _spacing(spacing, 2) if spacing is not None else (1.0, 1.0)

    rows, columns = np.nonzero(mask)
    if rows.size < 3:
        return 0.0, 0.0
    coordinates = np.stack([columns * sides[1], rows * sides[0]]).astype(np.float64)
    spread = np.clip(np.linalg.eigvalsh(np.cov(coordinates)), 0.0, None)
    return 2.0 * math.sqrt(spread[1]), 2.0 * math.sqrt(spread[0])


def _ramanujan(a: float, b: float) -> float:
    """An ellipse's circumference by Ramanujan's second approximation.

    There is no closed form. This one is right to about `1e-7` relative at the
    eccentricities a fetal skull has - four orders of magnitude below one pixel - where
    `pi * (a + b)`, the thing reached for instead, is wrong by 1.5% at an axis ratio of 0.7.
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def ellipse_circumference(mask: np.ndarray, spacing: Any = None) -> float:
    """The circumference of the ellipse a region implies.

    What a sonographer writes down: a fetal head circumference is the perimeter of the
    ellipse fitted to the skull, in millimetres, and no overlap score stands in for it -
    two masks of the same Dice can be 8 times apart in millimetres, depending on whether
    they are wrong in a ring or wrong in a ragged edge.

    Takes the largest filled component first, because a prediction in two pieces has second
    moments that belong to neither.

    Args:
        mask: One image; read as a boolean through `> 0`.
        spacing: Pixel size in millimetres, two numbers in the array's own axis order.
            Unset returns pixels. Measure on the grid the spacing was recorded for: a
            prediction resized to a model's input shape has pixels of a size nobody wrote
            down, so `Engine.predict(space="source")` is what this wants.

    Returns:
        The circumference, in millimetres when `spacing` was given and in pixels otherwise.

    Raises:
        MeasureError: If `mask` is not one image, or `spacing` is not two positive numbers.

    >>> mask = np.zeros((41, 41), dtype=bool)
    >>> rows, columns = np.ogrid[:41, :41]
    >>> mask[((rows - 20) / 10) ** 2 + ((columns - 20) / 20) ** 2 <= 1] = True
    >>> round(ellipse_circumference(mask, spacing=(0.5, 0.5)))    # half-millimetre pixels
    49
    """
    axes = ellipse_axes(largest_filled_object(mask), spacing)
    return _ramanujan(*axes)
