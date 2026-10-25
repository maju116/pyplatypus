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
        class_index: Which channel to measure, numbered from 0 as this package's channels
            are. The default is 1, the foreground of a binary segmentation.

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
