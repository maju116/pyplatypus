"""Per-object shape out of one label map, and the controls that say it is the right shape.

`maju116/pyplatypus#272`. `0.8.0a39` fitted an ellipse to a *whole* mask, which is HC18's
shape and not the general one, and the route a reader would reach for - `ellipse_axes` once
per object - rescans the whole array every time.

**The naive loop is this file's control rather than its competitor.** It is the thing a
caller writes, so agreement with it is what makes `instance_shapes` a replacement instead of
a different answer that happens to be faster, and the agreement asserted is *exact*: two
implementations that are merely close cannot be told from two that disagree about a case.

**And one test here carries a negative control instead of a positive one** - the constant
that generalises the fit to a volume. `sqrt(rank + 2)` is 2 for an image and `sqrt(5)` for a
volume, and reusing the image's 2 is not an error that changes a decision anywhere: the
ordering of the axes, the eccentricity and every ratio survive it, so it would have sat there
until somebody reported a length. §4's "an error that changes no decision is an error nobody
looks for", which is why the wrong constant is asserted to be wrong by a stated amount.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyplatypus.measure import MeasureError, ellipse_axes, instance_shapes
from tests.conftest import filled_ellipse


def _naive_axes(mask: np.ndarray, sides: tuple[float, ...]) -> tuple[float, ...]:
    """The three-line loop's body, written out: second moments of one object's coordinates.

    Deliberately not a call to `ellipse_axes`, so that this control is an independent
    implementation for the 3D case the shipped function refuses, and so that a mutation to
    the shipped code cannot silently mutate the control with it.
    """
    index = np.nonzero(mask)
    if index[0].size < len(index) + 1:
        return (0.0,) * len(index)
    coordinates = np.stack([axis * side for axis, side in zip(index, sides, strict=True)])
    spread = np.clip(np.linalg.eigvalsh(np.cov(coordinates.astype(np.float64))), 0.0, None)
    factor = math.sqrt(len(index) + 2)
    return tuple(factor * math.sqrt(value) for value in reversed(spread.tolist()))


def _scattered_objects(shape=(160, 160), count=12, seed=3) -> np.ndarray:
    """A label map of ellipses of different shapes and sizes, none of them round.

    Round objects are the fixture that cannot tell a transposed axis order from a correct
    one - which is how the whole-mask fit's own `spacing` test passed while being unable to
    distinguish `(row, column)` from `(column, row)`.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=np.int32)
    side = math.ceil(math.sqrt(count))
    step = min(shape) / side
    number = 0
    for i in range(side):
        for j in range(side):
            if number >= count:
                break
            number += 1
            a = step * rng.uniform(0.18, 0.46)
            b = a * rng.uniform(0.25, 0.9)
            piece = filled_ellipse(
                shape,
                ((i + 0.5) * step, (j + 0.5) * step),
                a,
                b,
                angle=rng.uniform(0.0, math.pi),
            )
            labels[piece] = number
    return labels


def test_one_pass_agrees_with_the_loop_a_caller_would_write_exactly():
    """Not "close": the same numbers, so the fast one is a replacement and not a variant."""
    labels = _scattered_objects()
    present = np.unique(labels[labels > 0]).tolist()
    assert len(present) == 12, "the fixture lost objects, so the comparison is about nothing"

    for spacing, sides in ((None, (1.0, 1.0)), ((0.7, 0.3), (0.7, 0.3))):
        rows = instance_shapes(labels, spacing=spacing)
        unit = "mm" if spacing is not None else "px"
        assert [row["object"] for row in rows] == present

        worst = 0.0
        for row in rows:
            major, minor = _naive_axes(labels == row["object"], sides)
            worst = max(
                worst,
                abs(row[f"major_{unit}"] - major),
                abs(row[f"minor_{unit}"] - minor),
            )
        assert worst == 0.0, f"spacing={spacing}: worst disagreement {worst:.3e}"

        # The control, because two implementations agreeing on zero is not agreement: the
        # objects measured are different from one another and none of them is degenerate.
        majors = [row[f"major_{unit}"] for row in rows]
        assert min(majors) > 0.0
        assert len({round(value, 9) for value in majors}) == len(majors)
        assert max(row["eccentricity"] for row in rows) > 0.5, "every object came out round"


def test_a_volume_needs_sqrt_five_and_the_image_constant_is_wrong_by_a_tenth():
    """For a uniform solid the variance along an axis is `semi^2 / (rank + 2)`.

    The negative control is the point: the image's 2 does not fail loudly on a volume, it
    returns axes that are 10.67% low with their order, their ratios and the eccentricity all
    intact.
    """
    size = 121
    centre = (size - 1) / 2
    truth = (50.0, 30.0, 20.0)
    z, y, x = np.ogrid[:size, :size, :size]
    ellipsoid = (
        ((z - centre) / truth[0]) ** 2
        + ((y - centre) / truth[1]) ** 2
        + ((x - centre) / truth[2]) ** 2
    ) <= 1.0
    labels = np.where(ellipsoid, 1, 0).astype(np.int32)

    row = instance_shapes(labels, spacing=(1.0, 1.0, 1.0))[0]
    assert set(row) == {"object", "major_mm", "middle_mm", "minor_mm"}, (
        "a volume has three axes and no single eccentricity"
    )
    got = (row["major_mm"], row["middle_mm"], row["minor_mm"])
    for value, expected in zip(got, truth, strict=True):
        assert abs(value - expected) / expected < 0.01, f"{got} against {truth}"

    # The negative control: the image's constant, applied to the same object.
    wrong = tuple(value * 2.0 / math.sqrt(5.0) for value in got)
    shortfall = [
        (expected - value) / expected for value, expected in zip(wrong, truth, strict=True)
    ]
    assert all(0.10 < value < 0.12 for value in shortfall), shortfall
    # ...and it survives every check that is not a length, which is why it needs a test.
    assert wrong[0] > wrong[1] > wrong[2]
    assert abs(wrong[1] / wrong[0] - got[1] / got[0]) < 1e-12


def test_eccentricity_is_unitless_and_is_not_spacing_free():
    """A circle on anisotropic voxels is physically an ellipse, and the grid cannot say so.

    Both directions, because one of them alone reads as a rounding difference.
    """
    size = 201
    centre = (size - 1) / 2
    rows, columns = np.ogrid[:size, :size]
    circle = np.where((rows - centre) ** 2 + (columns - centre) ** 2 <= 80.0**2, 1, 0)

    grid = instance_shapes(circle)[0]
    assert set(grid) == {"object", "major_px", "minor_px", "eccentricity"}
    assert grid["eccentricity"] == pytest.approx(0.0, abs=1e-12)

    stretched = instance_shapes(circle, spacing=(2.0, 1.0))[0]
    assert stretched["eccentricity"] == pytest.approx(math.sqrt(1.0 - 0.25), abs=5e-4)
    assert stretched["major_mm"] == pytest.approx(2.0 * grid["major_px"], rel=1e-9)

    # The other direction: a grid ellipse that the spacing turns back into a circle. Without
    # this, the assertion above is satisfied by anything that merely scales with `spacing`.
    ellipse = np.where(
        ((rows - centre) / 80.0) ** 2 + ((columns - centre) / 40.0) ** 2 <= 1.0, 1, 0
    )
    assert instance_shapes(ellipse)[0]["eccentricity"] == pytest.approx(0.8667, abs=5e-4)
    undone = instance_shapes(ellipse, spacing=(1.0, 2.0))[0]
    assert undone["eccentricity"] == pytest.approx(0.0685, abs=5e-3)


def test_the_axes_are_the_whole_mask_fit_when_there_is_one_object():
    """The generalisation has to reduce to the function it generalises, or it is a second
    implementation of the same rule - which is what §3F of the workbench's notes is a list
    of. One object, so `ellipse_axes` is asking the same question."""
    shape = (121, 181)
    labels = np.where(filled_ellipse(shape, (60, 90), 70.0, 25.0, angle=0.6), 1, 0)

    for spacing in (None, (0.4, 0.9)):
        row = instance_shapes(labels, spacing=spacing)[0]
        unit = "mm" if spacing is not None else "px"
        major, minor = ellipse_axes(labels, spacing=spacing)
        assert row[f"major_{unit}"] == pytest.approx(major, rel=1e-12)
        assert row[f"minor_{unit}"] == pytest.approx(minor, rel=1e-12)

    # The control: the two spacings give different numbers, so the agreement above is not
    # two functions both ignoring their argument.
    assert (
        instance_shapes(labels)[0]["major_px"] != instance_shapes(labels, (0.4, 0.9))[0]["major_mm"]
    )


def test_an_object_too_small_for_a_second_moment_is_zeros_and_not_an_error():
    """`rank + 1` points, because a 2x2 covariance needs three and a 3x3 needs four. The
    same answer `ellipse_axes` gives, rather than a refusal that would make one speck of
    noise fail a whole table."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[1, 1] = 1
    labels[3, 3:5] = 2
    labels[6, 6:9] = 3
    labels[10:16, 10:16] = 4

    rows = {row["object"]: row for row in instance_shapes(labels, spacing=(1.0, 1.0))}
    assert sorted(rows) == [1, 2, 3, 4]
    for number in (1, 2):
        assert rows[number]["major_mm"] == 0.0
        assert rows[number]["minor_mm"] == 0.0
        assert rows[number]["eccentricity"] == 0.0

    # The control: three collinear points DO have a second moment, and the object that is
    # genuinely measurable is measured - without which "everything is zero" would pass.
    assert rows[3]["major_mm"] > 0.0
    assert rows[3]["minor_mm"] == pytest.approx(0.0, abs=1e-12)
    assert rows[3]["eccentricity"] == pytest.approx(1.0, abs=1e-9)
    assert rows[4]["minor_mm"] > 0.0


def test_the_degenerate_threshold_is_the_rank_and_not_the_number_three():
    """A 2x2 covariance needs three points and a 3x3 needs four, so the threshold is
    `rank + 1`.

    This test exists because the mutation batch found its absence: `rank + 1` and a literal
    `3` are the same number for an image and differ for a volume, so every test here passed
    with the wrong threshold. §4's fixture that never contained the case, in the one
    direction the 2D cases cannot reach - and a three-voxel object in a volume is not exotic,
    it is what one speck of a prediction looks like.
    """
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[2, 2, 2] = 1
    labels[4, 4, 4:6] = 2
    labels[6, 6, 6:9] = 3  # three voxels: a 3x3 covariance cannot be formed
    labels[10:16, 10:16, 10:16] = 4

    rows = {row["object"]: row for row in instance_shapes(labels, spacing=(1.0, 1.0, 1.0))}
    assert sorted(rows) == [1, 2, 3, 4]
    for number in (1, 2, 3):
        assert (rows[number]["major_mm"], rows[number]["middle_mm"], rows[number]["minor_mm"]) == (
            0.0,
            0.0,
            0.0,
        ), f"object {number} claimed a second moment it cannot have"

    # The control: the cube IS measurable, so "all zeros" cannot pass this test, and its
    # axes are the three of a volume rather than two.
    assert rows[4]["minor_mm"] > 0.0
    assert set(rows[4]) == {"object", "major_mm", "middle_mm", "minor_mm"}


def test_an_absent_object_number_gets_no_row_and_an_empty_map_no_rows():
    """The same rule `instance_sizes` follows: a model that found 1, 2 and 4 found three
    objects, and inventing a row for 3 would put a zero-sized object in a table."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[2:9, 2:9] = 1
    labels[12:19, 12:19] = 2
    labels[22:29, 22:29] = 4

    assert [row["object"] for row in instance_shapes(labels)] == [1, 2, 4]
    assert instance_shapes(np.zeros((8, 8), dtype=int)) == []

    # The control: the numbering is read off the values and is not a position, so a map
    # whose only object is 7 reports 7.
    only = np.zeros((20, 20), dtype=np.int32)
    only[4:12, 4:12] = 7
    assert [row["object"] for row in instance_shapes(only)] == [7]


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: instance_shapes(np.zeros((2, 2, 2, 2))), "2 or 3 dimensions"),
        (lambda: instance_shapes(np.zeros((4,))), "2 or 3 dimensions"),
        (lambda: instance_shapes(np.zeros((4, 4)), (1, 1, 1)), "2 numbers"),
        (lambda: instance_shapes(np.zeros((4, 4, 4)), (1, 1)), "3 numbers"),
        (lambda: instance_shapes(np.zeros((4, 4)), (1, 0)), "positive"),
        (lambda: instance_shapes(np.zeros((4, 4)), (1, np.nan)), "positive"),
    ],
)
def test_a_spacing_of_the_wrong_shape_is_refused_by_name(call, match):
    """The wrong length does not measure badly, it measures a different thing - and for a
    shape it is worse than for a size, because a plausible number comes back either way."""
    with pytest.raises(MeasureError, match=match):
        call()
