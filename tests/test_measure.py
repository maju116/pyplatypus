"""Measurements in physical units, and the controls that say which quantity came back.

Most of this file moved here from `test_examples.py` with `maju116/pyplatypus#270`: the
arithmetic it tests used to live in `examples/measure_fetal_head_circumference.py`, which
the wheel does not ship, so the engine's own suite had nothing to say about the one number
that example exists to produce. Moving the code moved its tests, which is the honest version
of a port - a second copy held together by a comparison would have been the thing §3F of
the workbench's notes is a list of.

**Every test here names the wrong quantity it is rejecting**, because the mutation batch on
the original code taught that lesson expensively: eight mutations of the circumference were
caught and replacing it with the mask's *area* was not, since every assertion compared two
outputs of the same function. An absolute number cannot be fooled that way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyplatypus.measure import (
    MeasureError,
    ellipse_axes,
    ellipse_circumference,
    instance_sizes,
    largest_filled_object,
    mask_volume,
)
from tests.conftest import filled_ellipse


def _integrated_perimeter(a: float, b: float, points: int = 200_001) -> float:
    """The arc length of the parametric ellipse, by trapezoid.

    By hand rather than through `np.trapezoid`, which numpy only grew in 2.0: this suite
    runs against whatever the resolver picks on four Python versions.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, points)
    speed = np.sqrt((a * np.sin(angle)) ** 2 + (b * np.cos(angle)) ** 2)
    return float((0.5 * (speed[:-1] + speed[1:]) * np.diff(angle)).sum())


def test_a_volume_is_millilitres_and_says_so_in_the_arithmetic():
    """The unit is the point: a millilitre is a cubic centimetre, so the divisor is 1000.

    Worked by hand rather than against another call: 32 voxels of 2 x 2 x 5 mm are 640 cubic
    millimetres and 0.64 ml. Dropping the division answers 640, and dropping the spacing
    answers 0.032 - both are what this test is for, and both are named below.
    """
    mask = np.zeros((10, 10, 4), dtype=int)
    mask[2:6, 2:6, 1:3] = 1
    assert mask.sum() == 32

    got = mask_volume(mask, spacing=(2, 2, 5))
    assert got == pytest.approx(0.64), got
    assert abs(got - 640.0) > 1.0, "that is cubic millimetres, not millilitres"
    assert abs(got - 0.032) > 0.1, "that is a voxel count divided by 1000, with no spacing"


def test_the_class_is_matched_against_the_values_and_is_not_an_index():
    """Which is why there is no off-by-one between the two packages, measured rather than
    reasoned about.

    `pyplatypus#270` was filed saying the forward would own a translation, because R counts
    its classes from 1 and this package counts channels from 0. It owns none: the argument
    is compared with `==` against whatever is in the array, so R hands over its own mask and
    its own class number and both sides agree. Checked on R's own convention - a `which.max`
    map holding 1 and 2 - where `class_index=2` is the foreground, and 0 selects nothing at
    all because no such value is present.
    """
    engine_style = np.zeros((4, 4, 4), dtype=int)
    engine_style[0, 0, 0] = 1
    engine_style[1, 1, 1] = 2
    assert mask_volume(engine_style, spacing=(1, 1, 1)) == pytest.approx(0.001)
    assert mask_volume(engine_style, spacing=(1, 1, 1), class_index=2) == pytest.approx(0.001)
    assert mask_volume(engine_style, spacing=(1, 1, 1), class_index=0) == pytest.approx(0.062)
    assert mask_volume(engine_style, spacing=(1, 1, 1), class_index=7) == 0.0

    # R's convention, which is where this matters: a `which.max` map has no zero in it, so
    # an argument read as a 0-based index would select the background and report 0.
    r_style = np.ones((10, 10, 4), dtype=int)
    r_style[2:6, 2:6, 1:3] = 2
    assert int((r_style == 2).sum()) == 32
    assert mask_volume(r_style, spacing=(2, 2, 5), class_index=2) == pytest.approx(0.64)
    assert mask_volume(r_style, spacing=(2, 2, 5), class_index=0) == 0.0


def test_the_size_of_an_object_is_named_after_the_unit_its_rank_implies():
    """`area_mm2` for an image and `volume_ml` for a volume, and neither without spacing.

    A key whose unit lives in the documentation is a key that reaches a paper as a bare
    number, so there is no `size`. And an object number absent from the map gets no row: a
    model that found objects 1, 2 and 4 found three objects, not four.
    """
    labels = np.zeros((8, 8), dtype=int)
    labels[1:4, 1:4] = 1
    labels[5:7, 5:8] = 4

    bare = instance_sizes(labels)
    assert bare == [{"object": 1, "voxels": 9}, {"object": 4, "voxels": 6}], bare
    assert all("area_mm2" not in row for row in bare), "a count must not pretend to be a size"

    sized = instance_sizes(labels, spacing=(0.5, 0.5))
    assert [row["area_mm2"] for row in sized] == pytest.approx([2.25, 1.5])

    volume = np.zeros((4, 4, 4), dtype=int)
    volume[0:2, 0:2, 0:2] = 3
    rows = instance_sizes(volume, spacing=(1, 1, 10))
    assert rows == [{"object": 3, "voxels": 8, "volume_ml": pytest.approx(0.08)}], rows
    assert "area_mm2" not in rows[0], "a volume is not an area"


def test_the_moment_fit_returns_the_axes_of_an_ellipse_it_was_handed():
    """Exact for a true ellipse, which is the claim everything above rests on.

    Measured on a 120 x 90 ellipse rotated 0.3 rad: pixel counting is the only error, and it
    is under half a percent. A fit that dropped the factor of two, or returned the variances
    rather than their roots, misses by a factor rather than by a fraction.
    """
    major, minor = ellipse_axes(filled_ellipse((500, 600), (250, 300), 120.0, 90.0, angle=0.3))

    assert major > minor, "the major axis is returned first"
    assert abs(major - 120.0) / 120.0 < 0.005, major
    assert abs(minor - 90.0) / 90.0 < 0.005, minor

    # A circle, where the two must agree with each other as well as with the radius: an
    # ellipse test alone passes for a fit that swaps the axes under rotation.
    major, minor = ellipse_axes(filled_ellipse((400, 400), (200, 200), 80.0, 80.0))
    assert abs(major - minor) < 0.5 and abs(major - 80.0) / 80.0 < 0.005, (major, minor)


def test_the_accuracy_figures_in_the_docstring_are_the_measured_ones():
    """§3C: a number in prose needs a test holding it to the measurement, or no number.

    `_ramanujan`'s docstring carried two figures and both were wrong in opposite directions
    (`pyplatypus#328`) - 1.5% quoted at the wrong axis ratio, and `1e-7` understating the
    approximation by five orders of magnitude. The exact perimeter is `4a E(e^2)` from
    `scipy.special.ellipe`, which shares no code with `_ramanujan`.

    The control is the circle: both formulas are exact there, so a test that passed on a
    circle alone would say nothing about either.
    """
    from scipy.special import ellipe

    from pyplatypus.measure import _ramanujan

    def exact(a: float, b: float) -> float:
        return 4.0 * a * float(ellipe(1.0 - (b / a) ** 2))

    assert _ramanujan(1.0, 1.0) == pytest.approx(exact(1.0, 1.0), rel=1e-15), "the circle"
    assert math.pi * 2.0 == pytest.approx(exact(1.0, 1.0), rel=1e-15), "the circle, naively"

    claimed = {0.75: (0.0051, 8e-14), 0.7: (0.0077, 7e-13), 0.6: (0.0154, 2.4e-11)}
    for ratio, (naive_claim, ramanujan_claim) in claimed.items():
        truth = exact(1.0, ratio)
        naive = abs(math.pi * (1.0 + ratio) - truth) / truth
        ours = abs(_ramanujan(1.0, ratio) - truth) / truth
        assert naive == pytest.approx(naive_claim, rel=0.02), (ratio, naive)
        assert ours == pytest.approx(ramanujan_claim, rel=0.2), (ratio, ours)

    # And the docstring says those numbers, so a corrected figure here fails over there too.
    prose = _ramanujan.__doc__
    for said in ("8e-14", "7e-13", "0.51%", "0.77%", "1.5%", "0.75", "0.7", "0.6"):
        assert said in prose, said


def test_ramanujan_matches_an_integrated_perimeter_and_the_naive_formula_does_not():
    """The approximation, against the arc length of the parametric ellipse.

    Both halves are the test. Without the second, a circumference of `pi * (a + b)` - which
    is what somebody writes when they remember the circle - passes at every axis ratio a
    test like this would casually use, because it is exact for a circle and wrong by only
    0.51% at 0.75, where a fetal skull sits. The `naive_worst > 0.01` below is met by the
    150x75 case, a ratio of 0.5 at 2.7%, and this docstring named 0.7 and 1.5% until
    `pyplatypus#328` - neither of which is a row of the loop.
    """
    worst, naive_worst = 0.0, 0.0
    for a, b in ((100.0, 100.0), (120.0, 90.0), (120.0, 84.0), (150.0, 75.0)):
        exact = _integrated_perimeter(a, b, points=2_000_001)
        mask = filled_ellipse((600, 700), (300, 350), a, b)
        # On the shape rather than on the two floats, because the private helper is not the
        # surface: a circumference is what a caller gets, and the fit is inside it.
        measured = ellipse_circumference(mask)
        worst = max(worst, abs(measured - exact) / exact)
        naive_worst = max(naive_worst, abs(np.pi * (a + b) - exact) / exact)

    assert worst < 0.005, f"the measured circumference is off by {worst:.3e} relative"
    assert naive_worst > 0.01, (
        f"the naive formula is only {naive_worst:.3e} off on these axes, so this test would "
        "pass for it too and proves nothing"
    )


def test_the_largest_piece_is_taken_and_a_hole_in_it_is_closed():
    """A prediction is not an ellipse, and both repairs change the answer.

    A hole pushes the second moments outwards and an island drags the centre, so each is
    asserted against the clean ellipse's own axes rather than against a tolerance nobody
    measured.
    """
    clean = filled_ellipse((500, 600), (250, 300), 120.0, 90.0)
    expected = ellipse_axes(clean)

    holed = clean & ~filled_ellipse((500, 600), (250, 300), 40.0, 30.0)
    island = clean | filled_ellipse((500, 600), (60, 60), 25.0, 25.0)

    for broken in (holed, island):
        got = ellipse_axes(largest_filled_object(broken))
        assert abs(got[0] - expected[0]) < 0.5 and abs(got[1] - expected[1]) < 0.5, got
    # And the control: without the repair the same masks give a different answer, or this
    # test is asserting that two numbers are equal for reasons of its own.
    assert abs(ellipse_axes(holed)[0] - expected[0]) > 1.0
    assert abs(ellipse_axes(island)[0] - expected[0]) > 1.0


def test_the_circumference_repairs_before_it_measures_and_the_axes_do_not():
    """The two surfaces differ on purpose, so the difference is asserted rather than assumed.

    `ellipse_circumference` takes the largest filled component first because a prediction in
    two pieces has second moments belonging to neither; `ellipse_axes` measures what it was
    handed. A reader who swaps them gets a different number, and if that stopped being true
    the docstrings would be wrong with nothing to say so.
    """
    clean = filled_ellipse((500, 600), (250, 300), 120.0, 90.0)
    island = clean | filled_ellipse((500, 600), (60, 60), 25.0, 25.0)

    assert ellipse_circumference(island) == pytest.approx(ellipse_circumference(clean), rel=1e-3)
    raw = ellipse_axes(island)
    assert abs(raw[0] - ellipse_axes(clean)[0]) > 1.0, "the axes are supposed to repair nothing"


def test_spacing_multiplies_the_coordinates_and_not_the_answer():
    """The decision this module was written around, and the one a scalar cannot express.

    A circle of radius 100 pixels on pixels twice as tall as they are wide *is* an ellipse of
    semi-axes 200 and 100, and its perimeter is not the circle's perimeter times any single
    number. Measured: 968.9 mm here against 968.8 integrated, where the two scalings
    somebody would reach for answer 628.3 and 1256.7. An implementation that multiplied the
    answer has no factor available to it that lands within 2% of the truth, which is why
    both of those are named rather than only the one that is too small.
    """
    circle = filled_ellipse((401, 401), (200, 200), 100.0, 100.0)
    in_pixels = ellipse_circumference(circle)
    assert in_pixels == pytest.approx(2.0 * np.pi * 100.0, rel=0.005)

    got = ellipse_circumference(circle, spacing=(2.0, 1.0))
    exact = _integrated_perimeter(200.0, 100.0)
    assert got == pytest.approx(exact, rel=0.01), f"{got:.2f} against {exact:.2f} integrated"
    assert abs(got - in_pixels * 1.0) > 0.02 * exact, "that is the smaller side's scaling"
    assert abs(got - in_pixels * 2.0) > 0.02 * exact, "that is the larger side's scaling"

    # The control, and the reason the example's own composition stayed correct for HC18:
    # when the two sides are equal there is nothing to separate, and scaling coordinates and
    # scaling the answer have to agree to the last digit rather than merely closely.
    for side in (0.0612, 0.2, 1.7):
        assert ellipse_circumference(circle, spacing=(side, side)) == pytest.approx(
            in_pixels * side, rel=1e-12
        ), side

    # **A circle cannot tell `(row, column)` from `(column, row)`**, and this test was
    # written with one until R's mutation batch said so: reversing the pair gives the same
    # ellipse with its axes exchanged, so 968.88 either way. An already-elongated mask
    # separates them - 968.79 against 1637.79, 1.7 times apart - which is the one shape that
    # pins which number goes with which axis. The wide ellipse below is 200 columns by 50
    # rows, so the taller pixels have to land on the *short* side.
    oval = filled_ellipse((301, 501), (150, 250), 200.0, 50.0)
    assert ellipse_circumference(oval, spacing=(2.0, 1.0)) == pytest.approx(
        _integrated_perimeter(200.0, 100.0), rel=0.01
    )
    assert ellipse_circumference(oval, spacing=(1.0, 2.0)) == pytest.approx(
        _integrated_perimeter(400.0, 50.0), rel=0.01
    )


def test_a_ring_of_pixels_is_invisible_to_dice_and_large_in_millimetres():
    """The claim the HC18 example exists to make, on a shape worked out by hand.

    A 120 x 90 ellipse dilated by two pixels keeps a Dice above 0.97 - a number anybody
    would publish - while its circumference grows by about `2 * pi * 2` pixels, which at this
    dataset's coarsest pixel size is millimetres rather than a rounding difference. The
    ragged arm is the same statement from the other side: the boundary moves everywhere and
    the length does not.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    def dice(prediction, truth):
        total = float(prediction.sum() + truth.sum())
        return 1.0 if total == 0.0 else 2.0 * float(np.logical_and(prediction, truth).sum()) / total

    truth = filled_ellipse((500, 600), (250, 300), 120.0, 90.0)
    pixel = (0.2, 0.2)

    grown = binary_dilation(truth, iterations=2)
    overlap = dice(grown, truth)
    shift = ellipse_circumference(grown, pixel) - ellipse_circumference(truth, pixel)
    assert overlap > 0.97, overlap
    # Pinned to the geometry rather than to "large": a radius bigger by r makes a
    # circumference bigger by `2 * pi * r`, which is 2.51 mm here, and the measured 2.26 is
    # 0.90 of it because `binary_dilation`'s default element is a 4-connected cross - two
    # iterations reach 2 along the axes and 1.41 on the diagonal, so the diamond is smaller
    # than the disc its radius names. Both ends of the bound are therefore measured: the
    # shift sits in [0.8, 1.0] of the disc's prediction, and the two wrong quantities this
    # test exists to reject are a hundred times outside it - the mask's **area** answers
    # 240 mm, and a circumference left in pixels answers 11. A loose `shift > 2.0` passed
    # for the area, which is how this came to be written this way.
    expected = 2.0 * np.pi * 2.0 * pixel[0]
    assert 0.8 * expected < shift <= expected, (
        f"{shift:.3f} mm where a ring of two pixels predicts at most {expected:.3f}"
    )

    rng = np.random.default_rng(0)
    band = binary_dilation(truth, iterations=2) ^ binary_erosion(truth, iterations=2)
    ragged = truth ^ (band & (rng.random(truth.shape) < 0.5))
    rough = dice(ragged, truth)
    wobble = abs(ellipse_circumference(ragged, pixel) - ellipse_circumference(truth, pixel))
    assert rough < overlap, (
        f"the ragged mask scores {rough:.4f} and the dilated one {overlap:.4f}; the whole "
        "point is that Dice ranks them the other way round from the millimetres"
    )
    assert wobble < shift / 2.0, (
        f"roughing the boundary moved the length by {wobble:.3f} mm against the dilation's "
        f"{shift:.3f}; the fit is supposed to integrate the ripple away"
    )


def test_the_millimetres_are_a_length_and_not_some_other_quantity():
    """The headline number, against the arc length of the ellipse it was given.

    This is the test the mutation batch was missing: an area is out by a factor of fifty
    here and a circumference left in pixels by a factor of five, and neither is detectable
    by comparing two outputs of the same function.
    """
    a, b, pixel = 120.0, 90.0, 0.2
    mask = filled_ellipse((500, 600), (250, 300), a, b)

    exact = _integrated_perimeter(a, b) * pixel
    got = ellipse_circumference(mask, (pixel, pixel))
    assert abs(got - exact) / exact < 0.01, f"{got:.3f} mm against {exact:.3f} integrated"
    # Named so the failure says which wrong quantity was returned.
    assert abs(got - a * b * np.pi * pixel) > 1.0, "that is the area"
    assert abs(got - exact / pixel) > 1.0, "that is the length in pixels"


def test_too_few_pixels_have_no_second_moment_and_are_not_an_error():
    """Zero rather than a refusal, because an empty prediction is an ordinary outcome.

    A model that found nothing has measured nothing, and raising there would make a loop
    over a split fail on its worst case instead of reporting it.
    """
    empty = np.zeros((20, 20), dtype=bool)
    assert ellipse_axes(empty) == (0.0, 0.0)
    assert ellipse_circumference(empty, (0.5, 0.5)) == 0.0

    two = empty.copy()
    two[5, 5] = two[5, 6] = True
    assert ellipse_axes(two) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: mask_volume(np.zeros((4, 4)), (1, 1)), "3 dimensions"),
        (lambda: mask_volume(np.zeros((4, 4, 4)), (1, 1)), "3 numbers"),
        (lambda: mask_volume(np.zeros((4, 4, 4)), (1, 1, 0)), "positive"),
        (lambda: mask_volume(np.zeros((4, 4, 4)), (1, 1, np.nan)), "positive"),
        (lambda: instance_sizes(np.zeros((2, 2, 2, 2))), "2 or 3 dimensions"),
        (lambda: instance_sizes(np.zeros((4, 4)), (1, 1, 1)), "2 numbers"),
        (lambda: ellipse_axes(np.zeros((4, 4, 4))), "2 dimensions"),
        (lambda: ellipse_circumference(np.zeros((4, 4)), (1, 1, 1)), "2 numbers"),
    ],
)
def test_a_spacing_of_the_wrong_shape_is_refused_by_name(call, match):
    """Because the wrong length does not measure badly, it measures a different thing.

    Two numbers against a volume would report an area and three against an image a volume,
    and both would be plausible. Every one of these is a `MeasureError`, which R turns into
    `platypus_measure_error` by the rule it applies to every kind the engine has.
    """
    with pytest.raises(MeasureError, match=match):
        call()
