"""clDice, and the soft skeleton it is built on.

The skeleton is checked against `scipy.ndimage`'s own morphology rather than against a
second transcription of the same formula - the arrangement the detection metrics have with
`pycocotools` and the GIoU loss with `torchvision.ops`. scipy is already a dependency, so
the comparison runs rather than being baked in as frozen numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from pyplatypus.objectives import functional as f
from pyplatypus.objectives.metrics import ClDice, build_metrics
from pyplatypus.spec.components import ClDiceMetric


def skeleton_by_openings(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Lantuejoul's skeleton, with scipy's morphology and none of our code.

    `border_value=1` on the erosion because that is what `max_pool`'s -inf padding means
    for the complement: the image border is not eroded from outside.

    **Two different structuring elements, which is the point rather than an oversight.**
    `maju116/pyplatypus#405`: the reference clDice erodes with a cross and dilates with a
    square, so a reference written with one element for both shares the premise it is
    supposed to be checking. This used `np.ones((3,) * ndim)` for both until `0.8.0a72`,
    and it agreed with our code the whole time **because our code had the same premise** -
    an outside implementation is only an outside opinion if it disagrees about something.

    `generate_binary_structure(ndim, 1)` is the cross at any rank: the voxels sharing a
    face, which is one separable min-pool per axis written as a set.
    """
    cross = ndimage.generate_binary_structure(mask.ndim, 1)
    square = np.ones((3,) * mask.ndim, bool)
    eroded = mask.astype(bool)
    skeleton = np.zeros_like(eroded)
    for step in range(iterations + 1):
        opened = ndimage.binary_dilation(
            ndimage.binary_erosion(eroded, cross, border_value=1), square
        )
        skeleton |= eroded & ~opened
        if step < iterations:
            eroded = ndimage.binary_erosion(eroded, cross, border_value=1)
    return skeleton


def blobs(shape, seed, threshold=0.52, sigma=2.2):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random(shape), sigma) > threshold


def onehot(mask: torch.Tensor) -> torch.Tensor:
    """A `(H, W)` 0-1 mask as the `(1, 2, H, W)` one-hot the metrics take."""
    foreground = mask.float()[None, None]
    return torch.cat([1 - foreground, foreground], dim=1)


@pytest.mark.parametrize("seed", range(6))
def test_the_soft_skeleton_is_the_morphological_one(seed):
    """Exactly, not approximately: on a binary mask the pooling and the morphology are the
    same operation, so any difference is a defect rather than a tolerance."""
    mask = blobs((40, 40), seed)
    ours = f.soft_skeleton(torch.tensor(mask[None, None], dtype=torch.float32)).numpy()[0, 0]

    assert mask.sum() > 50, "a fixture with almost nothing in it would prove nothing"
    assert np.array_equal(ours, skeleton_by_openings(mask).astype(np.float32))


def test_the_soft_skeleton_is_rank_generic():
    """2D and 3D take the same path, which is this package's rule 3. A separate 3D
    implementation is what it exists to avoid."""
    volume = blobs((16, 16, 16), 0, sigma=1.8, threshold=0.5)
    ours = f.soft_skeleton(torch.tensor(volume[None, None], dtype=torch.float32)).numpy()[0, 0]

    assert volume.sum() > 50
    assert np.array_equal(ours, skeleton_by_openings(volume).astype(np.float32))


def test_a_one_pixel_line_is_its_own_skeleton():
    line = torch.zeros(1, 1, 9, 9)
    line[0, 0, 4, 1:8] = 1.0
    assert torch.equal(f.soft_skeleton(line), line)


def test_the_skeleton_is_differentiable():
    """The reason for pooling rather than a hard skeleton: this can become a loss."""
    x = torch.rand(1, 1, 12, 12, requires_grad=True)
    f.soft_skeleton(x, iterations=2).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0, "a skeleton no gradient reaches is a hard skeleton"


def test_too_few_iterations_leave_no_skeleton_at_all():
    """Not a partial skeleton - an empty one, which is the reason the metric refuses.

    A solid square's opening is the square, so `x - open(x)` is zero at every level until
    the erosion has worn it below the structuring element. Measured on a 12-pixel square:
    nothing at all until the fifth peel, and the centre from then on. My first guess was
    that the skeleton merely kept a solid middle, and it was wrong in both directions - the
    sum **grows** with iterations, because the skeleton is their union.
    """
    block = torch.zeros(1, 1, 24, 24)
    block[0, 0, 6:18, 6:18] = 1.0

    assert f.soft_skeleton(block, iterations=3).sum() == 0
    assert f.soft_skeleton(block, iterations=5).sum() == 4
    assert f.soft_skeleton(block, iterations=20).sum() == 4


def test_an_empty_skeleton_is_refused_rather_than_scored_a_perfect_one():
    """The silent failure this metric would otherwise have.

    With no centreline the ratios are smooth/smooth and the class scores 1.0. Measured: a
    24-pixel square at `iterations=1` gives clDice **1.0** to a prediction whose Dice is
    0.0017 - a perfect topology score for a model that found nothing.
    """
    truth = torch.zeros(40, 40)
    truth[8:32, 8:32] = 1.0  # far thicker than one peel can reduce
    rubbish = torch.zeros(40, 40)
    rubbish[0:4, 0:4] = 1.0

    p, t = onehot(rubbish), onehot(truth)
    tp, fp, fn = f.overlaps(p, t)
    assert f.dice_from_overlaps(tp, fp, fn, 1.0)[0, 1].item() < 0.01

    raw = f.cldice_from_masks(p, t, iterations=1, smooth=1.0)
    assert torch.isnan(raw[0, 1]), "an undefined score must not look like a perfect one"

    with pytest.raises(ValueError, match="found no centreline"):
        ClDice(iterations=1).reduce(raw)

    # And it does not fire once the peeling reaches the structure.
    assert (
        ClDice(iterations=20).reduce(f.cldice_from_masks(p, t, iterations=20, smooth=1.0)).item()
        < 0.5
    )


def test_the_background_is_left_out_because_its_score_means_nothing():
    """`include_background` defaults False here where the other metrics default True.

    The background's skeleton lies inside the background by construction, so it scores 1.0
    whatever the model did. Measured on a 3-pixel vessel found along 64% of its length.
    """
    truth = torch.zeros(64, 64)
    truth[30:33, 5:60] = 1.0
    found = torch.zeros(64, 64)
    found[30:33, 5:40] = 1.0

    per_class = f.cldice_from_masks(onehot(found), onehot(truth), 5, 1.0)[0]

    assert per_class[0].item() == pytest.approx(1.0), "the background cannot score otherwise"
    assert per_class[1].item() == pytest.approx(0.7865, abs=1e-3)
    assert ClDiceMetric().include_background is False
    assert ClDice().reduce(per_class[None]).item() == pytest.approx(0.7865, abs=1e-3)
    assert ClDice(include_background=True).reduce(per_class[None]).item() == pytest.approx(
        0.8933, abs=1e-3
    )


def test_a_boundary_error_and_a_topology_error_rank_the_other_way_round():
    """The whole argument for the metric, in one comparison.

    A vessel drawn a pixel thin on each side is still one connected vessel; a vessel cut in
    half is two. **Dice prefers the cut one**, because it loses fewer pixels. clDice does
    not. Numbers measured here, not quoted from the paper.
    """
    truth = torch.zeros(40, 120)
    truth[16:25, 10:110] = 1.0  # nine pixels wide

    thin = torch.zeros(40, 120)
    thin[17:24, 10:110] = 1.0  # the same vessel, two pixels narrower
    cut = truth.clone()
    cut[:, 55:65] = 0.0  # the same vessel, severed

    scores = {}
    for name, prediction in (("thin", thin), ("cut", cut)):
        p, t = onehot(prediction), onehot(truth)
        tp, fp, fn = f.overlaps(p, t)
        scores[name] = (
            f.dice_from_overlaps(tp, fp, fn, 0.0)[0, 1].item(),
            f.cldice_from_masks(p, t, 5, 0.0)[0, 1].item(),
        )

    assert scores["thin"] == pytest.approx((0.8750, 1.0000), abs=1e-3)
    assert scores["cut"] == pytest.approx((0.9474, 0.9425), abs=1e-3)
    assert scores["cut"][0] > scores["thin"][0], "Dice prefers the severed vessel"
    assert scores["cut"][1] < scores["thin"][1], "clDice does not"


def test_a_structure_one_pixel_wide_cannot_be_displaced_and_still_score():
    """The limit of the claim, pinned so the documentation does not overstate it.

    "Move it by one pixel and Dice halves while clDice does not" holds from three pixels
    wide. At one pixel a displacement leaves **no overlap at all**, and no overlap-based
    metric - this one included - can report anything but zero.
    """
    measured = {}
    for width in (1, 3, 9):
        low = 20 - width // 2
        truth = torch.zeros(40, 120)
        truth[low : low + width, 10:110] = 1.0
        moved = torch.zeros(40, 120)
        moved[low + 1 : low + 1 + width, 10:110] = 1.0

        p, t = onehot(moved), onehot(truth)
        tp, fp, fn = f.overlaps(p, t)
        measured[width] = (
            f.dice_from_overlaps(tp, fp, fn, 0.0)[0, 1].item(),
            f.cldice_from_masks(p, t, 5, 0.0)[0, 1].item(),
        )

    assert measured[1] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert measured[3] == pytest.approx((0.6667, 1.0), abs=1e-3)
    assert measured[9] == pytest.approx((0.8889, 1.0), abs=1e-3)


def test_an_identical_mask_scores_one():
    mask = onehot(torch.tensor(blobs((32, 32), 3), dtype=torch.float32))
    assert f.cldice_from_masks(mask, mask, 5, 0.0)[0, 1].item() == pytest.approx(1.0)


def test_the_metric_refuses_to_be_rebuilt_from_overlap_counts():
    """`combine` is how a tiled image is scored as one image, and a skeleton has no such
    form. It says so rather than returning something."""
    metric = ClDice()

    assert metric.accumulates is False
    with pytest.raises(TypeError, match="cannot be computed from overlap counts"):
        metric.combine(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))


def test_the_metric_is_built_from_the_specification():
    built = build_metrics([ClDiceMetric(iterations=7, include_background=False)])

    assert isinstance(built["cldice"], ClDice)
    assert built["cldice"].iterations == 7
    assert built["cldice"].include_background is False


# --- which two operators, asserted against scipy rather than against our own arithmetic ---
#
# `maju116/pyplatypus#405`. Everything above compares a skeleton; these two name the pair it
# is built from, because that is the thing the citation is about and the thing that was wrong.


@pytest.mark.parametrize("ndim", [2, 3])
def test_the_pair_is_the_reference_s_pair_a_cross_and_a_square(ndim):
    """A cross to erode and a square to dilate, which is Shit et al.'s own code.

    Asserted against `scipy.ndimage` at both ranks, with **the square erosion as the
    negative control**: without it this test passes on the implementation `#405` found,
    since a square erosion is also exactly some scipy call.
    """
    rng = np.random.default_rng(0)
    mask = rng.random((12,) * ndim) > 0.5
    x = torch.from_numpy(mask.astype(np.float32))[None, None]

    cross = ndimage.generate_binary_structure(ndim, 1)
    square = np.ones((3,) * ndim, bool)

    assert np.array_equal(
        f._erode_cross(x)[0, 0].numpy() > 0.5,
        ndimage.binary_erosion(mask, cross, border_value=1),
    ), "the skeleton's erosion is not a cross"
    assert np.array_equal(
        f._dilate(x)[0, 0].numpy() > 0.5,
        ndimage.binary_dilation(mask, square),
    ), "the dilation is not a square"

    # And the OTHER erosion is still the square, which is a claim worth making rather than
    # an implementation detail: `surface()` reads it and its tolerance band is calibrated
    # against a euclidean distance transform. `#405` changed this one first and 15 of
    # `test_surface_dice.py`'s assertions said so.
    assert np.array_equal(
        f._erode(x)[0, 0].numpy() > 0.5,
        ndimage.binary_erosion(mask, square, border_value=1),
    ), "`_erode` is no longer the square that `surface()` needs"

    assert not np.array_equal(
        ndimage.binary_erosion(mask, cross, border_value=1),
        ndimage.binary_erosion(mask, square, border_value=1),
    ), "a cross and a square erosion agree on this input, so it cannot tell them apart"


def test_a_diagonal_band_is_what_the_two_shapes_disagree_most_about():
    """The case `#405` was found on, with both bounds read off the fixture rather than pinned.

    A three-pixel-wide diagonal is thin, and a square erosion cannot peel it at all, because
    every pixel of it keeps a diagonal neighbour - so the "skeleton" stays almost the whole
    band. The cross peels it to one pixel per row, which is what a centreline of a diagonal
    IS. Measured on this fixture: 190 band pixels, 64 by the cross, 184 by the square.

    The first version of this test asserted `10 <= ours <= 25` from a figure remembered off a
    differently sized band, and failed on correct code at 64 - which is the right answer,
    since the band spans 64 rows. Hence both bounds below in the fixture's own terms.
    """
    side = 64
    band = torch.zeros(1, 1, side, side)
    for i in range(side):
        for offset in (-1, 0, 1):
            if 0 <= i + offset < side:
                band[0, 0, i, i + offset] = 1.0
    thickness = float(band.sum())

    ours = float(f.soft_skeleton(band).sum())
    assert ours == pytest.approx(side), (
        f"a centreline of this band is one pixel per row, so {side}; the cross gave {ours}"
    )

    # The square, as the control that this fixture separates the two at all.
    original = f._erode_cross
    try:
        f._erode_cross = f._erode  # the square, which is what `soft_skeleton` used to use
        square = float(f.soft_skeleton(band).sum())
    finally:
        f._erode_cross = original
    assert square > 0.9 * thickness, (
        f"a square erosion should keep nearly the whole {thickness}-pixel band and kept "
        f"{square}; the fixture does not separate the two shapes"
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_the_skeleton_is_contained_in_the_mask_it_came_from(ndim):
    """The claim the asymmetry is usually objected to with, settled in code.

    An opening whose two operators differ is not anti-extensive, so `dilate(erode(x))` can
    exceed `x` - and the skeleton is the RESIDUAL `relu(x - opened)`, so that can only make
    the residual smaller. Asymmetry loses skeleton; it cannot invent any, and this is what
    says so for a reader who arrives at `_erode`'s docstring unconvinced.
    """
    rng = np.random.default_rng(1)
    mask = ndimage.gaussian_filter(rng.random((24,) * ndim), 2.0) > 0.5
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    skeleton = f.soft_skeleton(x)
    assert float((skeleton - x).max()) <= 0.0, "the skeleton leaves the mask it was built from"
    assert float(skeleton.sum()) > 0.0, "an empty skeleton satisfies the line above for free"
