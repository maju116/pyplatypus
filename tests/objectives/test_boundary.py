"""The boundary loss: a term that knows *how far* a wrong voxel is from the truth.

Dice counts a voxel the same wherever it sits, which is why a model can reach 0.88 on it
while its volumes run a fifth too large - the overshoot is all at the boundary, and for a
small lesion the boundary is most of the object. These tests are about that asymmetry
rather than about the arithmetic.
"""

import numpy as np
import pytest
import torch

from pyplatypus.data.masks import MaskError, signed_distance
from pyplatypus.objectives.losses import build_loss, loss_needs_distance
from pyplatypus.spec.components import BoundaryLoss, DiceLoss, FocalLoss


def square(shape=(16, 16), side=6):
    """A one-hot mask with a centred square as class 1."""
    out = np.zeros((*shape, 2), np.float32)
    out[..., 0] = 1
    lo = [(s - side) // 2 for s in shape]
    sl = tuple(slice(l, l + side) for l in lo)
    out[(*sl, 1)] = 1
    out[(*sl, 0)] = 0
    return out


def ball(shape, radius):
    centre = tuple(s // 2 for s in shape)
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    inside = sum((g - c) ** 2 for g, c in zip(grids, centre)) <= radius**2
    out = np.zeros((*shape, 2), np.float32)
    out[..., 1] = inside
    out[..., 0] = ~inside
    return out


# --- the transform -----------------------------------------------------------------------


def test_the_distance_is_negative_inside_and_positive_outside():
    distance = signed_distance(square())
    assert distance[8, 8, 1] < 0  # the middle of the square
    assert distance[0, 0, 1] > 0  # a corner, far outside it
    assert distance.shape == (16, 16, 2)


def test_the_magnitude_is_the_distance_to_the_boundary_in_voxels():
    """A 6-wide square centred in 16: its middle is 3 rows from the edge."""
    distance = signed_distance(square(side=6))
    assert distance[8, 8, 1] == pytest.approx(-3.0)
    # One row outside the square's top edge is one away from it.
    assert distance[4, 8, 1] == pytest.approx(1.0)


def test_a_class_that_is_absent_has_no_boundary_and_is_left_at_zero():
    """Any other filling would be a number the loss then acts on, and there is nothing to
    be near or far from."""
    mask = np.zeros((8, 8, 3), np.float32)
    mask[..., 0] = 1  # class 1 and 2 never appear
    distance = signed_distance(mask)
    assert np.all(distance[..., 1] == 0)
    assert np.all(distance[..., 2] == 0)


def test_spacing_turns_voxels_into_millimetres():
    """The difference between a loss that means the same thing on two scanners and one
    that does not: the same anatomy at 1 mm and at 2.5 mm slices is the same distance in
    millimetres and a different one in voxels."""
    mask = ball((16, 16, 16), 4)
    voxels = signed_distance(mask)
    millimetres = signed_distance(mask, spacing=(1.0, 1.0, 2.5))
    # Along the third axis the spacing is 2.5, so a step out of the ball costs 2.5 not 1.
    assert millimetres[8, 8, 13, 1] == pytest.approx(voxels[8, 8, 13, 1] * 2.5, rel=0.2)
    assert not np.allclose(voxels, millimetres)


def test_a_mask_without_a_class_axis_is_refused():
    with pytest.raises(MaskError, match="one-hot mask"):
        signed_distance(np.zeros(8, np.float32))


# --- the property the loss exists for ------------------------------------------------------


def test_at_equal_overlap_the_surface_term_prefers_under_to_over_segmentation():
    """The asymmetry Dice does not see, and the reason this loss is here.

    Two predictions of near-equal Dice, one well over the truth and one well under it.
    Dice ranks them almost identically and if anything prefers the larger; the surface term
    prefers the smaller. Asserted as a direction, not a magnitude - the margin is 3% of the
    term's own scale and claiming more than its sign would be claiming more than was
    measured.
    """
    truth = ball((48, 48, 24), 8.0)
    over = ball((48, 48, 24), 9.0)
    under = ball((48, 48, 24), 7.1)

    def dice(pred):
        p, t = pred[..., 1], truth[..., 1]
        return (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)

    def surface(pred):
        return float((signed_distance(truth) * pred).mean())

    assert abs(dice(over) - dice(under)) < 0.05, "the two must be comparable on Dice"
    assert (over[..., 1].sum() - truth[..., 1].sum()) > 0
    assert (under[..., 1].sum() - truth[..., 1].sum()) < 0
    assert surface(under) < surface(over)


# --- the loss ------------------------------------------------------------------------------


def _tensors(mask):
    """Channels-first, as the trainer hands them over."""
    t = torch.from_numpy(np.moveaxis(mask, -1, 0))[None]
    return t, torch.from_numpy(np.moveaxis(signed_distance(mask), -1, 0))[None]


def test_the_loss_is_the_weighted_sum_of_its_two_terms():
    mask = square()
    target, distance = _tensors(mask)
    logits = torch.randn(1, 2, 16, 16)

    region = build_loss(DiceLoss())
    whole = build_loss(BoundaryLoss(region=DiceLoss(), alpha=0.25))

    from pyplatypus.objectives import functional as f

    surface = (f.probabilities(logits) * distance).mean()
    expected = 0.25 * region(logits, target) + 0.75 * surface
    assert float(whole(logits, target, distance)) == pytest.approx(float(expected))


def test_the_loss_refuses_to_run_without_the_map_rather_than_inventing_one():
    target, _ = _tensors(square())
    loss = build_loss(BoundaryLoss())
    with pytest.raises(ValueError, match="signed distance map"):
        loss(torch.randn(1, 2, 16, 16), target, None)


def test_only_the_boundary_loss_asks_for_a_distance_map():
    assert loss_needs_distance(BoundaryLoss()) is True
    assert loss_needs_distance(DiceLoss()) is False
    assert loss_needs_distance(FocalLoss()) is False
    assert build_loss(BoundaryLoss()).needs_distance is True
    assert build_loss(DiceLoss()).needs_distance is False


def test_focal_is_a_region_term_like_any_other():
    """What the issue asked for, spelled the way the spec spells everything else."""
    loss = build_loss(BoundaryLoss(region=FocalLoss(gamma=3.0), alpha=0.6))
    target, distance = _tensors(square())
    value = loss(torch.randn(1, 2, 16, 16), target, distance)
    assert torch.isfinite(value)
    assert type(loss.region).__name__ == "Focal"


def test_alpha_excludes_both_ends():
    """At 1 the surface term does nothing and the name is a lie; at 0 nothing measures how
    much of the object was found, and the term alone is minimised by a confident prediction
    deep inside a shrunken one."""
    from pydantic import ValidationError

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValidationError):
            BoundaryLoss(alpha=bad)


def test_a_boundary_loss_cannot_be_its_own_region_term():
    """Two surface terms with no region term holding either down. Refused by the type
    rather than by a check, since `region` is the union without this one in it."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BoundaryLoss(region={"name": "boundary"})


def test_the_default_alpha_is_the_measured_one_and_not_the_obvious_one():
    """0.5 looks like the natural default and was measured to be unusable: three seeds gave
    a volume error spread of ±28.28% against the baseline's ±3.11%, and every number worse.
    Pinned so that a later tidy-up cannot quietly restore the symmetrical-looking value."""
    assert BoundaryLoss().alpha == 0.9
