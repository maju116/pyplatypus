"""Every loss, exercised the way training will use it."""

import pytest
import torch

from pyplatypus.objectives import build_loss
from pyplatypus.objectives import functional as f
from pyplatypus.spec.components import (
    CceDiceLoss,
    CceLoss,
    ComboLoss,
    DiceLoss,
    FocalLoss,
    FocalTverskyLoss,
    IouLoss,
    LovaszLoss,
    TverskyLoss,
)

ALL = [
    IouLoss(),
    DiceLoss(),
    CceLoss(),
    CceDiceLoss(),
    FocalLoss(),
    TverskyLoss(),
    FocalTverskyLoss(),
    ComboLoss(),
    LovaszLoss(),
]
IDS = [type(s).__name__ for s in ALL]


def example(rank=2, size=8, n_class=2, seed=0):
    torch.manual_seed(seed)
    spatial = (size,) * rank
    target = f.as_onehot(torch.randint(0, n_class, (2, *spatial)), n_class)
    logits = torch.randn(2, n_class, *spatial, requires_grad=True)
    return logits, target


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_every_loss_returns_a_finite_scalar(spec):
    logits, target = example()
    value = build_loss(spec)(logits, target)
    assert value.ndim == 0
    assert torch.isfinite(value)


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_every_loss_produces_a_gradient(spec):
    logits, target = example()
    build_loss(spec)(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_every_loss_works_on_volumes(spec):
    """PLAN.md rule 3: reduce over all dims except batch and channel, and 3D is free."""
    logits, target = example(rank=3, size=6)
    value = build_loss(spec)(logits, target)
    assert value.ndim == 0 and torch.isfinite(value)


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_a_better_prediction_scores_lower(spec):
    """The property that makes a loss a loss. Catches a flipped sign, which nothing else
    here would notice - training would still run and still converge on nonsense."""
    torch.manual_seed(3)
    target = f.as_onehot(torch.randint(0, 2, (2, 16, 16)), 2)
    good = (target - 0.5) * 8.0
    bad = (target - 0.5) * -8.0
    loss = build_loss(spec)
    assert loss(good, target) < loss(bad, target)


@pytest.mark.parametrize(
    "spec",
    [IouLoss(), DiceLoss(), TverskyLoss(), LovaszLoss()],
    ids=["iou", "dice", "tversky", "lovasz"],
)
def test_overlap_losses_vanish_on_a_perfect_prediction(spec):
    target = f.as_onehot(torch.randint(0, 2, (2, 16, 16)), 2)
    logits = (target - 0.5) * 40
    assert build_loss(spec)(logits, target).item() == pytest.approx(0.0, abs=2e-3)


def test_cce_dice_sits_between_its_two_halves():
    logits, target = example(size=16)
    blended = build_loss(CceDiceLoss(cce_weight=0.5))(logits, target)
    cce = build_loss(CceLoss())(logits, target)
    dice = build_loss(DiceLoss())(logits, target)
    assert min(cce, dice) <= blended <= max(cce, dice)


def test_cce_weight_of_one_is_plain_cross_entropy():
    logits, target = example(size=16)
    assert build_loss(CceDiceLoss(cce_weight=1.0))(logits, target).item() == pytest.approx(
        build_loss(CceLoss())(logits, target).item(), rel=1e-5
    )


def test_focal_tversky_with_gamma_one_is_tversky_loss():
    logits, target = example(size=16)
    assert build_loss(FocalTverskyLoss(gamma=1.0))(logits, target).item() == pytest.approx(
        build_loss(TverskyLoss())(logits, target).item(), rel=1e-5
    )


def test_multiclass_losses_run():
    logits, target = example(n_class=5, size=12)
    for spec in ALL:
        assert torch.isfinite(build_loss(spec)(logits, target))


def test_shape_is_what_carries_the_rank_not_a_flag():
    """A 2D batch and the volume holding the same numbers must score identically, because
    nothing in the reduction cares how the space is arranged."""
    torch.manual_seed(7)
    flat_logits = torch.randn(2, 2, 64)
    flat_target = f.as_onehot(torch.randint(0, 2, (2, 64)), 2)
    cube_logits = flat_logits.reshape(2, 2, 4, 4, 4)
    cube_target = flat_target.reshape(2, 2, 4, 4, 4)
    for spec in ALL:
        loss = build_loss(spec)
        assert loss(flat_logits, flat_target).item() == pytest.approx(
            loss(cube_logits, cube_target).item(), rel=1e-5
        )


# The four identities the new component descriptions state. Each one says "this option at
# this value *is* another loss", which is the most useful thing a description can say about
# a knob - and the thing most likely to be quietly wrong after a refactor, because every
# test above would still pass. §3C: a number in prose needs a test holding it there.


def _probe():
    torch.manual_seed(11)
    logits = torch.randn(2, 3, 16, 16)
    target = f.as_onehot(torch.randint(0, 3, (2, 16, 16)), 3)
    return logits, target


def test_tversky_at_alpha_half_is_dice_at_twice_the_smoothing():
    """What `tversky`'s description warns about: leaving `alpha` at its default does not
    give Tversky, it gives Dice with a different floor. Multiplying the Tversky formula
    through by two is where the doubling comes from."""
    logits, target = _probe()
    probs = f.probabilities(logits)
    tversky = f.tversky_coefficient(probs, target, alpha=0.5, smooth=1.0)
    assert torch.allclose(tversky, f.dice_coefficient(probs, target, smooth=2.0), atol=1e-6)
    # and not Dice at the same smoothing, or the warning would be pointless
    assert not torch.allclose(tversky, f.dice_coefficient(probs, target, smooth=1.0), atol=1e-6)


def test_iou_is_a_monotone_function_of_dice_so_the_two_rank_alike():
    """`iou`'s description claims the same ordering and a lower number. Without smoothing
    the coefficient is exactly dice / (2 - dice), which is strictly increasing on [0, 1]."""
    logits, target = _probe()
    probs = f.probabilities(logits)
    dice = f.dice_coefficient(probs, target, smooth=0.0)
    iou = f.iou_coefficient(probs, target, smooth=0.0)
    assert torch.allclose(iou, dice / (2 - dice), atol=1e-6)

    torch.manual_seed(12)
    candidates = [torch.randn(1, 3, 16, 16) for _ in range(12)]
    by_dice = sorted(
        range(12),
        key=lambda i: f.dice_coefficient(f.probabilities(candidates[i]), target[:1], 1.0)
        .mean()
        .item(),
    )
    by_iou = sorted(
        range(12),
        key=lambda i: f.iou_coefficient(f.probabilities(candidates[i]), target[:1], 1.0)
        .mean()
        .item(),
    )
    assert by_dice == by_iou


def test_focal_at_gamma_zero_is_cross_entropy():
    """`focal`'s description says gamma is the whole feature. If this ever stopped holding,
    the loss would have grown a second behaviour nobody documented."""
    logits, target = _probe()
    assert f.focal(logits, target, gamma=0.0).item() == pytest.approx(
        f.cross_entropy(logits, target).item(), rel=1e-6
    )
    assert f.focal(logits, target, gamma=2.0).item() != pytest.approx(
        f.cross_entropy(logits, target).item(), rel=1e-6
    )


def test_focal_tversky_at_gamma_one_is_tversky():
    """Because mean(1 - TI) is 1 - mean(TI). Stated in `focal_tversky`'s description, and
    the same shape of claim as focal's."""
    logits, target = _probe()
    tversky = build_loss(TverskyLoss(alpha=0.5, smooth=1.0))
    focal_tversky = build_loss(FocalTverskyLoss(alpha=0.5, gamma=1.0, smooth=1.0))
    assert focal_tversky(logits, target).item() == pytest.approx(
        tversky(logits, target).item(), rel=1e-6
    )
    harsher = build_loss(FocalTverskyLoss(alpha=0.5, gamma=2.0, smooth=1.0))
    assert harsher(logits, target).item() != pytest.approx(tversky(logits, target).item(), rel=1e-6)
