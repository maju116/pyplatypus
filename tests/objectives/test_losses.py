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


def test_iou_is_exactly_a_function_of_dice_per_class():
    """The identity `iou == dice / (2 - dice)`, which holds for ONE coefficient.

    Asserted elementwise, which is the only scope on which it is true: the per-class vector
    before any reduction. `IouMetric`'s docstring names this test, so renaming one means
    renaming both.
    """
    logits, target = _probe()
    probs = f.probabilities(logits)
    dice = f.dice_coefficient(probs, target, smooth=0.0)
    iou = f.iou_coefficient(probs, target, smooth=0.0)
    assert torch.allclose(iou, dice / (2 - dice), atol=1e-6)


def test_dice_and_iou_can_rank_one_report_differently():
    """And the claim that stood here instead, which was false. `pyplatypus#360`.

    This test used to sort twelve random tensors by each metric's `.mean()` and assert the
    orders agreed. They did, and that is all it measured: `d / (2 - d)` is convex, so the
    mean of the IoUs is not that function of the mean Dice, and two models whose per-class
    scores are spread differently can swap places. Twelve draws happened not to cross.

    Asserted as the property rather than as a warning, so that a future change which really
    did make the two agree - a different reduction, say - fails here and gets read rather
    than being taken as a silent improvement. The fixture is a strip with a large class and
    a small one, because the spread between classes is the whole mechanism.
    """
    strip = [0] * 30 + [1] * 20 + [2] * 10
    target = f.as_onehot(torch.tensor(strip).reshape(1, 1, -1), 3)

    def logits_for(rows):
        value = torch.full((1, 3, 1, len(rows)), -10.0)
        for position, klass in enumerate(rows):
            value[0, klass, 0, position] = 10.0
        return value

    def reported(rows):
        probs = f.probabilities(logits_for(rows))
        # `[:, 1:]`, not `[1:]`: the coefficient comes back as (batch, class), so slicing
        # the first axis drops the batch and leaves an empty tensor whose mean is NaN.
        # Background is dropped because `include_background=False` is what a report uses.
        dice = f.dice_coefficient(probs, target, smooth=0.0)[:, 1:].mean().item()
        iou = f.iou_coefficient(probs, target, smooth=0.0)[:, 1:].mean().item()
        return dice, iou

    # A finds the small class and misses the large one; B splits its effort.
    a_dice, a_iou = reported([0] * 30 + [0] * 20 + [2] * 6 + [0] * 4)
    b_dice, b_iou = reported([0] * 30 + [1] * 2 + [0] * 18 + [2] * 4 + [0] * 6)

    assert (a_dice - b_dice) * (a_iou - b_iou) < 0, (
        "dice and iou agreed on this pair, so either the reduction changed or the fixture "
        f"stopped separating them: dice {a_dice:.4f} vs {b_dice:.4f}, iou {a_iou:.4f} vs {b_iou:.4f}"
    )

    # The control: the identity still holds per class on the same prediction, so the
    # disagreement is the reduction and not a broken coefficient.
    probs = f.probabilities(logits_for([0] * 30 + [0] * 20 + [2] * 6 + [0] * 4))
    dice = f.dice_coefficient(probs, target, smooth=0.0)
    iou = f.iou_coefficient(probs, target, smooth=0.0)
    assert torch.allclose(iou, dice / (2 - dice), atol=1e-6)


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


# `pyplatypus#429`. Both tests below this one are named after `gamma = 1`, which is the
# single value where mean-of-powers equals power-of-mean - so neither can see where the power
# is taken, and the one that does run `gamma = 2` only asserts that the answer DIFFERS from
# Tversky's, which any wrong implementation that is not the identity satisfies.
#
# What closes it is a value from outside. The closed form is the mean over (sample, class) of
# `(1 - TI)^gamma` with `TI = (tp + s) / (tp + alpha*fn + (1-alpha)*fp + s)`, and
# `_FOCAL_TVERSKY_BY_HAND` computes exactly that in plain Python - no torch, no helper out of
# `functional`, the softmax written out - so agreement is two implementations of one formula
# rather than the formula compared with itself.
#
# The fixture is integer logits over a 2x2 frame, two samples and three classes, with a label
# map where every class is present somewhere and none is present everywhere: a class absent
# from a sample makes `tp` and `fn` zero and drops whole terms of the formula out unexercised.
_FT_LOGITS = [
    [[[2.0, -1.0], [0.0, 1.0]], [[-1.0, 3.0], [1.0, -2.0]], [[0.0, 0.0], [2.0, 1.0]]],
    [[[-2.0, 1.0], [1.0, 0.0]], [[3.0, -1.0], [-1.0, 2.0]], [[1.0, 2.0], [0.0, -1.0]]],
]
_FT_LABELS = [[[0, 1], [2, 0]], [[1, 2], [0, 1]]]


def _focal_tversky_fixture():
    logits = torch.tensor(_FT_LOGITS, dtype=torch.float32)
    target = torch.zeros_like(logits)
    for sample, labels in enumerate(_FT_LABELS):
        for row in range(2):
            for column in range(2):
                target[sample, labels[row][column], row, column] = 1.0
    return logits, target


def _focal_tversky_by_hand(alpha: float, gamma: float, smooth: float) -> float:
    """The paper's formula, in plain Python, with nothing imported from the package."""
    import math

    terms = []
    for sample, labels in zip(_FT_LOGITS, _FT_LABELS):
        n_class = len(sample)
        for c in range(n_class):
            tp = fp = fn = 0.0
            for row in range(2):
                for column in range(2):
                    column_logits = [sample[k][row][column] for k in range(n_class)]
                    biggest = max(column_logits)
                    exps = [math.exp(v - biggest) for v in column_logits]
                    probability = exps[c] / sum(exps)
                    truth = 1.0 if labels[row][column] == c else 0.0
                    tp += probability * truth
                    fp += probability * (1.0 - truth)
                    fn += (1.0 - probability) * truth
            index = (tp + smooth) / (tp + alpha * fn + (1.0 - alpha) * fp + smooth)
            terms.append((1.0 - index) ** gamma)
    return sum(terms) / len(terms)


# Both ends measured, which is what a bound costs. The implementation agrees with the hand
# value to 1.2e-09 at gamma 1.5 and to 7.4e-09 across the four gammas - float32 noise over
# sums of twelve terms. The four mutations this test exists for move it by:
#
#     the power taken AFTER the mean          0.0578573768 -> 0.0551081531   2.75e-03
#     alpha and beta swapped                               -> 0.0594312200   1.57e-03
#     gamma ignored, treated as 1                          -> 0.1448140070   8.70e-02
#     alpha ignored, treated as 0.5                        -> 0.0586104880   7.53e-04
#
# So `abs=1e-6` sits 830 times above the noise and 750 times below the smallest of the four.
# A bound with only one of those measured is a guess that happens to be green.
_FOCAL_TVERSKY_TOLERANCE = 1e-6


@pytest.mark.parametrize("gamma", [1.0, 1.5, 2.0, 3.0])
def test_focal_tversky_equals_the_PAPERS_formula_computed_HERE_at_any_gamma(gamma):
    """The identity case is in the list first, and on purpose.

    `gamma = 1` is where every existing test lives, so running it here is the control that
    the hand implementation is right about the case the suite already agrees on - and the
    three above it are the cases nothing could see. A parametrisation whose first case fails
    says the instrument is wrong rather than the subject.
    """
    logits, target = _focal_tversky_fixture()
    loss = build_loss(FocalTverskyLoss(alpha=0.6, gamma=gamma, smooth=1.0))
    assert loss(logits, target).item() == pytest.approx(
        _focal_tversky_by_hand(0.6, gamma, 1.0), abs=_FOCAL_TVERSKY_TOLERANCE
    )


def test_focal_tversky_is_the_mean_of_POWERS_and_not_the_power_of_the_mean():
    """The one thing `gamma = 1` makes invisible, asked as its own question.

    Jensen: the power is convex above 1, so taking it per class before averaging is not the
    same as averaging and then taking it - and the two are equal exactly at `gamma = 1`,
    which is the value both older tests here are named after. Measured on this fixture at
    gamma 1.5: 0.0578573768 against 0.0551081531, which is 2.75e-03 apart and 2.3 million
    times the float32 noise.
    """
    logits, target = _focal_tversky_fixture()
    loss = build_loss(FocalTverskyLoss(alpha=0.6, gamma=1.5, smooth=1.0))
    mean_of_powers = _focal_tversky_by_hand(0.6, 1.5, 1.0)
    power_of_mean = _focal_tversky_by_hand(0.6, 1.0, 1.0) ** 1.5

    assert loss(logits, target).item() == pytest.approx(
        mean_of_powers, abs=_FOCAL_TVERSKY_TOLERANCE
    )
    # And the wrong answer is named rather than merely excluded, so that a future reader can
    # see which of the two this package computes without re-deriving Jensen.
    assert mean_of_powers > power_of_mean
    assert loss(logits, target).item() != pytest.approx(power_of_mean, abs=1e-4)


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


def test_focal_tverskys_gamma_says_it_is_NOT_the_papers_gamma():
    """`pyplatypus#429`. The field is the reciprocal of Abraham and Khan's and the names match.

    Their exponent is `1/gamma` with gamma in [1, 3]; this field IS the exponent. So their
    recommended 4/3 is this field's 0.75, and a reader who copies 4/3 straight out of the
    paper gets a sharper exponent where the paper meant a flatter one - in the direction that
    makes the loss concentrate harder rather than less.

    **That has already happened once in this project.** The eleven-loss table of
    `pyplatypus#203` ran `focal_tversky` at `alpha 0.7, gamma 1.33` - the paper's own number -
    and `JOURNAL.md` records it as the worst row, concluding that *the focal exponent costs
    more than the asymmetry it is layered on*. True of exponent 1.33, which is what ran; not
    a statement about the paper's setting, which is exponent 0.75.

    The author's decision on 2026-11-08 was to keep this field's meaning and state the
    conversion, because changing what a released field means would alter a stored
    configuration **silently** - both readings are valid positive floats, so nothing could
    refuse the old one. This test is what keeps the statement there, and it asks for the
    three pieces a reader needs rather than for the wording: who, that it is not the same
    parameter, and which way to convert.
    """
    from pyplatypus.spec.components import FocalTverskyLoss

    description = FocalTverskyLoss.model_fields["gamma"].description
    assert description, "the field has no description at all"
    assert "Abraham" in description, (
        "the field is the reciprocal of a published parameter of the same name and does not "
        "say whose"
    )
    assert "1/gamma" in description, (
        "naming the paper without naming the relation leaves a reader to guess which way "
        "round it is, which is the thing that already cost this project a table row"
    )
    assert "not the gamma" in description.lower(), (
        "a reader has to be told the two are different parameters before the conversion "
        "means anything"
    )
    # Both directions of the conversion, because stating one invites the reader to apply it
    # backwards - and the numbers are the paper's own recommendation and its image here.
    assert "4/3" in description and "0.75" in description
