"""Numbers checked against arithmetic done by hand.

A sign error or a swapped alpha here is invisible - training still runs, the loss still
falls, the model just learns the wrong thing. So every coefficient is pinned to a value
worked out on paper for one tiny example.

The worked example, 1 sample, 2 classes, 4 pixels:

    target      class 1 at pixels 0,1        class 0 at pixels 2,3
    prediction  class 1 at pixel  0          class 0 at pixels 1,2,3

    class 1:  TP=1  FP=0  FN=1
    class 0:  TP=2  FP=1  FN=0
"""

import pytest
import torch

from pyplatypus.objectives import functional as f


@pytest.fixture
def worked_example():
    target = torch.tensor([[[0.0, 0, 1, 1], [1.0, 1, 0, 0]]])  # (1, 2, 4)
    prediction = torch.tensor([[[0.0, 1, 1, 1], [1.0, 0, 0, 0]]])
    return prediction, target


def test_overlaps_match_the_worked_example(worked_example):
    prediction, target = worked_example
    tp, fp, fn = f.overlaps(prediction, target)
    assert tp.tolist() == [[2.0, 1.0]]
    assert fp.tolist() == [[1.0, 0.0]]
    assert fn.tolist() == [[0.0, 1.0]]


def test_dice_without_smoothing(worked_example):
    """class 0: 2*2/(2*2+1+0) = 4/5    class 1: 2*1/(2*1+0+1) = 2/3"""
    prediction, target = worked_example
    dice = f.dice_coefficient(prediction, target, smooth=0.0)
    assert dice[0, 0].item() == pytest.approx(4 / 5)
    assert dice[0, 1].item() == pytest.approx(2 / 3)


def test_dice_with_smoothing(worked_example):
    """class 0: (4+1)/(4+1+0+1) = 5/6    class 1: (2+1)/(2+0+1+1) = 3/4"""
    prediction, target = worked_example
    dice = f.dice_coefficient(prediction, target, smooth=1.0)
    assert dice[0, 0].item() == pytest.approx(5 / 6)
    assert dice[0, 1].item() == pytest.approx(3 / 4)


def test_iou_without_smoothing(worked_example):
    """class 0: 2/(2+1+0) = 2/3    class 1: 1/(1+0+1) = 1/2"""
    prediction, target = worked_example
    iou = f.iou_coefficient(prediction, target, smooth=0.0)
    assert iou[0, 0].item() == pytest.approx(2 / 3)
    assert iou[0, 1].item() == pytest.approx(1 / 2)


def test_tversky_at_half_is_exactly_dice(worked_example):
    """The defining property of Tversky, and a good check that alpha is not swapped."""
    prediction, target = worked_example
    tversky = f.tversky_coefficient(prediction, target, alpha=0.5, smooth=0.0)
    dice = f.dice_coefficient(prediction, target, smooth=0.0)
    assert torch.allclose(tversky, dice)


def test_tversky_alpha_weights_false_negatives(worked_example):
    """class 1 has the false negative: 1/(1 + 0.7*1 + 0.3*0) = 1/1.7
    class 0 has the false positive: 2/(2 + 0.7*0 + 0.3*1) = 2/2.3"""
    prediction, target = worked_example
    tversky = f.tversky_coefficient(prediction, target, alpha=0.7, smooth=0.0)
    assert tversky[0, 1].item() == pytest.approx(1 / 1.7)
    assert tversky[0, 0].item() == pytest.approx(2 / 2.3)


def test_raising_alpha_punishes_missed_foreground(worked_example):
    """Higher alpha should push a recall-shy prediction down, not up."""
    prediction, target = worked_example
    lenient = f.tversky_coefficient(prediction, target, alpha=0.2, smooth=0.0)[0, 1]
    strict = f.tversky_coefficient(prediction, target, alpha=0.8, smooth=0.0)[0, 1]
    assert strict < lenient


def test_cross_entropy_of_a_coin_flip():
    """Uniform logits give p = 0.5 everywhere, so the loss is exactly ln 2."""
    logits = torch.zeros(1, 2, 4)
    target = torch.tensor([[[0.0, 0, 1, 1], [1.0, 1, 0, 0]]])
    assert f.cross_entropy(logits, target).item() == pytest.approx(0.6931471, abs=1e-6)


def test_focal_with_gamma_zero_is_cross_entropy():
    logits = torch.randn(2, 3, 8, 8)
    target = f.as_onehot(torch.randint(0, 3, (2, 8, 8)), 3)
    assert f.focal(logits, target, gamma=0.0).item() == pytest.approx(
        f.cross_entropy(logits, target).item(), rel=1e-5
    )


def test_focal_down_weights_the_easy_pixels():
    """At p = 0.5 the factor is (1 - 0.5)^2 = 0.25, so focal is a quarter of CE."""
    logits = torch.zeros(1, 2, 4)
    target = torch.tensor([[[0.0, 0, 1, 1], [1.0, 1, 0, 0]]])
    assert f.focal(logits, target, gamma=2.0).item() == pytest.approx(0.25 * 0.6931471, abs=1e-6)


# ------------------------------------------------------ combo, against the paper itself
# `pyplatypus#429`. Both of `combo`'s knobs survived being removed, and the reason is the
# defaults: at `alpha = 0.5` and `ce_ratio = 0.5` both weight swaps are the identity, so a
# test at the defaults cannot see either. What closes it is a value from outside at a setting
# where neither is.
#
# The outside reference is the paper the docstring cites - Taghanaki et al.,
# arXiv:1805.02798, equation (3) - TRANSCRIBED BY HAND from its own page rather than
# remembered:
#
#     L = alpha * ( -1/N SUM_i [ beta*(t_i ln p_i) + (1-beta)*((1-t_i) ln(1-p_i)) ] )
#         - (1 - alpha) * SUM_{k=1..K} ( (2 SUM_i p_i t_i + S) / (SUM_i p_i + SUM_i t_i + S) )
#
# Two readings in it are easy to get wrong from memory and both matter below: the Dice term
# is **subtracted** and is the Dice **coefficient** rather than `1 - Dice`, and it is a **sum
# over the K channels** rather than a mean. The paper's own experiments use a sigmoid over
# all channels and, for the multi-class case, flatten the one-hot volumes and compute the
# loss as if binary - so the setting its numbers live at is one channel.
_COMBO_ONE_CHANNEL = ([[2.0, -1.0, 0.0, 1.0, -2.0, 3.0]], [[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
_COMBO_THREE_CHANNEL = (
    [[2.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 1.0, -2.0], [0.0, 0.0, 2.0, 1.0]],
    [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
)
_COMBO_SETTINGS = [(0.5, 0.5), (0.3, 0.7), (0.8, 0.2), (0.25, 0.9)]
# Measured at both ends: the implementation agrees with the transcription to 4.23e-08 across
# all eight comparisons, and removing either knob moves the value by at least 1.1e-02. So
# 1e-6 is 23 times above the noise and 11,000 below the defect.
_COMBO_TOLERANCE = 1e-6


def _combo_tensors(case):
    logits, truth = case
    return (
        torch.tensor([logits], dtype=torch.float32),
        torch.tensor([truth], dtype=torch.float32),
    )


def _combo_paper_equation_three(case, alpha: float, beta: float, smooth: float = 1.0) -> float:
    """Equation (3) verbatim, in plain Python, with nothing imported from the package."""
    import math

    eps = 1e-7
    logits, truth = case

    def squash(x):
        return min(max(1.0 / (1.0 + math.exp(-x)), eps), 1 - eps)

    flat_p = [squash(x) for channel in logits for x in channel]
    flat_t = [t for channel in truth for t in channel]
    modified_cross_entropy = -sum(
        beta * (t * math.log(p)) + (1 - beta) * ((1 - t) * math.log(1 - p))
        for p, t in zip(flat_p, flat_t)
    ) / len(flat_p)
    dice_sum = 0.0
    for channel_logits, channel_truth in zip(logits, truth):
        p = [squash(x) for x in channel_logits]
        t = list(channel_truth)
        dice_sum += (2 * sum(a * b for a, b in zip(p, t)) + smooth) / (sum(p) + sum(t) + smooth)
    return alpha * modified_cross_entropy - (1 - alpha) * dice_sum


@pytest.mark.parametrize("alpha,ce_ratio", _COMBO_SETTINGS)
def test_combo_is_the_PAPERS_equation_three_plus_a_constant_at_one_channel(alpha, ce_ratio):
    """At the setting the paper's own numbers live at, the two agree up to `1 - alpha`.

    Ours writes the Dice half as `(1 - alpha) * (1 - dice)` where the paper writes
    `- (1 - alpha) * dice`, so the two differ by `(1 - alpha)` exactly - a constant in the
    prediction, so the same gradient and the same minimiser. That is a stronger statement
    than any single pinned number: it holds at four different `(alpha, ce_ratio)` pairs,
    including the two where neither knob is at its default.

    The first pair is the defaults, and it is in the list as the control: it is the setting
    every other test of `combo` uses, where both weight swaps are the identity.
    """
    logits, target = _combo_tensors(_COMBO_ONE_CHANNEL)
    ours = f.combo(logits, target, alpha, ce_ratio, multilabel=True).item()
    theirs = _combo_paper_equation_three(_COMBO_ONE_CHANNEL, alpha, ce_ratio)
    assert ours - theirs == pytest.approx(1 - alpha, abs=_COMBO_TOLERANCE)


def test_combo_AVERAGES_the_dice_term_over_channels_where_the_paper_SUMS_it():
    """The one place this implementation departs from the equation it cites.

    Equation (3) sums the Dice coefficient over the K channels, so its Dice half grows with
    the number of classes and `alpha` means something different per dataset. This package
    takes the **mean**, which is the same argument `binary_cross_entropy`'s own docstring
    makes one function over - *the mean over channels keeps one voxel worth the same whatever
    it was asked*.

    At one channel the two are identical and the difference is the constant above. At three
    the difference stops being constant, and that is what this pins: `(1 - alpha) * (1 + (K -
    1) * mean_dice)` rather than `(1 - alpha)`. Measured at `alpha = 0.5`: 1.2995642 against
    the constant's 0.5.
    """
    logits, target = _combo_tensors(_COMBO_THREE_CHANNEL)
    alpha, ce_ratio = 0.5, 0.5
    ours = f.combo(logits, target, alpha, ce_ratio, multilabel=True).item()
    theirs = _combo_paper_equation_three(_COMBO_THREE_CHANNEL, alpha, ce_ratio)

    assert ours - theirs != pytest.approx(1 - alpha, abs=1e-3)
    assert ours - theirs == pytest.approx(1.2995642, abs=1e-5)

    # And the relationship itself, so the claim above is held rather than asserted in prose:
    # ours - theirs = (1 - alpha) * (1 + (K - 1) * mean_dice), where the mean dice is what
    # ours uses and the sum is K times it.
    probabilities = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    mean_dice = f.dice_coefficient(probabilities, target, 1.0).mean().item()
    n_channel = logits.shape[1]
    assert ours - theirs == pytest.approx(
        (1 - alpha) * (1 + (n_channel - 1) * mean_dice), abs=_COMBO_TOLERANCE
    )


@pytest.mark.parametrize("alpha,ce_ratio", _COMBO_SETTINGS)
def test_combos_two_knobs_both_bite_away_from_their_defaults(alpha, ce_ratio):
    """Removing either knob has to move the answer, which at the defaults it does not.

    `alpha = 0.5` makes the two halves equal and `ce_ratio = 0.5` makes the two
    cross-entropy terms equal, so both swaps are the identity exactly there - which is why
    both survived every test this package had. Asked here at four settings, three of them
    away from the defaults.
    """
    logits, target = _combo_tensors(_COMBO_THREE_CHANNEL)
    honest = f.combo(logits, target, alpha, ce_ratio, multilabel=True).item()
    alpha_ignored = f.combo(logits, target, 0.5, ce_ratio, multilabel=True).item()
    ratio_ignored = f.combo(logits, target, alpha, 0.5, multilabel=True).item()

    if alpha != 0.5:
        assert honest != pytest.approx(alpha_ignored, abs=1e-3)
    if ce_ratio != 0.5:
        assert honest != pytest.approx(ratio_ignored, abs=1e-3)


# ------------------------------------------------- the three branches nothing ever ran
# `pyplatypus#429`. `label_smoothing` was never passed non-zero anywhere in either suite and
# `focal`'s multi-class `alpha` was never passed at all - `binary_focal`'s was, which is the
# kind of near-miss that makes a grep for the name look reassuring. So three branch bodies
# were dead code with prose over them, and the prose makes claims: `binary_cross_entropy`'s
# docstring says in so many words that its smoothing divides by two rather than by
# `n_class`, "because each channel is a two-outcome question".
#
# All three agree with an outside reference today, so this is HOLDING rather than correctness
# - which is the honest description and also the reason to write it: a branch nothing runs is
# a branch the next edit can change for free.
#
# The reference is `_BY_HAND` below: plain Python, `math` only, nothing imported from the
# package, the log-softmax and the stable BCE form written out. Agreement is then two
# implementations of one formula rather than a formula compared with itself.
_BRANCH_LOGITS = [
    [[[2.0, -1.0], [0.0, 1.0]], [[-1.0, 3.0], [1.0, -2.0]], [[0.0, 0.0], [2.0, 1.0]]],
    [[[-2.0, 1.0], [1.0, 0.0]], [[3.0, -1.0], [-1.0, 2.0]], [[1.0, 2.0], [0.0, -1.0]]],
]
_BRANCH_LABELS = [[[0, 1], [2, 0]], [[1, 2], [0, 1]]]
_BRANCH_N_CLASS = 3

# Both ends measured. The six comparisons below agree with the implementation to 4.34e-08 at
# worst - float32 noise - and the smallest thing a mutation moves is 7.7e-03, which is the
# BCE divisor. So 1e-6 sits 23 times above the noise and 7,700 below the defect.
_BRANCH_TOLERANCE = 1e-6


def _branch_fixture():
    logits = torch.tensor(_BRANCH_LOGITS, dtype=torch.float32)
    target = torch.zeros_like(logits)
    for sample, labels in enumerate(_BRANCH_LABELS):
        for row in range(2):
            for column in range(2):
                target[sample, labels[row][column], row, column] = 1.0
    return logits, target


def _branch_cells():
    for sample in range(2):
        for row in range(2):
            for column in range(2):
                logits = [_BRANCH_LOGITS[sample][k][row][column] for k in range(_BRANCH_N_CLASS)]
                truth = [
                    1.0 if _BRANCH_LABELS[sample][row][column] == k else 0.0
                    for k in range(_BRANCH_N_CLASS)
                ]
                yield logits, truth


def _log_softmax_by_hand(column):
    import math

    biggest = max(column)
    total = sum(math.exp(v - biggest) for v in column)
    return [v - biggest - math.log(total) for v in column]


def _cce_by_hand(label_smoothing: float) -> float:
    total = 0.0
    for column, truth in _branch_cells():
        log_probs = _log_softmax_by_hand(column)
        smoothed = [t * (1 - label_smoothing) + label_smoothing / _BRANCH_N_CLASS for t in truth]
        total += -sum(t * l for t, l in zip(smoothed, log_probs))
    return total / 8


def _bce_by_hand(label_smoothing: float) -> float:
    """Every channel contributes and the mean is over all of them, and the smoothing divides
    by TWO - both of which `binary_cross_entropy`'s own docstring states as decisions."""
    import math

    total, n = 0.0, 0
    for column, truth in _branch_cells():
        for k in range(_BRANCH_N_CLASS):
            x = column[k]
            t = truth[k] * (1 - label_smoothing) + label_smoothing / 2
            total += math.log1p(math.exp(-abs(x))) + max(x, 0.0) - x * t
            n += 1
    return total / n


def _focal_by_hand(gamma: float, alpha: float | None) -> float:
    import math

    total = 0.0
    for column, truth in _branch_cells():
        log_probs = _log_softmax_by_hand(column)
        weights = [max(1 - math.exp(l), 0.0) ** gamma for l in log_probs]
        if alpha is not None:
            per_class = [alpha] * _BRANCH_N_CLASS
            per_class[0] = 1 - alpha
            weights = [w * a for w, a in zip(weights, per_class)]
        total += -sum(t * w * l for t, w, l in zip(truth, weights, log_probs))
    return total / 8


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.3])
def test_cross_entropys_label_smoothing_divides_by_the_CLASS_COUNT(label_smoothing):
    """Zero is in the list first: it is the branch that DOES execute, so it is the control
    that the hand implementation is right where the suite already agrees."""
    logits, target = _branch_fixture()
    assert f.cross_entropy(logits, target, label_smoothing).item() == pytest.approx(
        _cce_by_hand(label_smoothing), abs=_BRANCH_TOLERANCE
    )


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.3])
def test_binary_cross_entropys_label_smoothing_divides_by_TWO_not_the_class_count(
    label_smoothing,
):
    """The distinction its docstring makes, asked of the code for the first time.

    Each channel is a two-outcome question, so smoothing it towards `n_class` outcomes would
    pull a binary target towards a number that is not in it. At `0.1` over three classes the
    two divisors differ by 7.7e-03 on this fixture - 0.5341815515 against 0.5418204403 - which
    is 177,000 times the float32 noise and the smallest separation in this block.
    """
    logits, target = _branch_fixture()
    assert f.binary_cross_entropy(logits, target, label_smoothing).item() == pytest.approx(
        _bce_by_hand(label_smoothing), abs=_BRANCH_TOLERANCE
    )


@pytest.mark.parametrize("alpha", [None, 0.75, 0.25])
def test_focals_alpha_puts_ONE_MINUS_alpha_on_the_background_class(alpha):
    """`None` first, which is the branch that executes, then the one that never did.

    The branch is two lines and the thing worth holding is the reversal: `weights[0]` takes
    `1 - alpha` and every other class takes `alpha`, so raising `alpha` moves weight onto the
    foreground. Written without the reversal - `alpha` on every class - the same fixture gives
    0.0300485410 against 0.0151686410 at `alpha = 0.75`, so it is a factor of two out and in
    the direction that makes the loss look larger.
    """
    logits, target = _branch_fixture()
    assert f.focal(logits, target, 2.0, alpha).item() == pytest.approx(
        _focal_by_hand(2.0, alpha), abs=_BRANCH_TOLERANCE
    )


def test_focals_alpha_INTERPOLATES_between_the_background_and_the_foreground():
    """The reversal, held at its two endpoints, where it is exact rather than directional.

    THE FIRST VERSION OF THIS TEST ASSERTED A DIRECTION AND THE DIRECTION IS NOT A PROPERTY
    OF `alpha`. It claimed that raising `alpha` raises the loss because five of this
    fixture's eight cells are foreground - a count, not a measurement. Measured: the three
    background cells carry 0.2381 of unweighted loss against 0.0824 over the five foreground
    ones, because the model here happens to be worse on the background, so raising `alpha`
    LOWERS the loss. The direction follows whichever cells the model is currently wrong
    about, and no fixture makes it a fact about the parameter.

    What is exact is the interpolation. `weight = base * a_c` with `a_0 = 1 - alpha` and
    `a_c = alpha` above it, so the loss is linear in `alpha` between two fixed ends:

        alpha = 0    0.0297598000    the background class alone
        alpha = 1    0.0103049213    the foreground classes alone
        alpha = 0.5  0.0200323607    the midpoint, to 0.00e+00

    **Written without the reversal** - `alpha` on every class - the same two ends are 0.0 and
    0.0400647213: a line through the origin rather than one between two sums. The endpoints
    are what tell the two apart, and neither of them is a direction.
    """
    logits, target = _branch_fixture()
    background_only = f.focal(logits, target, 2.0, 0.0).item()
    foreground_only = f.focal(logits, target, 2.0, 1.0).item()
    midpoint = f.focal(logits, target, 2.0, 0.5).item()

    assert background_only == pytest.approx(0.0297598000, abs=_BRANCH_TOLERANCE)
    assert foreground_only == pytest.approx(0.0103049213, abs=_BRANCH_TOLERANCE)
    # Linear in alpha, which is what "a weight per class" means and what an unreversed
    # implementation would also satisfy - so it is asserted BESIDE the endpoints, not
    # instead of them.
    assert midpoint == pytest.approx((background_only + foreground_only) / 2, abs=_BRANCH_TOLERANCE)
    # And the unreversed form is named rather than merely excluded: it puts zero weight on
    # everything at alpha = 0, so the first endpoint is the assertion that does the work.
    assert background_only > 0.0


def test_a_perfect_prediction_scores_one():
    target = f.as_onehot(torch.randint(0, 3, (2, 8, 8)), 3)
    for coefficient in (f.dice_coefficient, f.iou_coefficient, f.tversky_coefficient):
        assert coefficient(target, target).mean().item() == pytest.approx(1.0)


def test_lovasz_is_zero_for_a_perfect_prediction():
    target = f.as_onehot(torch.randint(0, 2, (1, 8, 8)), 2)
    logits = (target - 0.5) * 40  # saturates softmax to ~0/1
    assert f.lovasz_softmax(logits, target).item() == pytest.approx(0.0, abs=1e-4)


def _lovasz_case(n_label_values: int):
    """A fixed, deliberately IMPERFECT case, built the same way both tests below need it.

    float64 and seeded, because the figures under them are baked to ten digits.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 6, 6, dtype=torch.float64)
    labels = torch.randint(0, n_label_values, (2, 6, 6))
    target = torch.zeros(2, 3, 6, 6, dtype=torch.float64).scatter_(1, labels[:, None], 1.0)
    return logits, target


def test_lovasz_agrees_with_the_PUBLISHED_algorithm_on_a_fixed_case():
    """`pyplatypus#429`: a value from outside, because every test over this was a property.

    The ten tests that execute `lovasz_softmax` assert finiteness, a decrease, and a perfect
    prediction scoring 0 - and the twelfth review pass mutated the union, `|e|` to `e^2`, the
    absent-class rule and the SORT ORDER, and all four survived every one of them. They
    survive because a perfect prediction has every error 0 whatever the weights on them are.

    Where this number comes from, so it can be re-derived rather than trusted: Berman et
    al.'s own `lovasz_losses.py` (`bermanmaxim/LovaszSoftmax`) was read and its
    `lovasz_grad` + `lovasz_softmax_flat` transcribed by hand - never executed, because
    downloaded code is data. That transcription was then controlled on a two-pixel two-class
    case worked out on paper:

        class 0: fg=[0,1] p=[0.4,0.7] -> errors [0.4,0.3], sorted fg [0,1], grad [0.5,0.5] -> 0.35
        class 1: fg=[1,0] p=[0.6,0.3] -> errors [0.4,0.3], sorted fg [1,0], grad [1.0,0.0] -> 0.40
        the mean over the two present classes                                        -> 0.375

    and both the transcription and this package answer exactly 0.375 there. On the fixed
    case below both answer 0.7027821477, and the review that found the gap measured the same
    figure independently.

    The ascending sort - the one mutation of the four that is a real defect rather than a
    guard - moves this by 0.3485, and a two-layer network trained under it stalls at mIoU
    0.29 where Berman's reaches 0.97 over four seeds.
    """
    logits, target = _lovasz_case(3)
    assert f.lovasz_softmax(logits, target).item() == pytest.approx(0.7027821477, abs=1e-9)


def test_lovasz_skips_classes_that_are_absent():
    """Scoring an absent class as perfect would quietly inflate the loss's opinion.

    **This asserted 0 on a saturated prediction until `pyplatypus#429`**, which is §4's *a
    test named after a rule*: with the prediction perfect, an absent class scored contributes
    `max(error) = 0`, so the rule could not fail here and the mutation that scores absent
    classes passed. The case is imperfect now, and the two figures are what separate the two
    behaviours - 0.0992 apart, measured by the transcription the test above describes.
    """
    logits, target = _lovasz_case(2)  # class 2 never occurs
    assert target[:, 2].sum() == 0, "the fixture must leave one class absent"

    assert f.lovasz_softmax(logits, target).item() == pytest.approx(0.6683985262, abs=1e-9)
    # What scoring the absent class would give on this very case, so the number the rule
    # avoids is written down rather than implied.
    assert f.lovasz_softmax(logits, target).item() != pytest.approx(0.7676393149, abs=1e-6)


def test_onehot_accepts_indices_or_onehot():
    indices = torch.randint(0, 3, (2, 4, 4))
    onehot = f.as_onehot(indices, 3)
    assert onehot.shape == (2, 3, 4, 4)
    assert torch.equal(f.as_onehot(onehot, 3), onehot)
    assert torch.equal(onehot.argmax(1), indices)


def test_spatial_dims_is_everything_after_batch_and_channel():
    """The one line that makes every one of these functions work in 3D."""
    assert f.spatial_dims(torch.zeros(1, 2, 4)) == (2,)
    assert f.spatial_dims(torch.zeros(1, 2, 4, 4)) == (2, 3)
    assert f.spatial_dims(torch.zeros(1, 2, 4, 4, 4)) == (2, 3, 4)
