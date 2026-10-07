"""Metrics report on the mask the user is handed, not on the probabilities."""

import pytest
import torch

from pyplatypus.objectives import build_metrics
from pyplatypus.objectives import functional as f
from pyplatypus.objectives.metrics import build_metric
from pyplatypus.spec.components import DiceMetric, IouMetric, TverskyMetric

ALL = [IouMetric(), DiceMetric(), TverskyMetric()]
IDS = ["iou", "dice", "tversky"]


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_a_perfect_prediction_scores_one(spec):
    target = f.as_onehot(torch.randint(0, 2, (2, 16, 16)), 2)
    logits = (target - 0.5) * 40
    assert build_metric(spec)(logits, target).item() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_metrics_stay_within_zero_and_one(spec):
    torch.manual_seed(1)
    target = f.as_onehot(torch.randint(0, 3, (2, 16, 16)), 3)
    logits = torch.randn(2, 3, 16, 16)
    value = build_metric(spec)(logits, target).item()
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_metrics_work_on_volumes(spec):
    target = f.as_onehot(torch.randint(0, 2, (1, 8, 8, 8)), 2)
    logits = torch.randn(1, 2, 8, 8, 8)
    assert torch.isfinite(build_metric(spec)(logits, target))


def test_metrics_use_hard_predictions_not_probabilities():
    """An unsure model scores lower on a hard metric than on a soft one. Reporting the
    soft number would flatter it, and the gap grows exactly when the model is worst."""
    target = f.as_onehot(torch.tensor([[1, 1, 0, 0]]), 2)
    hesitant = torch.tensor([[[-0.1, -0.1, 0.1, 0.1], [0.1, 0.1, -0.1, -0.1]]])

    hard = build_metric(DiceMetric(smooth=0.0))(hesitant, target).item()
    soft = f.dice_coefficient(f.probabilities(hesitant), target, smooth=0.0).mean().item()
    assert hard == pytest.approx(1.0)  # the argmax is in fact correct
    assert soft < 0.6  # the probabilities are barely off a coin flip


def test_excluding_the_background_changes_the_number():
    """In medical images the background is most of the picture, so averaging it in
    flatters the score. Papers report foreground only."""
    target = torch.zeros(1, 2, 100)
    target[:, 0, :95] = 1.0  # 95% background
    target[:, 1, 95:] = 1.0
    logits = torch.full((1, 2, 100), -1.0)
    logits[:, 0] = 1.0  # predicts background everywhere

    with_background = build_metric(DiceMetric(smooth=0.0))(logits, target).item()
    foreground_only = build_metric(DiceMetric(smooth=0.0, include_background=False))(
        logits, target
    ).item()

    assert foreground_only == pytest.approx(0.0)  # missed every nucleus
    assert with_background > 0.48  # yet looks like a passing grade


def test_dropping_the_background_needs_a_second_class():
    target = torch.ones(1, 1, 4)
    with pytest.raises(ValueError, match="at least two classes"):
        build_metric(DiceMetric(include_background=False))(torch.zeros(1, 1, 4), target)


def test_tversky_at_half_is_dice_at_twice_the_smoothing():
    """Exactly Dice without smoothing, and Dice at 2s with it - multiplying Tversky
    through by two turns (TP + s) into (2TP + 2s). Worth pinning down, because comparing
    a Tversky run against a Dice run at the same `smooth` compares two different numbers."""
    torch.manual_seed(2)
    target = f.as_onehot(torch.randint(0, 2, (2, 16, 16)), 2)
    logits = torch.randn(2, 2, 16, 16)

    tversky = build_metric(TverskyMetric(alpha=0.5, smooth=0.0))(logits, target).item()
    assert tversky == pytest.approx(build_metric(DiceMetric(smooth=0.0))(logits, target).item())
    assert build_metric(TverskyMetric(alpha=0.5, smooth=1.0))(
        logits, target
    ).item() == pytest.approx(build_metric(DiceMetric(smooth=2.0))(logits, target).item())


def test_smoothing_hands_an_absent_class_a_perfect_score():
    """The reason metrics may use smooth=0 while losses may not: a class that appears
    nowhere scores 1.0 with smoothing, which quietly lifts the average."""
    target = torch.zeros(1, 2, 16)
    target[:, 0] = 1.0  # class 1 is absent from both
    logits = torch.tensor([[[5.0] * 16, [-5.0] * 16]])

    smoothed = build_metric(DiceMetric(smooth=1.0))(logits, target).item()
    honest = build_metric(DiceMetric(smooth=0.0))(logits, target)
    assert smoothed == pytest.approx(1.0)
    assert torch.isnan(honest)


def test_metrics_are_keyed_by_name():
    built = build_metrics([DiceMetric(), IouMetric()])
    assert set(built) == {"dice", "iou"}


def test_metrics_do_not_build_a_graph():
    """They run every validation batch; a retained graph would leak memory for nothing."""
    target = f.as_onehot(torch.randint(0, 2, (1, 8, 8)), 2)
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    assert build_metric(DiceMetric())(logits, target).requires_grad is False
