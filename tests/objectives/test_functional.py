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
    target = torch.tensor([[[0.0, 0, 1, 1], [1.0, 1, 0, 0]]])       # (1, 2, 4)
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
    assert f.focal(logits, target, gamma=2.0).item() == pytest.approx(
        0.25 * 0.6931471, abs=1e-6
    )


def test_a_perfect_prediction_scores_one():
    target = f.as_onehot(torch.randint(0, 3, (2, 8, 8)), 3)
    for coefficient in (f.dice_coefficient, f.iou_coefficient, f.tversky_coefficient):
        assert coefficient(target, target).mean().item() == pytest.approx(1.0)


def test_lovasz_is_zero_for_a_perfect_prediction():
    target = f.as_onehot(torch.randint(0, 2, (1, 8, 8)), 2)
    logits = (target - 0.5) * 40          # saturates softmax to ~0/1
    assert f.lovasz_softmax(logits, target).item() == pytest.approx(0.0, abs=1e-4)


def test_lovasz_skips_classes_that_are_absent():
    """Scoring an absent class as perfect would quietly inflate the loss's opinion."""
    target = torch.zeros(1, 3, 16)
    target[:, 0] = 1.0                     # only the background is present
    logits = (target - 0.5) * 40
    assert f.lovasz_softmax(logits, target).item() == pytest.approx(0.0, abs=1e-4)


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
