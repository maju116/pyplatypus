"""Generalized Dice, and where it may be named.

Two halves. The first is the arithmetic, and its control is an identity rather than a
second transcription: **with every class the same size this is exactly Dice pooled over the
classes**, so one of the two implementations already in this package computes the expected
value. The second half is the question that decides whether the loss is usable at all -
which tasks and which blocks accept it - asked of the configuration a user writes rather
than of the union.
"""

from __future__ import annotations

import pytest
import torch

from pyplatypus.errors import ConfigError
from pyplatypus.objectives import functional as f
from pyplatypus.objectives.losses import GeneralizedDice, build_loss
from pyplatypus.spec import from_dict
from pyplatypus.spec.components import DiceLoss, GeneralizedDiceLoss


def balanced(classes: int = 3, side: int = 8, seed: int = 0):
    """A one-hot target whose classes are the same size, and a softmax prediction."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, classes, side, side, generator=generator)
    target = torch.zeros(2, classes, side, side)
    flat = target.view(2, classes, -1)
    per_class = flat.shape[-1] // classes
    for index in range(classes):
        flat[:, index, index * per_class : (index + 1) * per_class] = 1.0
    return logits, target


def test_with_equal_classes_it_is_exactly_dice_pooled_over_them():
    """The identity the docstring claims, and the reason `smooth` here means what it means
    everywhere else in this file.

    It is also the control that caught the first version: the weights are of order
    1/volume^2, so an unscaled `smooth` of 1.0 was not a small regulariser but the whole
    numerator - the coefficient read 0.855 where pooled Dice reads 0.355. Rescaling the
    weights to a mean of one leaves the published quantity unchanged, because the ratio is
    invariant to a common factor, and makes this identity hold.
    """
    logits, target = balanced()
    probs = logits.softmax(dim=1)
    dims = (1, 2, 3)
    pooled = (2 * (probs * target).sum(dims) + 1.0) / ((probs + target).sum(dims) + 1.0)

    ours = f.generalized_dice_coefficient(probs, target)
    assert torch.allclose(ours, pooled, atol=1e-6)


def test_it_weights_a_rare_class_above_an_equal_share_and_the_table_says_by_how_much():
    """The description's table, held to the measurement rather than to a memory of it. §3C.

    **The premise this test was first written with was wrong**, which is why the table is
    here: `dice` in this package is a mean over the class channels, so it already prefers
    finding a 1% lesion to missing it - the usual "Dice is dominated by the background" is
    about a Dice pooled over channels and is false here. What this loss adds is weight
    beyond an equal share, and the honest way to show that is the margin rather than the
    value: it is twice `dice`'s on the same pair.

    The indifference of `cce` in that table is the thing worth seeing: a false positive and
    a missed lesion of the same size are the same number of wrong pixels to a per-pixel
    objective, and `cce_dice` - the loss every arm of the SIIM table used - is half that.
    """
    from pyplatypus.objectives.losses import Cce, CceDice, Dice

    side = 40
    target = torch.zeros(1, 2, side, side)
    target[0, 1, 2:6, 2:6] = 1.0  # 16 of 1600 pixels: 1% of the frame
    target[0, 0] = 1.0 - target[0, 1]

    found = target.clone()
    found[0, 1, 20:24, 20:24] = 1.0  # the lesion, and a false positive of equal size
    found[0, 0] = 1.0 - found[0, 1]
    missed = torch.zeros_like(target)
    missed[0, 0] = 1.0  # everything called background: the lesion is gone

    def logits_of(mask):
        return torch.where(mask > 0.5, 8.0, -8.0)

    measured = {}
    for name, loss in [
        ("dice", Dice()),
        ("cce", Cce()),
        ("cce_dice", CceDice()),
        ("generalized_dice", GeneralizedDice()),
    ]:
        measured[name] = (
            loss(logits_of(found), target).item(),
            loss(logits_of(missed), target).item(),
        )

    assert measured["dice"] == pytest.approx((0.1658, 0.4731), abs=5e-5)
    assert measured["cce"] == pytest.approx((0.1600, 0.1600), abs=5e-5)
    assert measured["cce_dice"] == pytest.approx((0.1629, 0.3166), abs=5e-5)
    assert measured["generalized_dice"] == pytest.approx((0.3278, 0.9511), abs=5e-5)

    gap = {name: miss - find for name, (find, miss) in measured.items()}
    assert gap["cce"] == pytest.approx(0.0, abs=1e-6), (
        "a per-pixel objective cannot tell them apart"
    )
    assert gap["generalized_dice"] > 2 * gap["dice"], "the weight beyond an equal share"

    # And the weights themselves, which is where the factor comes from: 1 to 9801 rather
    # than 1 to 1, read off the same formula the loss uses.
    volume = target.sum((2, 3))
    weight = 1.0 / volume.square()
    weight = weight / weight.sum() * 2
    assert (weight[0, 1] / weight[0, 0]).item() == pytest.approx(9801, rel=1e-3)


def test_an_absent_class_is_weighted_as_the_rarest_present_one_rather_than_ignored():
    """The decision, held to its arithmetic. A class with no volume has an infinite weight;
    the published implementations give it the largest finite weight in the sample, so
    inventing a class that is not in the picture costs as much as missing the rarest class
    that is. Weight zero - the other obvious choice - would make it free.
    """
    target = torch.zeros(1, 3, 4, 4)
    target[0, 0] = 1.0  # only class 0 is present
    invented = torch.zeros(1, 3, 4, 4)
    invented[0, 2] = 1.0  # all of it called class 2, which is absent

    # Flat weights after the borrow (one present class, two absent, all three equal), so
    # the pooled form gives the expected number exactly: nothing intersects, the totals
    # are 16 for class 0 and 16 for class 2.
    assert f.generalized_dice_coefficient(invented, target).item() == pytest.approx(
        1.0 / (16 + 16 + 1.0), abs=1e-6
    )

    # And it is not free: predicting the present class instead scores far better.
    correct = f.generalized_dice_coefficient(target, target).item()
    assert correct > 0.9


def test_a_sample_with_nothing_in_it_is_scored_the_way_dice_scores_it():
    """The second decision. With no class present there is no finite weight to borrow, so
    the weights are flat and an empty truth behaves exactly like `dice` on one. Without
    that fallback every weight is zero, the ratio is smooth/smooth, and a model predicting
    a lesion on a clean frame reads a perfect 1.0.
    """
    empty = torch.zeros(1, 2, 4, 4)
    confident = torch.zeros(1, 2, 4, 4)
    confident[0, 1] = 1.0

    ours = f.generalized_dice_coefficient(confident, empty).item()
    assert ours == pytest.approx(1.0 / (16 + 1.0), abs=1e-6)
    assert ours < 0.1, "an invented lesion on an empty frame is not a perfect score"


def test_the_loss_is_one_minus_the_coefficient_and_has_a_gradient():
    """It is a loss rather than a metric, so the gradient is the thing that has to exist."""
    logits, target = balanced()
    logits.requires_grad_(True)
    loss = GeneralizedDice()(logits, target)

    expected = 1 - f.generalized_dice_coefficient(logits.softmax(1), target).mean()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_it_works_on_overlapping_classes_rather_than_refusing_them():
    """`regions` makes each channel its own yes-or-no, and the weighting is per class
    either way - so this supports them, and the sigmoid arrives through `probabilities`."""
    spec = GeneralizedDiceLoss()
    assert spec.supports_multilabel is True

    logits = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    target = (torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(2)) > 0.6).float()
    loss = build_loss(spec, multilabel=True)
    assert loss.multilabel is True
    value = loss(logits, target)
    assert torch.isfinite(value) and 0 <= value.item() <= 1


def test_the_target_it_scores_is_classes_rather_than_a_vector_field():
    """`continuous` is what the instance task reads to refuse `loss: mse` over a one-hot
    target and `loss: dice` over a flow field. This scores classes, so it is False here -
    and that is what puts it in both segmentation listings and in neither flow block."""
    assert GeneralizedDiceLoss.continuous is False
    assert DiceLoss.continuous is False


# ------------------------------------------------------------ where it may be named
# The question the implementation is not finished without: a loss that is in the union and
# refused by every spec is a name in a listing and nothing else. Asked through
# `from_dict`, which is the configuration a user actually writes.
def test_a_semantic_model_can_name_it():
    spec = from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": ".",
                "validation_path": ".",
                "colormap": [[0, 0, 0], [255, 255, 255]],
            },
            "models": [
                {
                    "name": "m",
                    "input_shape": [64, 64],
                    "loss": {"name": "generalized_dice", "smooth": 0.5},
                }
            ],
        }
    )
    assert spec.models[0].loss.name == "generalized_dice"
    assert spec.models[0].loss.smooth == 0.5


def test_an_instance_model_can_name_it_for_the_mask():
    """The mask is classes at either separation, so the same loss serves it."""
    spec = from_dict(
        {
            "task": "instance_segmentation",
            "data": {"train_path": ".", "validation_path": "."},
            "models": [
                {
                    "name": "m",
                    "input_shape": [64, 64],
                    "separation": {"name": "rim", "width": 2},
                    "loss": {"name": "generalized_dice"},
                }
            ],
        }
    )
    assert spec.models[0].loss.name == "generalized_dice"


def test_the_boundary_loss_can_wrap_it_as_its_region_term():
    """The combination worth having, and the reason it is in `RegionLossSpec` as well: the
    case that motivates this loss is a lesion in a thousandth of a frame, and the case that
    motivates `boundary` is a small object whose boundary is most of it. They are the same
    picture.
    """
    spec = from_dict(
        {
            "task": "semantic_segmentation",
            "data": {
                "train_path": ".",
                "validation_path": ".",
                "colormap": [[0, 0, 0], [255, 255, 255]],
            },
            "models": [
                {
                    "name": "m",
                    "input_shape": [64, 64],
                    "loss": {
                        "name": "boundary",
                        "region": {"name": "generalized_dice"},
                        "alpha": 0.5,
                    },
                }
            ],
        }
    )
    built = build_loss(spec.models[0].loss)
    assert isinstance(built.region, GeneralizedDice)
    assert built.needs_distance is True


def test_a_flow_block_refuses_it_by_name():
    """The other half of the applicability question. A flow field is a vector per pixel and
    this scores classes, so naming it there is refused before anything is built - the
    refusal `continuous` exists for."""
    with pytest.raises(ConfigError, match="generalized_dice' is defined over classes"):
        from_dict(
            {
                "task": "instance_segmentation",
                "data": {"train_path": ".", "validation_path": "."},
                "models": [
                    {
                        "name": "m",
                        "input_shape": [64, 64],
                        "separation": {
                            "name": "flows",
                            "loss": {"name": "generalized_dice"},
                        },
                    }
                ],
            }
        )


def test_a_detector_cannot_name_it_at_all():
    """Detection chooses no loss component - its objective is fixed - so this is refused
    there for the reason every other loss is, rather than for one of its own."""
    with pytest.raises(ConfigError, match=r"models\[0\].loss"):
        from_dict(
            {
                "task": "object_detection",
                "data": {
                    "train_path": ".",
                    "validation_path": ".",
                    "classes": ["a"],
                    "mode": "config_file",
                },
                "models": [
                    {
                        "name": "d",
                        "architecture": "yolo3",
                        "input_shape": [64, 64],
                        "loss": {"name": "generalized_dice"},
                    }
                ],
            }
        )
