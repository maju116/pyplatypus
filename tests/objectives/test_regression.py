"""The regression losses, and the question they exist to force: *where is the target
defined?*

A continuous instance target - a flow field, or horizontal and vertical distances - is
defined on the objects and undefined off them, so the builder writes zero on the
background. On the 536 Data Science Bowl cases the foreground is 13.97% of the pixels at
256x256, which makes an unmasked mean 86% a statement about a value a single output bias
satisfies exactly. These tests are about that distinction rather than about the arithmetic,
which torch owns and which the first test here holds it to. `pyplatypus#209`.
"""

from typing import get_args

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pyplatypus.data.masks import onehot_to_foreground
from pyplatypus.objectives import functional as f
from pyplatypus.objectives.losses import (
    build_loss,
    loss_needs_distance,
    loss_needs_mask,
)
from pyplatypus.spec.components import (
    BoundaryLoss,
    DiceLoss,
    L1Loss,
    LossSpec,
    MseLoss,
    SmoothL1Loss,
)
from pyplatypus.spec.loader import from_dict


def two_blobs(size=16, side=4):
    """Two squares, their flow field, and the mask of where that field means anything.

    The flow is the offset to the square's own centre, scaled to roughly [-1, 1] - which is
    what `flows` will build - and it is **exactly zero at each centre**, which is the reason
    the mask cannot be read off the target.
    """
    target = np.zeros((2, size, size), np.float32)
    mask = np.zeros((1, size, size), np.float32)
    for top, left in ((2, 2), (size - side - 2, size - side - 2)):
        rows = np.arange(top, top + side)
        cols = np.arange(left, left + side)
        centre_y, centre_x = rows.mean(), cols.mean()
        for y in rows:
            for x in cols:
                target[0, y, x] = (centre_y - y) / (side / 2)
                target[1, y, x] = (centre_x - x) / (side / 2)
                mask[0, y, x] = 1.0
    return torch.from_numpy(target)[None], torch.from_numpy(mask)[None]


# --- the arithmetic, held to torch's own ---------------------------------------------


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (f.mse, F.mse_loss),
        (f.l1, F.l1_loss),
        (lambda p, t: f.smooth_l1(p, t, 1.0), F.smooth_l1_loss),
    ],
    ids=["mse", "l1", "smooth_l1"],
)
def test_unmasked_is_torch_s_own_mean(ours, theirs):
    """A control that a wrong reduction could not produce, against an implementation this
    repository does not own."""
    generator = torch.Generator().manual_seed(0)
    prediction = torch.randn(2, 3, 8, 8, generator=generator)
    target = torch.randn(2, 3, 8, 8, generator=generator)
    assert ours(prediction, target).item() == pytest.approx(
        theirs(prediction, target).item(), rel=1e-6
    )


def test_smooth_l1_is_quadratic_inside_beta_and_linear_outside():
    """Two points, both exact: at half of `beta` it is `0.5 * e^2 / beta`, and at four
    times `beta` it is `|e| - 0.5 * beta`."""
    beta = 2.0
    near = f.smooth_l1(torch.tensor([1.0]), torch.tensor([0.0]), beta).item()
    far = f.smooth_l1(torch.tensor([8.0]), torch.tensor([0.0]), beta).item()
    assert near == pytest.approx(0.5 * 1.0**2 / beta)
    assert far == pytest.approx(8.0 - 0.5 * beta)


# --- where the target is defined -----------------------------------------------------


def test_the_mean_is_per_selected_voxel_rather_than_per_pixel():
    """The exact relation, which is the whole claim: with the target zero off the mask and
    the prediction zero, the two readings differ by exactly the foreground fraction. A
    division by the picture's size instead of the mask's total fails this by a factor of
    seven on this fixture, and by seven on the real data too."""
    target, mask = two_blobs()
    prediction = torch.zeros_like(target)
    fraction = mask.expand_as(target).mean().item()
    masked = f.mse(prediction, target, mask=mask).item()
    whole = f.mse(prediction, target).item()
    assert whole == pytest.approx(masked * fraction, rel=1e-6)
    assert fraction == pytest.approx(32 / 256)


def test_the_two_readings_disagree_about_which_prediction_is_better():
    """The test the family exists for, and it needs no training to make the point.

    Two predictions: one right on the background and wrong on the objects, one the other
    way round. `everywhere` prefers the first because the background is 87% of the picture;
    `foreground` prefers the second. Any implementation that quietly averaged over the
    whole picture would agree with itself twice here instead of disagreeing once.
    """
    target, mask = two_blobs()
    background_right = torch.zeros_like(target)  # exact off the objects, nothing on them
    foreground_right = target + (1 - mask) * 0.5  # exact on them, 0.5 out everywhere else

    whole = build_loss(MseLoss(where="everywhere"))
    on_objects = build_loss(MseLoss())

    assert whole(background_right, target).item() < whole(foreground_right, target).item()
    assert (
        on_objects(foreground_right, target, mask).item()
        < on_objects(background_right, target, mask).item()
    )


def test_a_single_channel_mask_selects_every_component_of_a_vector_target():
    """One foreground, both flow components. A mask applied to the first channel only
    would halve the number here."""
    target, mask = two_blobs()
    prediction = torch.zeros_like(target)
    both = f.mse(prediction, target, mask=mask).item()
    first_only = f.mse(prediction[:, :1], target[:, :1], mask=mask).item()
    second_only = f.mse(prediction[:, 1:], target[:, 1:], mask=mask).item()
    assert both == pytest.approx((first_only + second_only) / 2, rel=1e-6)


def test_an_empty_mask_is_zero_rather_than_nan():
    """A tile with no object in it has nothing to average. 167 of 536 Data Science Bowl
    cases are under 5% foreground, so with small tiles this is not a corner case."""
    target = torch.randn(1, 2, 8, 8)
    empty = torch.zeros(1, 1, 8, 8)
    value = f.mse(torch.zeros_like(target), target, mask=empty)
    assert value.item() == 0.0
    assert not torch.isnan(value)


def test_a_mask_that_does_not_match_the_target_is_refused():
    target = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="does not match"):
        f.mse(target, target, mask=torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError, match="does not match"):
        f.mse(target, target, mask=torch.zeros(2, 1, 8, 8))


def test_nothing_activates_the_prediction():
    """The failure this family is one sigmoid away from. A flow of 3.0 predicted as 3.0 is
    exactly right; through a sigmoid it would be 0.95 against 3.0 and the loss would be
    large, non-zero and still differentiable - a number and a gradient for a model that is
    correct."""
    value = torch.full((1, 2, 4, 4), 3.0)
    assert f.mse(value, value).item() == 0.0
    assert f.l1(value, value).item() == 0.0
    assert f.smooth_l1(value, value).item() == 0.0
    assert build_loss(MseLoss(where="everywhere"))(value, value).item() == 0.0


# --- how the mask gets there ---------------------------------------------------------


def test_foreground_without_a_mask_is_refused_and_names_the_loader_flag():
    target, _ = two_blobs()
    loss = build_loss(MseLoss())
    with pytest.raises(ValueError, match="with_foreground=True"):
        loss(torch.zeros_like(target), target)


def test_everywhere_needs_no_mask_and_takes_two_arguments():
    target, _ = two_blobs()
    loss = build_loss(MseLoss(where="everywhere"))
    assert loss.needs_mask is False
    assert loss(torch.zeros_like(target), target).item() > 0


@pytest.mark.parametrize(
    ("spec", "wanted"),
    [
        (MseLoss(), True),
        (L1Loss(), True),
        (SmoothL1Loss(), True),
        (MseLoss(where="everywhere"), False),
        (L1Loss(where="everywhere"), False),
        (SmoothL1Loss(where="everywhere"), False),
        (DiceLoss(), False),
        (BoundaryLoss(), False),
    ],
)
def test_loss_needs_mask_is_answered_from_the_specification(spec, wanted):
    """From the spec, because the loader is built before the loss is - the same reason
    `loss_needs_distance` is a function of the spec rather than of the module."""
    assert loss_needs_mask(spec) is wanted
    assert build_loss(spec).needs_mask is wanted


def test_the_two_data_flags_are_never_both_set():
    """The loader carries one extra tensor, and the trainer unpacks by length. That is only
    honest while no loss asks for two, so this asks every loss in the union rather than the
    three that were in mind when it was written."""
    members = get_args(get_args(LossSpec)[0])
    assert len(members) == 14, "a loss was added; this test is the reason to look"
    # Looked, for `generalized_dice` in `0.8.0a26`: a region loss on one-hot classes,
    # so neither flag, and the loader still has one slot to fill.
    for member in members:
        spec = member()
        assert not (loss_needs_distance(spec) and loss_needs_mask(spec)), member.__name__
        built = build_loss(spec)
        assert not (built.needs_distance and built.needs_mask), member.__name__


@pytest.mark.parametrize("name", ["mse", "l1", "smooth_l1"])
def test_every_regression_loss_is_reachable_from_a_configuration(name):
    """From the configuration a user can write, which is a `separation: flows` block's own
    `loss` and not a model's. This asked it of a *semantic* model until `0.8.0a21`, and that
    spec was accepted and trained - `pyplatypus#223`."""
    spec = from_dict(
        {
            "task": "instance_segmentation",
            "data": {"train_path": ".", "validation_path": "."},
            "models": [
                {
                    "name": "m",
                    "input_shape": [64, 64],
                    "separation": {
                        "name": "flows",
                        "loss": {"name": name, "where": "everywhere"},
                    },
                }
            ],
        },
        check_paths=False,
    )
    assert spec.models[0].separation.loss.where == "everywhere"


def test_no_loss_a_semantic_model_can_name_asks_for_the_foreground():
    """`Engine.loader` still reads `loss_needs_mask(model.loss)`, and since `0.8.0a21`
    nothing a semantic specification can name answers yes. The branch stays because the
    loader's question is the *loss's* - it is built before the loss object exists, so it
    reads the flag rather than the task - and this test is what keeps that honest: if a
    loss a semantic model may name ever wants the mask, the branch has a user again and the
    second assertion below is the one that fails.

    **This used to build the loader from a semantic spec with `loss: mse` and assert the
    batch had three items.** It could, because that spec was accepted and trained
    (`pyplatypus#223`). What it proved about the slot is proved on the path that still has a
    user by `test_the_loader_carries_the_foreground_for_flows_and_not_for_rim`, which checks
    the foreground's content rather than only the batch's length.
    """
    from pyplatypus import available_losses
    from pyplatypus.spec.components import LossSpec

    members = {m.model_fields["name"].default: m for m in get_args(get_args(LossSpec)[0])}
    nameable = {row["name"] for row in available_losses("semantic_segmentation")}
    assert len(nameable) == 11
    for name in nameable:
        assert not loss_needs_mask(members[name]()), name
    # And the three that do want it are exactly the three a semantic model cannot name, so
    # the branch is unreachable by configuration rather than merely unused today.
    for name in ("mse", "l1", "smooth_l1"):
        assert name not in nameable
        assert loss_needs_mask(members[name]())


def test_a_loader_cannot_be_asked_for_both_extras():
    from pyplatypus.training.torch_data import TorchSegmentationDataset

    with pytest.raises(ValueError, match="one extra tensor"):
        TorchSegmentationDataset(object(), with_distance=True, with_foreground=True)


# --- and it trains -------------------------------------------------------------------


def test_a_flow_shaped_target_trains_to_a_number_that_falls():
    """Small enough for the suite, and it is the masked objective being optimised: the
    model sees the mask as its input channel and has to put the right vector in each
    object. Without this the family would be arithmetic nothing had used."""
    torch.manual_seed(0)
    target, mask = two_blobs()
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 2, 3, padding=1))
    loss = build_loss(MseLoss())
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)

    first = loss(model(mask), target, mask).item()
    for _ in range(80):
        optimiser.zero_grad()
        value = loss(model(mask), target, mask)
        value.backward()
        optimiser.step()
    last = loss(model(mask), target, mask).item()
    assert last < first / 2, f"{first:.4f} -> {last:.4f}"


def test_onehot_to_foreground_reads_the_namings_background_convention():
    """Channel 0 is background in the exclusive namings - which is what
    `labels_to_classes` and `colours_to_classes` both mean by sending an unmatched voxel to
    index 0 - and `regions` has no background channel at all."""
    exclusive = np.eye(3, dtype=np.float32)[np.array([[0, 1], [2, 0]])]
    assert onehot_to_foreground(exclusive)[..., 0].tolist() == [[0, 1], [1, 0]]

    # Two regions, and a voxel in neither: `region_vocabulary`'s row of zeros.
    regions = np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], np.float32)
    assert onehot_to_foreground(regions, multilabel=True)[..., 0].tolist() == [[0, 1], [1, 1]]
    # Read as exclusive, the same array disagrees about three of the four voxels - the
    # first channel stops being background and becomes a class. Which is why the flag is
    # passed rather than guessed from the array.
    assert onehot_to_foreground(regions)[..., 0].tolist() == [[1, 0], [1, 0]]
