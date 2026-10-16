"""Normalised surface Dice, and the two morphological pieces it is built from.

Checked against `scipy.ndimage` rather than against a second transcription of the same
formula - the arrangement clDice has with scipy's morphology, the detection metrics with
`pycocotools` and the GIoU loss with `torchvision.ops`. The control here is sharper than a
reimplementation: `within` answers "is there a surface voxel inside this ball" by dilating
with the ball, and `distance_transform_edt` answers it by computing every distance exactly.
Two different algorithms, so agreement is evidence rather than a shared bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from pyplatypus.objectives import functional as f
from pyplatypus.objectives.metrics import SurfaceDice, build_metrics
from pyplatypus.spec.components import SurfaceDiceMetric


def blobs(shape, seed, threshold=0.52, sigma=2.2):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random(shape), sigma) > threshold


def as_batch(mask: np.ndarray) -> torch.Tensor:
    return torch.tensor(mask[None, None], dtype=torch.float32)


def onehot(mask: np.ndarray) -> torch.Tensor:
    """A 0-1 mask as the `(1, 2, *spatial)` one-hot the metrics take."""
    foreground = torch.tensor(mask, dtype=torch.float32)[None, None]
    return torch.cat([1 - foreground, foreground], dim=1)


def boundary_by_erosion(mask: np.ndarray) -> np.ndarray:
    """The border voxels, with scipy's morphology and none of our code.

    `border_value=1` because that is what `max_pool`'s -inf padding means for the
    complement: the image border is not eroded from outside.
    """
    element = np.ones((3,) * mask.ndim, bool)
    return mask & ~ndimage.binary_erosion(mask, element, border_value=1)


def nsd_by_transform(prediction, target, tau, spacing, smooth=1.0):
    """Normalised surface Dice with scipy alone: surfaces by erosion, distances by EDT."""
    surface_p = boundary_by_erosion(prediction)
    surface_t = boundary_by_erosion(target)
    to_p = ndimage.distance_transform_edt(~surface_p, sampling=spacing)
    to_t = ndimage.distance_transform_edt(~surface_t, sampling=spacing)
    agreed = (to_p[surface_t] <= tau).sum() + (to_t[surface_p] <= tau).sum()
    return (agreed + smooth) / (surface_t.sum() + surface_p.sum() + smooth)


@pytest.mark.parametrize("shape", [(40, 40), (18, 20, 16)])
@pytest.mark.parametrize("seed", range(4))
def test_the_surface_is_the_morphological_boundary(shape, seed):
    """Exactly, at either rank: a voxel is on the surface when a neighbour is outside, and
    `mask - erode(mask)` is that statement. Any difference is a defect, not a tolerance."""
    mask = blobs(shape, seed)
    assert mask.sum() > 40, "a fixture with almost nothing in it would prove nothing"

    ours = f.surface_of(as_batch(mask)).numpy()[0, 0].astype(bool)
    assert np.array_equal(ours, boundary_by_erosion(mask))


def test_a_mask_running_off_the_edge_has_no_surface_there():
    """The convention, stated in `surface_of` and worth a test because the alternative is
    defensible: the edge of the picture is not a boundary of the object, and counting it
    would score agreement on the field of view. A full mask therefore has no surface at
    all, which is also what makes two full masks score 1.0 rather than 0/0."""
    full = np.ones((12, 12), bool)
    assert f.surface_of(as_batch(full)).sum().item() == 0

    half = np.zeros((12, 12), bool)
    half[:, :6] = True
    surface = f.surface_of(as_batch(half)).numpy()[0, 0]
    # One column of surface, down the middle, and nothing along the three picture edges
    # the mask also touches.
    assert surface[:, 5].all()
    assert not surface[:, :5].any()
    assert surface.sum() == 12


@pytest.mark.parametrize(
    ("shape", "spacing"),
    [((40, 40), (1.0, 1.0)), ((40, 40), (0.5, 2.0)), ((18, 20, 16), (1.0, 1.0, 5.0))],
)
@pytest.mark.parametrize("tau", [0.0, 1.0, 2.5, 5.0])
def test_within_is_the_distance_transform_thresholded(shape, spacing, tau):
    """A ball dilation and an exact distance transform, which must agree voxel for voxel.

    The anisotropic rows are the ones that matter: at 0.5 by 2.0 a tolerance of 1 reaches
    two voxels one way and none the other, so a kernel sized in voxels rather than in
    millimetres would pass the isotropic row and fail these.
    """
    mask = blobs(shape, 1)
    surface = boundary_by_erosion(mask)
    assert surface.sum() > 20

    theirs = ndimage.distance_transform_edt(~surface, sampling=spacing) <= tau
    ours = f.within(as_batch(surface), tau, spacing).numpy()[0, 0].astype(bool)
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize(
    ("shape", "spacing"),
    [((40, 40), (1.0, 1.0)), ((40, 40), (0.5, 2.0)), ((18, 20, 16), (1.0, 1.0, 5.0))],
)
@pytest.mark.parametrize("tau", [1.0, 2.5])
def test_surface_dice_agrees_with_the_distance_transform(shape, spacing, tau):
    """The whole metric against the scipy-only version, on pairs that genuinely disagree."""
    truth = blobs(shape, 7)
    predicted = blobs(shape, 107, threshold=0.50)
    assert 0.2 < (truth & predicted).sum() / truth.sum() < 0.95, "a pair that agrees or "
    "disagrees completely would not discriminate"

    ours = f.surface_dice_from_masks(as_batch(predicted), as_batch(truth), tau, spacing).item()
    assert ours == pytest.approx(nsd_by_transform(predicted, truth, tau, spacing), abs=1e-6)


def test_the_tolerance_is_physical_rather_than_in_voxels():
    """The claim the required `tau` rests on: doubling the spacing halves what a tolerance
    reaches, so the same number means a different thing on a different acquisition - and
    `tau` at twice the spacing has to be twice as large to mean the same thing."""
    truth = blobs((40, 40), 3)
    predicted = blobs((40, 40), 103, threshold=0.50)
    at_one = f.surface_dice_from_masks(as_batch(predicted), as_batch(truth), 2.0, (1.0, 1.0))
    at_two = f.surface_dice_from_masks(as_batch(predicted), as_batch(truth), 2.0, (2.0, 2.0))
    doubled = f.surface_dice_from_masks(as_batch(predicted), as_batch(truth), 4.0, (2.0, 2.0))

    assert at_two < at_one, "a coarser grid cannot reach as far for the same tolerance"
    assert doubled == pytest.approx(at_one, abs=1e-6)


def test_a_spacing_of_the_wrong_length_is_refused_rather_than_broadcast():
    """Two numbers for a volume would measure a different shape of thing than the one
    passed, and silently: the ball would be built for a rank the mask does not have."""
    volume = as_batch(blobs((10, 12, 14), 2))
    with pytest.raises(ValueError, match="one spacing per spatial axis"):
        f.surface_dice_from_masks(volume, volume, 1.0, (1.0, 1.0))


def test_a_perfect_prediction_scores_one_and_a_displaced_one_does_not():
    """Both ends, because a metric that reads 1.0 for everything passes the first alone."""
    truth = np.zeros((30, 30), bool)
    truth[8:20, 8:20] = True
    moved = np.roll(truth, 5, axis=0)

    perfect = f.surface_dice_from_masks(as_batch(truth), as_batch(truth), 1.0).item()
    displaced = f.surface_dice_from_masks(as_batch(moved), as_batch(truth), 1.0).item()
    assert perfect == pytest.approx(1.0)
    assert displaced < 0.5

    # And the tolerance is what the metric is about: the same displacement inside tau.
    forgiven = f.surface_dice_from_masks(as_batch(moved), as_batch(truth), 6.0).item()
    assert forgiven == pytest.approx(1.0)


def test_two_masks_with_no_surface_score_one_rather_than_dividing_by_zero():
    """The `smooth` decision, and it is the same one `dice` takes: an empty prediction of an
    empty truth is right rather than undefined. Two solid masks have no surface either -
    see `surface_of` on the picture's edge - so they land here too."""
    empty = as_batch(np.zeros((8, 8), bool))
    solid = as_batch(np.ones((8, 8), bool))
    assert f.surface_dice_from_masks(empty, empty, 1.0).item() == pytest.approx(1.0)
    assert f.surface_dice_from_masks(solid, solid, 1.0).item() == pytest.approx(1.0)
    # But an empty prediction of a real truth is not forgiven.
    truth = np.zeros((20, 20), bool)
    truth[5:12, 5:12] = True
    nothing = as_batch(np.zeros((20, 20), bool))
    assert f.surface_dice_from_masks(nothing, as_batch(truth), 1.0).item() < 0.1


def test_the_background_class_reads_higher_than_the_foreground_as_the_description_says():
    """`include_background` defaults to False here and the description gives the numbers, so
    those numbers are held to the measurement rather than to a memory of it. §3C.

    The reason is structural: the background's boundary is the foreground's boundary plus
    the edge of the picture, and the edge agrees between any two masks whatever the model
    did - so it is agreement on the field of view.
    """
    backgrounds, foregrounds = [], []
    for seed in range(4):
        truth = blobs((48, 48), seed)
        predicted = blobs((48, 48), seed + 50, threshold=0.50)
        background, foreground = f.surface_dice_from_masks(onehot(predicted), onehot(truth), 2.0)[
            0
        ].tolist()
        assert background > foreground, seed
        backgrounds.append(background)
        foregrounds.append(foreground)

    assert min(backgrounds) == pytest.approx(0.6110, abs=5e-5)
    assert max(backgrounds) == pytest.approx(0.7030, abs=5e-5)
    assert min(foregrounds) == pytest.approx(0.5583, abs=5e-5)
    assert max(foregrounds) == pytest.approx(0.6266, abs=5e-5)


def test_the_metric_cannot_be_accumulated_from_overlap_counts():
    """A boundary is a property of a whole mask, like a skeleton. `accumulates` is False and
    `combine` says so rather than returning a number from counts that cannot carry it -
    which is what makes a tiled run report this per case instead of per batch."""
    metric = SurfaceDice(tau=1.0)
    assert metric.accumulates is False
    with pytest.raises(TypeError, match="property of a whole mask"):
        metric.combine(torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))


def test_the_tolerance_is_required_because_no_default_is_honest():
    """A tolerance belongs to the structure and the dataset. The metric therefore has no
    default for it, and a configuration that leaves it out is refused by name."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="tau"):
        SurfaceDiceMetric()
    with pytest.raises(pydantic.ValidationError):
        SurfaceDiceMetric(tau=0.0)
    assert SurfaceDiceMetric(tau=1.5).tau == 1.5


def test_the_spacing_reaches_the_metric_from_the_builder_and_not_from_the_spec():
    """`tau` is in millimetres only because something passes the spacing in. The spec
    carries the tolerance and the data carries the voxel, so the builder joins them - the
    same path `multilabel` takes, and the same reason."""
    built = build_metrics([SurfaceDiceMetric(tau=2.0)], spacing=(0.5, 0.5, 5.0))
    assert built["surface_dice"].spacing == (0.5, 0.5, 5.0)
    # And every metric gets the attribute, so nothing has to know which metrics read it.
    assert build_metrics([SurfaceDiceMetric(tau=2.0)])["surface_dice"].spacing is None


def test_the_metric_scores_a_hard_prediction_rather_than_the_logits():
    """Inherited, and worth asserting here because this metric reads the *shape* of a mask:
    fed probabilities it would find a boundary everywhere the model is unsure."""
    truth = np.zeros((24, 24), bool)
    truth[6:18, 6:18] = True
    logits = torch.where(onehot(truth) > 0.5, 0.2, -0.2)  # correct but barely
    metric = SurfaceDice(tau=1.0)
    assert metric(logits, onehot(truth)).item() == pytest.approx(1.0)
