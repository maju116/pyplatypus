"""Nested regions: overlapping classes named as unions of a label map's values.

The case is BraTS and the lesion-inside-an-organ family: the ground truth gives every
voxel one value, and the classes that get scored are unions of those values. So the
overlap is a *convention applied to exclusive data*, which is why nothing new is read off
disk and why the whole feature lands in two places - a third way of naming classes, and
the one step that expands a class array into channels.

Most of what is tested here is the second kind of mistake, where a wrong answer still
looks like an answer: a background voxel quietly joining the first region, an `argmax`
reporting one class for a voxel that is in two, a loss whose scale moves with the number
of regions, and a region naming a value the data does not hold.
"""

import warnings

import numpy as np
import pytest
import torch
from PIL import Image

from pyplatypus import Engine, from_dict
from pyplatypus.data.masks import MaskError, labels_to_regions, onehot_to_colours, region_vocabulary
from pyplatypus.errors import ConfigError
from pyplatypus.objectives import functional as f
from pyplatypus.objectives.losses import build_loss
from pyplatypus.objectives.metrics import build_metrics
from pyplatypus.spec.components import CceLoss, DiceLoss, DiceMetric, LovaszLoss
from pyplatypus.spec.data import SegmentationData

#: The three scored regions of BraTS, over a ground truth holding {0, 1, 2, 4}.
BRATS = [
    {"name": "enhancing", "of": [4]},
    {"name": "core", "of": [1, 4]},
    {"name": "whole", "of": [1, 2, 4]},
]


def data(**overrides):
    block = {"train_path": "t", "validation_path": "v", "regions": BRATS}
    block.update(overrides)
    return SegmentationData(**block)


# --------------------------------------------------------------------- the naming


def test_regions_are_a_third_naming_and_the_three_are_exclusive():
    assert data().n_class == 3
    assert data().multilabel is True
    assert data().label_map is True, "regions read the file `labels` reads"
    for block, why in [
        ({"labels": [0, 1], "regions": BRATS}, "labels and regions"),
        ({"colormap": [[0, 0, 0], [255, 255, 255]], "regions": BRATS}, "colormap and regions"),
        ({"regions": None}, "no naming at all"),
    ]:
        with pytest.raises(ValueError, match="exactly one of"):
            data(**block)


def test_neither_labels_nor_a_colormap_is_multilabel():
    assert data(labels=[0, 1], regions=None).multilabel is False
    assert data(colormap=[[0, 0, 0], [255, 255, 255]], regions=None).multilabel is False


def test_two_regions_are_the_minimum_and_a_region_of_one_value_is_normal():
    # BraTS's enhancing tumour is a single value, so `of` of length one must be accepted.
    assert data(regions=[{"name": "a", "of": [4]}, {"name": "b", "of": [1, 4]}]).n_class == 2
    with pytest.raises(ValueError, match="at least 2 items"):
        data(regions=[{"name": "only", "of": [1]}])


def test_two_regions_may_not_be_the_same_question_twice():
    with pytest.raises(ValueError, match="region names must be distinct"):
        data(regions=[{"name": "same", "of": [1]}, {"name": "same", "of": [2]}])
    with pytest.raises(ValueError, match="both made of"):
        # Written in different orders, so a naive comparison would let them through.
        data(regions=[{"name": "a", "of": [1, 4]}, {"name": "b", "of": [4, 1]}])
    with pytest.raises(ValueError, match="says nothing twice"):
        data(regions=[{"name": "a", "of": [1, 1]}, {"name": "b", "of": [2]}])


# ----------------------------------------------------------- vocabulary and table


def test_the_vocabulary_is_every_value_mentioned_once_and_sorted():
    values, table = region_vocabulary([r["of"] for r in BRATS])
    assert values == [1, 2, 4]
    assert table.shape == (4, 3), "one row per value plus one, one column per region"


def test_row_zero_is_zeros_rather_than_the_first_region():
    """The mistake `labels_to_classes` would have made if it had been reused unchanged.

    It sends an unmatched voxel to index 0 because index 0 is a class there - the
    background a colormap puts first. Here index 0 must mean "no listed value", and on
    BraTS the lowest listed value is 1, which belongs to both `core` and `whole`. Reusing
    it would have put every background voxel in two of the three regions.
    """
    values, table = region_vocabulary([r["of"] for r in BRATS])
    assert table[0].tolist() == [0.0, 0.0, 0.0]
    assert table[values.index(1) + 1].tolist() == [0.0, 1.0, 1.0]


def test_the_table_puts_a_value_in_every_region_that_contains_it():
    values, table = region_vocabulary([r["of"] for r in BRATS])
    expected = {1: [0, 1, 1], 2: [0, 0, 1], 4: [1, 1, 1]}
    for value, membership in expected.items():
        assert table[values.index(value) + 1].tolist() == [float(x) for x in membership]


def test_a_value_no_region_mentions_lands_in_no_channel():
    values, table = region_vocabulary([r["of"] for r in BRATS])
    mask = np.array([[[0.0], [1.0]], [[2.0], [4.0]]], dtype=np.float32)
    indices, unmatched = labels_to_regions(mask, values)
    assert indices.tolist() == [[0, 1], [2, 3]]
    assert unmatched == pytest.approx(0.25), "one of the four voxels is the background"
    assert table[indices][0, 0].tolist() == [0.0, 0.0, 0.0]


def test_a_resampled_label_map_still_matches():
    """Float values arrive from resampling and from a round trip through float32, so the
    comparison is a tolerance rather than equality - the same reason `labels_to_classes`
    has one. A label silently failing to match would become background."""
    values, _ = region_vocabulary([r["of"] for r in BRATS])
    mask = np.array([[[4.0 - 1e-6], [1.0 + 1e-6]]], dtype=np.float32)
    indices, unmatched = labels_to_regions(mask, values)
    assert unmatched == 0.0
    assert indices.tolist() == [[3, 1]]


def test_a_label_map_with_several_channels_is_refused():
    values, _ = region_vocabulary([r["of"] for r in BRATS])
    with pytest.raises(MaskError, match="use a colormap"):
        labels_to_regions(np.zeros((4, 4, 3), np.float32), values)


# ----------------------------------------------------------------- the data path


def write_case(root, name, *, radii=(7, 14, 22), size=64):
    """One BraTS-shaped case: concentric rings valued 4, 2 and 1 outwards."""
    sample = root / name
    (sample / "images").mkdir(parents=True, exist_ok=True)
    (sample / "masks").mkdir(parents=True, exist_ok=True)
    yy, xx = np.ogrid[:size, :size]
    distance = np.sqrt((yy - size // 2) ** 2 + (xx - size // 2) ** 2)
    mask = np.zeros((size, size), np.uint8)
    mask[distance < radii[2]] = 1
    mask[distance < radii[1]] = 2
    mask[distance < radii[0]] = 4
    Image.fromarray(mask).save(sample / "masks" / "seg.png")
    image = np.clip(mask.astype(np.float32) * 40 + 20, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([image] * 3, axis=-1)).save(sample / "images" / "t1.png")
    return sample


def spec_for(root, *, regions=None, loss="cce_dice", metrics=None, **model):
    block = {
        "name": "m",
        "architecture": "u_net",
        "input_shape": [64, 64],
        "channels": 3,
        "blocks": 2,
        "filters": 4,
        "batch_size": 2,
        "epochs": 1,
        "loss": {"name": loss},
        "metrics": metrics if metrics is not None else [{"name": "dice"}],
    }
    block.update(model)
    return from_dict(
        {
            "task": "semantic_segmentation",
            "seed": 1,
            "data": {
                "train_path": str(root),
                "validation_path": str(root),
                "mode": "nested_dirs",
                "regions": BRATS if regions is None else regions,
            },
            "models": [block],
        }
    )


@pytest.fixture
def brats_like(tmp_path):
    root = tmp_path / "train"
    for index in range(4):
        write_case(root, f"case_{index}")
    return root


def test_the_target_puts_a_voxel_in_every_region_that_contains_it(brats_like):
    spec = spec_for(brats_like)
    _, target = Engine(spec).dataset(spec.models[0], "train")[0]
    assert target.shape == (64, 64, 3)
    enhancing, core, whole = (target[..., c] > 0 for c in range(3))
    # Nested by construction, so each region must contain the one inside it entirely.
    assert np.all(core[enhancing]), "every enhancing voxel is in the core"
    assert np.all(whole[core]), "every core voxel is in the whole tumour"
    assert enhancing.sum() < core.sum() < whole.sum()
    assert (target.sum(axis=-1) > 1).sum() == core.sum(), (
        "the voxels in more than one region are exactly the core, since it nests inside "
        "the whole tumour and contains the enhancing part"
    )


def test_the_target_is_not_one_hot(brats_like):
    """What would be true if the exclusive path had been taken by mistake."""
    spec = spec_for(brats_like)
    _, target = Engine(spec).dataset(spec.models[0], "train")[0]
    assert not np.all(target.sum(axis=-1) == 1)


def test_a_region_naming_a_value_the_data_does_not_hold_is_refused(brats_like):
    """`of: [3]` against data holding 4 is a channel of zeros, which trains to predict
    nothing and scores a Dice of 0 with every other number looking ordinary."""
    regions = [{"name": "typo", "of": [3]}, {"name": "whole", "of": [1, 2, 4]}]
    with pytest.raises(Exception, match="region 'typo'"):
        Engine(spec_for(brats_like, regions=regions)).fit()


def test_the_unmatched_fraction_is_high_and_not_warned_about(brats_like):
    """Background is deliberately not a region, so it counts as unmatched. Warning here
    would fire on every correct BraTS run and teach the reader to ignore the warning."""
    spec = spec_for(brats_like)
    engine = Engine(spec)
    report = engine.dataset(spec.models[0], "train").inspect_masks(limit=4)
    assert report.unmatched > 0.5, "most of the picture is background"
    assert report.missing_classes == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine.fit()


# ------------------------------------------------------------------- objectives


def test_probabilities_are_a_sigmoid_and_do_not_sum_to_one():
    logits = torch.tensor([[[[2.0]], [[1.0]], [[0.5]]]])
    assert f.probabilities(logits).sum(dim=1).item() == pytest.approx(1.0)
    independent = f.probabilities(logits, multilabel=True)
    assert independent.sum(dim=1).item() > 1.0
    assert independent[0, 0, 0, 0].item() == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())


#: `torchvision.ops.sigmoid_focal_loss` on the problem `focal_reference_problem` builds,
#: read off torchvision 0.22.1 and baked in. Baked rather than imported because torchvision
#: arrives only with the `encoders` extra, so a test importing it would skip on the ordinary
#: install - and a skip reads exactly like a pass. The same arrangement the detection metrics
#: have with `pycocotools`.
TORCHVISION_FOCAL = {
    (2.0, None): 0.34679388999938965,
    (2.0, 0.25): 0.1932813674211502,
    (1.0, 0.75): 0.22521907091140747,
    (0.0, None): 0.7979759573936462,
    (3.0, 0.5): 0.12680160999298096,
}


def focal_reference_problem():
    """The logits and target the baked numbers were measured on."""
    torch.manual_seed(1)
    return torch.randn(2, 3, 8, 8), (torch.rand(2, 3, 8, 8) > 0.6).float()


def test_the_focal_reference_problem_is_not_trivial():
    """A fixture whose target is all zeros or all ones would agree with anything."""
    _, target = focal_reference_problem()
    share = float(target.mean())
    assert 0.2 < share < 0.6, f"the reference target is {share:.0%} positive"
    assert len(set(TORCHVISION_FOCAL.values())) == len(TORCHVISION_FOCAL), (
        "five settings giving one number would mean gamma and alpha do nothing"
    )


@pytest.mark.parametrize(("gamma", "alpha"), sorted(TORCHVISION_FOCAL, key=str))
def test_binary_focal_agrees_with_torchvision(gamma, alpha):
    logits, target = focal_reference_problem()
    assert f.binary_focal(logits, target, gamma, alpha).item() == pytest.approx(
        TORCHVISION_FOCAL[(gamma, alpha)], abs=2e-8
    )


def test_binary_focal_is_binary_cross_entropy_at_gamma_zero():
    """A control the implementation could not satisfy by accident: focal with no focusing
    and no class weight *is* cross-entropy, so the two must agree exactly."""
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 8, 8)
    target = (torch.rand(2, 3, 8, 8) > 0.6).float()
    assert f.binary_focal(logits, target, gamma=0.0).item() == pytest.approx(
        f.binary_cross_entropy(logits, target).item(), abs=1e-7
    )


def test_binary_cross_entropy_does_not_grow_with_the_number_of_regions():
    """The reduction that differs from the categorical form, and why.

    Summing over channels would make three regions read three times the scale of one, so
    `cce_weight` in a blended loss would mean something different on every dataset. The
    mean keeps one voxel worth the same whatever it was asked.
    """
    torch.manual_seed(2)
    one = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    narrow = f.binary_cross_entropy(one, target)
    wide = f.binary_cross_entropy(one.repeat(1, 4, 1, 1), target.repeat(1, 4, 1, 1))
    assert wide.item() == pytest.approx(narrow.item(), rel=1e-6)


def test_every_overlap_loss_reads_the_flag_rather_than_the_functional_form():
    """One home for the softmax-or-sigmoid decision: if a loss called the functional
    `probabilities` directly, its number would not move when the flag does."""
    torch.manual_seed(3)
    logits = torch.randn(2, 3, 8, 8)
    target = (torch.rand(2, 3, 8, 8) > 0.6).float()
    for spec in (DiceLoss(name="dice"), CceLoss(name="cce")):
        exclusive = build_loss(spec)(logits, target)
        overlapping = build_loss(spec, multilabel=True)(logits, target)
        assert exclusive.item() != pytest.approx(overlapping.item())


def test_a_metric_thresholds_each_channel_rather_than_taking_the_strongest():
    """One voxel, truly in two regions, and the model says both. An `argmax` has to pick
    one and scores it wrong; a per-channel threshold agrees with the truth."""
    logits = torch.tensor([[[[3.0]], [[2.0]], [[-4.0]]]])
    target = torch.tensor([[[[1.0]], [[1.0]], [[0.0]]]])
    exclusive = build_metrics([DiceMetric(name="dice")])["dice"]
    overlapping = build_metrics([DiceMetric(name="dice")], multilabel=True)["dice"]
    assert overlapping(logits, target).item() == pytest.approx(1.0)
    # The number the changelog quotes. Loosely bounded it would pass for a metric that was
    # merely different rather than wrong in this particular way.
    assert exclusive(logits, target).item() == pytest.approx(0.8333333, abs=1e-6)


def test_the_trainer_hands_the_flag_to_its_loss_and_every_metric():
    """A hole a mutation found: the flag reaching `predict` but not the objectives.

    Nothing about the shapes changes if it does not, so training runs to the end and
    reports numbers - a softmax loss over channels that are not alternatives, and metrics
    scoring an `argmax` of a model that was never asked for one. The only symptom is that
    the numbers are wrong, which is the symptom this project trusts least.
    """
    from pyplatypus.models.unet import build_model
    from pyplatypus.spec.models import SegmentationModel
    from pyplatypus.training.trainer import Trainer

    spec = SegmentationModel(
        name="m",
        architecture="u_net",
        input_shape=[32, 32],
        channels=3,
        blocks=1,
        filters=2,
        loss={"name": "cce_dice"},
        metrics=[{"name": "dice"}, {"name": "iou"}],
    )
    network = build_model(spec, n_class=3)
    for multilabel in (False, True):
        trainer = Trainer(network, spec, device="cpu", multilabel=multilabel)
        assert trainer.loss_fn.multilabel is multilabel
        assert [metric.multilabel for metric in trainer.metrics.values()] == [multilabel] * 2


def test_include_background_is_ignored_with_regions_rather_than_dropping_one():
    """Channel 0 is the first region somebody asked for, not a convention."""
    logits = torch.tensor([[[[3.0]], [[-3.0]]]])
    target = torch.tensor([[[[1.0]], [[0.0]]]])
    metric = build_metrics([DiceMetric(name="dice", include_background=False)], multilabel=True)[
        "dice"
    ]
    assert metric(logits, target).item() == pytest.approx(1.0)


# --------------------------------------------------------------- what is refused


def test_lovasz_is_refused_with_regions(brats_like):
    with pytest.raises(ConfigError, match="mutually exclusive classes"):
        spec_for(brats_like, loss="lovasz")
    with pytest.raises(ValueError, match="cannot be used with overlapping"):
        build_loss(LovaszLoss(name="lovasz"), multilabel=True)


def test_an_explicit_include_background_false_is_refused_but_a_default_is_not(brats_like):
    """`cldice` defaults it to False, and a default is not a request - the same distinction
    `channels_match_the_data` draws with `model_fields_set`."""
    with pytest.raises(ConfigError, match="does not have"):
        spec_for(brats_like, metrics=[{"name": "dice", "include_background": False}])
    spec = spec_for(brats_like, metrics=[{"name": "cldice"}])
    assert spec.models[0].metrics[0].include_background is False


def test_a_colour_map_refuses_an_overlapping_prediction():
    """One voxel gets one colour, so a voxel in two classes cannot be written as one.
    A softmax output never trips it: its channels sum to one."""
    colormap = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    assert onehot_to_colours(np.array([[[0.2, 0.5, 0.3]]]), colormap)[0, 0].tolist() == [255, 0, 0]
    with pytest.raises(MaskError, match="2 classes at once"):
        onehot_to_colours(np.array([[[0.1, 0.9, 0.8]]]), colormap)


# ------------------------------------------------------------------ end to end


def test_the_class_fingerprint_tells_three_regions_from_three_classes(brats_like):
    """Written into every run record and every weights sidecar. `n_class` alone cannot
    tell the two apart, and weights from one must not be read back as the other."""
    engine = Engine(spec_for(brats_like))
    fingerprint = engine._class_fingerprint()
    assert fingerprint["n_class"] == 3
    assert fingerprint["regions"] == [{"name": r["name"], "of": r["of"]} for r in BRATS]
    assert "labels" not in fingerprint and "colormap" not in fingerprint


def test_a_run_with_regions_trains_predicts_and_evaluates(brats_like):
    """The join, not the halves. Green on both sides proves neither."""
    spec = spec_for(brats_like)
    engine = Engine(spec)
    history = engine.fit()["m"]
    assert len(history) == 1
    assert {"train_loss", "train_dice", "val_dice"} <= set(history.columns)

    prediction = engine.predict("m", "validation")[0]
    assert prediction.shape == (64, 64, 3)
    assert prediction.sum(axis=-1).max() > 1.0, "sigmoid channels, not a softmax"

    row = engine.evaluate("validation")[0]
    assert 0.0 <= row["dice"] <= 1.0
