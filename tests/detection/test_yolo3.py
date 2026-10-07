"""The architecture, and the two checks that are not the same check.

The parameter count says the **inventory** is right - that Darknet-53 and the three heads
contain what they should, which is what the published COCO weights will be loaded into. It
says nothing about the **wiring**, and the first version of `forward` here proved that: it
passed a feature map straight to the smallest head and skipped the neck in front of it, and
the parameter count was still exactly 61,949,149 because the unused module was still there.
The gradient test is what caught it.
"""

import pytest
import torch

from pyplatypus.detection.metrics import DetectionError
from pyplatypus.detection.yolo3 import LEAKY_SLOPE, Darknet53, build_yolo3

#: YOLOv3's published parameter count for 80 classes and three anchors per grid.
PUBLISHED_PARAMETERS = 61_949_149


def test_the_parameter_count_is_the_published_one():
    """Which is the one external check on the architecture's inventory, and the thing the
    COCO weights require: a structure that merely resembles Darknet-53 loads them cleanly
    and predicts nonsense."""
    model = build_yolo3(n_class=80, anchors_per_grid=3)
    assert sum(p.numel() for p in model.parameters()) == PUBLISHED_PARAMETERS


def test_every_parameter_receives_a_gradient():
    """The check the parameter count cannot make. A module built and never called still
    counts towards the total - which is exactly the bug this caught."""
    model = build_yolo3(n_class=3, anchors_per_grid=2)
    sum(output.sum() for output in model(torch.zeros(2, 3, 128, 128))).backward()
    unused = [name for name, p in model.named_parameters() if p.grad is None]
    assert unused == []


def test_the_three_grids_come_out_coarsest_first():
    out = build_yolo3(n_class=80)(torch.zeros(1, 3, 416, 416))
    assert [tuple(t.shape) for t in out] == [
        (1, 13, 13, 3, 85),
        (1, 26, 26, 3, 85),
        (1, 52, 52, 3, 85),
    ]


def test_the_output_is_channels_last_matching_the_encoder():
    """So the loss compares like with like and a mismatch cannot hide in a permutation."""
    from pyplatypus.detection.encode import encode

    out = build_yolo3(n_class=3)(torch.zeros(1, 3, 416, 416))
    encoded = encode([[10, 10, 50, 50]], [0], n_class=3)
    for predicted, target in zip(out, encoded.targets, strict=True):
        assert tuple(predicted.shape[1:]) == target.shape


@pytest.mark.parametrize("n_class, anchors", [(1, 1), (3, 3), (3, 5), (80, 2)])
def test_classes_and_anchors_are_the_callers(n_class, anchors):
    """The flexibility the old package had and most YOLOv3 code does not: the head's width
    is `anchors_per_grid * (5 + n_class)` and both come from the caller."""
    out = build_yolo3(n_class=n_class, anchors_per_grid=anchors)(torch.zeros(1, 3, 160, 160))
    assert [tuple(t.shape) for t in out] == [
        (1, 5, 5, anchors, 5 + n_class),
        (1, 10, 10, anchors, 5 + n_class),
        (1, 20, 20, anchors, 5 + n_class),
    ]


def test_a_rectangular_non_standard_input_works():
    out = build_yolo3(n_class=3)(torch.zeros(1, 3, 320, 608))
    assert [tuple(t.shape) for t in out] == [
        (1, 10, 19, 3, 8),
        (1, 20, 38, 3, 8),
        (1, 40, 76, 3, 8),
    ]


def test_the_backbone_is_independent_of_classes_and_anchors():
    """Which is precisely why COCO's backbone can be reused on three classes of blood cell
    while its heads cannot: the heads' width depends on both."""
    first = sum(p.numel() for p in build_yolo3(n_class=3).backbone.parameters())
    second = sum(
        p.numel() for p in build_yolo3(n_class=80, anchors_per_grid=5).backbone.parameters()
    )
    assert first == second


def test_darknet_returns_three_feature_maps_at_strides_32_16_and_8():
    deep, middle, shallow = Darknet53(3)(torch.zeros(1, 3, 256, 256))
    assert deep.shape == (1, 1024, 8, 8)
    assert middle.shape == (1, 512, 16, 16)
    assert shallow.shape == (1, 256, 32, 32)


def test_the_leaky_slope_is_darknets_not_torchs():
    """0.1, not torch's default 0.01. Not a free choice - the published weights were fitted
    against this function."""
    assert LEAKY_SLOPE == 0.1
    model = build_yolo3(n_class=1)
    slopes = {m.negative_slope for m in model.modules() if isinstance(m, torch.nn.LeakyReLU)}
    assert slopes == {0.1}


def test_the_bias_is_dropped_wherever_batch_norm_follows():
    """Not a saving: it is what the published weights contain. A bias tensor this package
    expected and the file did not have would stop the load."""
    model = build_yolo3(n_class=1)
    for module in model.modules():
        norm = getattr(module, "norm", None)
        conv = getattr(module, "conv", None)
        if conv is not None:
            assert (conv.bias is None) == (norm is not None)


@pytest.mark.parametrize("shape", [(1, 3, 400, 416), (1, 3, 416, 100)])
def test_an_input_the_strides_do_not_divide_is_refused(shape):
    with pytest.raises(DetectionError, match="not divisible by 32"):
        build_yolo3(n_class=3)(torch.zeros(*shape))


def test_the_wrong_number_of_channels_is_refused():
    with pytest.raises(DetectionError, match="takes 3 channels"):
        build_yolo3(n_class=3, in_channels=3)(torch.zeros(1, 1, 64, 64))


def test_something_that_is_not_a_batch_of_images_is_refused():
    with pytest.raises(DetectionError, match="batch of images"):
        build_yolo3(n_class=3)(torch.zeros(3, 64, 64))


@pytest.mark.parametrize("n_class, anchors", [(0, 3), (3, 0)])
def test_impossible_shapes_are_refused_at_construction(n_class, anchors):
    with pytest.raises(DetectionError):
        build_yolo3(n_class=n_class, anchors_per_grid=anchors)


def test_a_single_channel_model_trains_on_grayscale():
    """Blood smears are colour, but a radiograph is not, and the backbone's first
    convolution is the only thing that changes."""
    model = build_yolo3(n_class=2, in_channels=1)
    out = model(torch.zeros(1, 1, 64, 64))
    assert out[0].shape == (1, 2, 2, 3, 7)
