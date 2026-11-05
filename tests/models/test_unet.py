"""Shapes, composition and rank. The functional proof lives in test_learning.py."""

import pytest
import torch

from pyplatypus.models import UShapedEncoder, build_model
from pyplatypus.models.layers import ModelError
from pyplatypus.spec.common import Architecture
from pyplatypus.spec.models import SegmentationModel

ARCHITECTURES = list(Architecture)


def spec(**overrides):
    base = {"name": "m", "input_shape": (64, 64), "channels": 3, "blocks": 3, "filters": 8}
    return SegmentationModel(**{**base, **overrides})


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_preserves_the_input_size(architecture):
    model = build_model(spec(architecture=architecture), n_class=2)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 2, 64, 64)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_works_in_3d(architecture):
    """PLAN.md rule 2. If this needs new code, the builder stopped being rank-generic."""
    model = build_model(
        spec(architecture=architecture, input_shape=(32, 32, 32), channels=1, blocks=2), n_class=2
    )
    out = model(torch.randn(1, 1, 32, 32, 32))
    assert out.shape == (1, 2, 32, 32, 32)


@pytest.mark.parametrize(
    "modifier",
    [
        {"separable_conv": True},
        {"upsample": True},
        {"batch_normalization": False},
        {"spatial_dropout": False, "dropout": 0.2},
        {"block_width": 1},
        {"block_width": 4},
        {"activation": "gelu"},
        {"initialiser": "glorot_uniform"},
    ],
)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_modifiers_compose_with_every_architecture(architecture, modifier):
    """The point of replacing the boolean soup with an enum: 4 x 32 combinations that
    all mean something, instead of 256 of which most were nonsense."""
    model = build_model(spec(architecture=architecture, **modifier), n_class=2)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_separable_convolutions_are_much_cheaper():
    plain = sum(p.numel() for p in build_model(spec(), n_class=2).parameters())
    separable = sum(
        p.numel() for p in build_model(spec(separable_conv=True), n_class=2).parameters()
    )
    assert separable < plain / 3


def test_linknet_is_smaller_than_u_net():
    """Adding skips instead of concatenating halves the decoder's input width."""
    unet = sum(p.numel() for p in build_model(spec(), n_class=2).parameters())
    linknet = sum(
        p.numel()
        for p in build_model(spec(architecture=Architecture.LINKNET), n_class=2).parameters()
    )
    assert linknet < unet


def test_nested_architecture_is_larger():
    unet = sum(p.numel() for p in build_model(spec(), n_class=2).parameters())
    nested = sum(
        p.numel()
        for p in build_model(
            spec(architecture=Architecture.U_NET_PLUS_PLUS), n_class=2
        ).parameters()
    )
    assert nested > unet


def test_deep_supervision_returns_one_output_per_depth():
    model = build_model(
        spec(architecture=Architecture.U_NET_PLUS_PLUS, deep_supervision=True), n_class=2
    )
    outputs = model(torch.randn(1, 3, 64, 64))
    assert isinstance(outputs, tuple) and len(outputs) == 3
    assert all(o.shape == (1, 2, 64, 64) for o in outputs)


def test_outputs_are_logits_not_probabilities():
    """The loss applies its own activation; a model that already softmaxed would be
    silently wrong and hard to spot."""
    model = build_model(spec(), n_class=2)
    out = model(torch.randn(4, 3, 64, 64))
    assert out.min() < 0.0
    assert not torch.allclose(out.sum(dim=1), torch.ones(4, 64, 64), atol=1e-3)


@pytest.mark.parametrize("blocks,size", [(1, 32), (2, 64), (4, 128), (5, 128)])
def test_depth_and_size_combinations(blocks, size):
    model = build_model(spec(blocks=blocks, input_shape=(size, size)), n_class=2)
    assert model(torch.randn(1, 3, size, size)).shape == (1, 2, size, size)


def test_channels_other_than_three(spec_channels=1):
    model = build_model(spec(channels=spec_channels), n_class=2)
    assert model(torch.randn(1, spec_channels, 64, 64)).shape == (1, 2, 64, 64)


def test_multiclass_output():
    model = build_model(spec(), n_class=5)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 5, 64, 64)


def test_encoder_is_a_seam_the_decoder_accepts():
    """A pretrained backbone slots in here later without the decoder changing."""
    encoder = UShapedEncoder(2, 3, blocks=3, filters=8)
    model = build_model(spec(), encoder=encoder, n_class=2)
    assert model.encoder is encoder
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_an_encoder_of_the_wrong_depth_is_rejected():
    encoder = UShapedEncoder(2, 3, blocks=2, filters=8)
    with pytest.raises(ModelError, match="levels"):
        build_model(spec(blocks=3), encoder=encoder, n_class=2)


def test_encoder_reports_its_channels():
    encoder = UShapedEncoder(2, 3, blocks=3, filters=8)
    assert encoder.channels == (8, 16, 32, 64)
    features = encoder(torch.randn(1, 3, 64, 64))
    assert [f.shape[1] for f in features] == [8, 16, 32, 64]
    assert [f.shape[-1] for f in features] == [64, 32, 16, 8]


def test_gradients_reach_every_parameter():
    """A layer built but never wired in would train silently at zero. Catch it here."""
    model = build_model(spec(architecture=Architecture.U_NET_PLUS_PLUS), n_class=2)
    model(torch.randn(1, 3, 64, 64)).sum().backward()
    unused = [name for name, p in model.named_parameters() if p.grad is None]
    assert unused == []


class StubEncoder(torch.nn.Module):
    """A configurable stand-in, so the contract can be broken one way at a time.

    `stride_first` is what every ImageNet backbone does: the stem halves the input
    before anything is handed back as a skip connection.
    """

    def __init__(self, widths, *, stride_first=False, declare=None, drop_last=False):
        super().__init__()
        self.channels = tuple(declare if declare is not None else widths)
        self.drop_last = drop_last
        first_stride = 2 if stride_first else 1
        self.stem = torch.nn.Conv2d(3, widths[0], 3, stride=first_stride, padding=1)
        self.rest = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1)
                for i in range(len(widths) - 1)
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        features = [x]
        for layer in self.rest:
            x = layer(x)
            features.append(x)
        return features[:-1] if self.drop_last else features


def test_an_encoder_that_downsamples_before_the_first_skip_is_rejected():
    """The head is a 1x1 convolution on level 0, so level 0 sets the prediction's size.

    Without this check the model builds, trains, and produces masks at half the
    resolution of the ones it is scored against - and the only symptom is a tensor-size
    error raised from inside the loss, nowhere near the cause.
    """
    encoder = StubEncoder((8, 16, 32, 64), stride_first=True)
    with pytest.raises(ModelError, match="level 0"):
        build_model(spec(), encoder=encoder, n_class=2)


def test_the_rejection_names_the_reason_a_backbone_would_do_this():
    encoder = StubEncoder((8, 16, 32, 64), stride_first=True)
    with pytest.raises(ModelError, match="stem strides by 2"):
        build_model(spec(), encoder=encoder, n_class=2)


def test_an_encoder_that_misdeclares_its_widths_is_rejected():
    """The decoder sizes its convolutions from `channels` before seeing a tensor, so a
    wrong declaration is a wrong decoder."""
    encoder = StubEncoder((8, 16, 32, 64), declare=(8, 16, 32, 999))
    with pytest.raises(ModelError, match="declares channels"):
        build_model(spec(), encoder=encoder, n_class=2)


def test_an_encoder_returning_fewer_maps_than_it_declares_is_rejected():
    encoder = StubEncoder((8, 16, 32, 64), drop_last=True)
    with pytest.raises(ModelError, match="returned 3 feature maps"):
        build_model(spec(), encoder=encoder, n_class=2)


def test_the_probe_leaves_the_encoder_in_training_mode():
    """Verification runs a forward pass, so it must not quietly switch BatchNorm off for
    the rest of the run."""
    encoder = UShapedEncoder(2, 3, blocks=3, filters=8)
    assert encoder.training
    build_model(spec(), encoder=encoder, n_class=2)
    assert encoder.training


def test_verification_costs_one_tiny_forward_pass():
    """The probe is the smallest input the level count allows, not the spec's shape."""
    seen = []

    class Watching(UShapedEncoder):
        def forward(self, x):
            seen.append(tuple(x.shape))
            return super().forward(x)

    build_model(
        spec(input_shape=(256, 256), blocks=3),
        encoder=Watching(2, 3, blocks=3, filters=8),
        n_class=2,
    )
    assert seen == [(1, 3, 8, 8)]


# --- the fields actually reaching the modules ---------------------------------------------
#
# `pyplatypus#386`, and it is the largest unheld piece the tenth review pass measured: a field
# is validated in `spec/`, documented in the schema and in R's `@param`, and then read **once**
# by a builder - and until this block nothing asserted that the built object reflects it. Six
# reads in this file could each be replaced by a constant with the whole suite green.
#
# `test_modifiers_compose_with_every_architecture` above is why that was invisible: it builds
# with each modifier and asserts the output **shape**, which is unchanged by whether the
# modifier did anything at all. These count the modules instead, which is the cheapest
# question that can tell the two apart.


def _kinds(net):
    """How many modules of each class the network holds, by class name."""
    from collections import Counter

    return Counter(type(module).__name__ for module in net.modules())


def _depthwise(net):
    """Convolutions acting per channel, which is what `separable` builds."""
    return sum(
        1
        for module in net.modules()
        if hasattr(module, "groups") and module.groups > 1 and module.groups == module.in_channels
    )


def test_dropout_reaches_the_network_and_zero_builds_none():
    with_drop = _kinds(build_model(spec(dropout=0.5), n_class=2))
    without = _kinds(build_model(spec(dropout=0.0), n_class=2))
    assert with_drop["Dropout2d"] > 0
    assert without["Dropout2d"] == 0 and without["Dropout"] == 0


def test_spatial_dropout_decides_WHICH_dropout_is_built():
    """The field is not whether to drop but what: whole feature maps against single
    activations. Both answers build a module, so a count of "some dropout" cannot see it."""
    spatial = _kinds(build_model(spec(dropout=0.5, spatial_dropout=True), n_class=2))
    plain = _kinds(build_model(spec(dropout=0.5, spatial_dropout=False), n_class=2))
    assert spatial["Dropout2d"] > 0 and spatial["Dropout"] == 0
    assert plain["Dropout"] > 0 and plain["Dropout2d"] == 0


def test_batch_normalization_false_builds_no_normalisation():
    assert _kinds(build_model(spec(batch_normalization=True), n_class=2))["BatchNorm2d"] > 0
    assert _kinds(build_model(spec(batch_normalization=False), n_class=2))["BatchNorm2d"] == 0


def test_upsample_decides_whether_the_decoder_LEARNS_its_upsampling():
    """`upsample: true` asks for interpolation, which is the one that has no parameters -
    so a transposed convolution surviving in the tree is the field being ignored."""
    interpolated = _kinds(build_model(spec(upsample=True), n_class=2))
    learned = _kinds(build_model(spec(upsample=False), n_class=2))
    assert interpolated["ConvTranspose2d"] == 0 and interpolated["Upsample"] > 0
    assert learned["ConvTranspose2d"] > 0 and learned["Upsample"] == 0


def test_separable_conv_reaches_the_UPSAMPLING_path_too():
    """And it is only reachable with `upsample: true`, which is why the decoder's own read
    of this field was unheld while the encoder blocks' read was not: the learned branch
    takes no `separable` argument at all, so every test that leaves `upsample` alone is
    blind to this one."""
    separable = build_model(spec(upsample=True, separable_conv=True), n_class=2)
    plain = build_model(spec(upsample=True, separable_conv=False), n_class=2)
    assert _depthwise(separable.ups) > 0
    assert _depthwise(plain.ups) == 0


def test_the_initialiser_a_specification_names_is_the_one_that_runs():
    """Not a distribution test: two initialisers from one seed give different weights, and
    a constant in place of the field gives the same weights twice. That is exactly the
    mutation this is written for, and it needs no tolerance."""
    torch.manual_seed(0)
    he = build_model(spec(initialiser="he_normal"), n_class=2)
    torch.manual_seed(0)
    glorot = build_model(spec(initialiser="glorot_uniform"), n_class=2)

    first = next(iter(he.encoder.state_dict()))
    assert not torch.allclose(he.encoder.state_dict()[first], glorot.encoder.state_dict()[first])

    # The control, so the comparison is not reading the seed rather than the field.
    torch.manual_seed(0)
    again = build_model(spec(initialiser="he_normal"), n_class=2)
    assert torch.allclose(he.encoder.state_dict()[first], again.encoder.state_dict()[first])
