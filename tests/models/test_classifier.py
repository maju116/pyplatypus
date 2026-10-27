"""The pooled head, and the two encoder checks it deliberately does not ask for.

The network is three lines after the encoder, so most of what is worth asserting here is
about the seam: that a classifier reads the deepest width and nothing else, that the
decoder's requirements are not imposed on it - which is what makes `convnext_*` and
`swin_*` usable where a U-shaped model refuses them by name - and that what arrived
pretrained is still there afterwards.

Nothing here downloads weights. `pretrained=False` throughout, because every question below
is about what construction does to whatever the backbone was given, and random weights
answer that exactly as well.
"""

import pytest
import torch
from torch import nn

from pyplatypus.models import build_classifier
from pyplatypus.models.classifier import PooledClassifier
from pyplatypus.models.encoders import PretrainedEncoder, UShapedEncoder
from pyplatypus.models.layers import ModelError
from pyplatypus.spec import ClassificationModel

timm = pytest.importorskip("timm", reason="the `encoders` extra is not installed")


def spec(**overrides):
    return ClassificationModel(**{"name": "stage", "input_shape": (64, 64), **overrides})


# --- the network ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, blocks",
    [((64, 64), 4), ((30, 30), 3), ((32, 64, 64), 3), ((97, 97), 2)],
    ids=["square", "odd and small", "a volume", "not a multiple of anything"],
)
def test_one_row_of_logits_per_sample_at_any_size(shape, blocks):
    """A classifier has no decoder to line up with, so nothing has to divide by 2^blocks -
    97 and 30 are in here on purpose, and so is rank 3, which no pretrained backbone can
    do and ours can."""
    net = build_classifier(spec(input_shape=shape, blocks=blocks), n_class=5)
    out = net(torch.zeros(2, 3, *shape))
    assert out.shape == (2, 5)
    assert out.dtype == torch.float32


def test_the_head_is_sized_from_the_declared_width():
    """`filters * 2^blocks` for our encoder, and the head is built from the declaration
    rather than from a tensor - which is why the declaration is measured."""
    net = build_classifier(spec(blocks=3, filters=8), n_class=7)
    assert net.encoder.channels[-1] == 8 * 2**3
    assert isinstance(net.head, nn.Linear)
    assert net.head.in_features == 64
    assert net.head.out_features == 7


def test_the_logits_carry_no_activation():
    """The loss applies its own, as everywhere else in this package. A head that had been
    squashed would answer inside (0, 1) for every input; this one does not."""
    torch.manual_seed(0)
    net = build_classifier(spec(), n_class=3)
    out = net(torch.randn(8, 3, 64, 64))
    assert out.min() < 0 or out.max() > 1


@pytest.mark.parametrize("how", ["avg", "max"])
def test_both_poolings_are_reachable_and_are_not_the_same_function(how):
    """A field that selected nothing would pass a test that only checks the shape."""
    net = build_classifier(spec(pooling=how), n_class=2)
    torch.manual_seed(0)
    x = torch.randn(4, 3, 64, 64)
    net.eval()
    with torch.no_grad():
        mine = net(x)
        other = build_classifier(spec(pooling="max" if how == "avg" else "avg"), n_class=2)
    other.load_state_dict(net.state_dict())
    other.eval()
    with torch.no_grad():
        theirs = other(x)
    assert not torch.allclose(mine, theirs), "avg and max pooling gave the same answer"


def test_head_dropout_reaches_the_head_and_only_the_head():
    """`dropout` sits between convolutions and `head_dropout` before the dense layer. The
    second is the only one left when the contracting path is somebody else's, which is why
    it exists - so it has to be the thing that is actually built."""
    assert isinstance(build_classifier(spec(), n_class=2).drop, nn.Identity)
    built = build_classifier(spec(head_dropout=0.4), n_class=2)
    assert isinstance(built.drop, nn.Dropout)
    assert built.drop.p == 0.4


def test_the_initialiser_reaches_the_dense_head():
    """`initialise` knew only about convolutions until this head arrived, so a linear layer
    would have kept torch's default and the field would have been a setting that does
    nothing. Measured as a difference between two initialisers, with the same seed."""
    weights = {}
    for how in ("he_normal", "glorot_uniform"):
        torch.manual_seed(0)
        weights[how] = build_classifier(spec(initialiser=how), n_class=16).head.weight.detach()
    assert not torch.allclose(weights["he_normal"], weights["glorot_uniform"])


# --- the seam ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["resnet34", "efficientnet_b0", "convnext_tiny", "swin_tiny_patch4_window7_224"]
)
def test_a_backbone_a_u_shaped_model_refuses_is_usable_here(name):
    """The claim this slice is for. `convnext_*` and `swin_*` start at 1/4 and cannot supply
    the 1/2 level a decoder recovers fine detail from; a classifier pools the deepest level
    and needs none of it.

    The control is the other half: the same two names are still refused for a pyramid, which
    `tests/models/test_pretrained.py` holds.
    """
    net = build_classifier(spec(input_shape=(224, 224), encoder=name), n_class=3)
    assert net(torch.zeros(1, 3, 224, 224)).shape == (1, 3)
    assert len(net.encoder.channels) == 1, "a classifier takes one level, not a pyramid"

    # And it is the *deepest* level, asked of timm rather than assumed: a shallower one
    # would build and run, with the head sized from whatever it was given, and the only
    # symptom would be a classifier reading 1/4-scale features.
    offered = list(timm.create_model(name, features_only=True, pretrained=False).feature_info)
    assert net.encoder.channels[-1] == offered[-1]["num_chs"]
    assert offered[-1]["reduction"] == max(entry["reduction"] for entry in offered)


def test_a_channels_last_backbone_is_transposed_at_the_seam():
    """timm returns `swin_*` features channels-last - (1, 7, 7, 768) against convnext's
    (1, 768, 7, 7) - and the layout is read off the wrapper's own `output_fmt` rather than
    guessed from the shapes, which cannot be told apart when the channel count equals the
    grid.

    Both values of that flag occur, which is what makes this a test rather than a reading.
    """
    channels_last = PretrainedEncoder(
        "swin_tiny_patch4_window7_224", in_channels=3, bottleneck_only=True
    )
    channels_first = PretrainedEncoder("convnext_tiny", in_channels=3, bottleneck_only=True)
    assert channels_last.channels_last
    assert not channels_first.channels_last

    for built in (channels_last, channels_first):
        features = built(torch.zeros(1, 3, 224, 224))
        assert features[-1].shape[1] == built.channels[-1]


def test_a_pyramid_still_needs_blocks_and_a_bottleneck_still_refuses_them():
    """The two mistakes are refused by name rather than one of them being ignored: a
    pyramid without `blocks` has nothing to count, and `blocks` with `bottleneck_only`
    describes a pyramid that is not being built."""
    with pytest.raises(ModelError, match="a pyramid needs `blocks`"):
        PretrainedEncoder("resnet34", in_channels=3)
    with pytest.raises(ModelError, match="describe a pyramid"):
        PretrainedEncoder("resnet34", in_channels=3, blocks=4, bottleneck_only=True)


def test_a_backbone_that_cannot_run_at_this_size_says_so_while_the_model_is_built():
    """`swin_*` fixes its window to the size it was trained at. timm answers with a bare
    assertion naming neither the field nor the encoder, so the probe's failure is turned
    into a message that names both - and it happens here rather than on the first batch,
    which is what passing `input_shape` to the probe buys."""
    with pytest.raises(ModelError) as caught:
        build_classifier(spec(encoder="swin_tiny_patch4_window7_224"), n_class=3)
    message = str(caught.value)
    assert "(64, 64)" in message
    assert "`input_shape`" in message


def test_what_arrived_pretrained_survives_construction():
    """`#294` in the file that would otherwise have repeated it: `initialise` walks every
    layer under the network, and a backbone is one of them.

    The control is the second assertion - the head must have been initialised, or this
    would pass on a model that initialises nothing at all.
    """
    built = PretrainedEncoder("resnet18", in_channels=3, bottleneck_only=True)
    before = {name: p.detach().clone() for name, p in built.named_parameters() if p.dim() == 4}
    assert before, "no convolutions found in the backbone"

    net = PooledClassifier(
        spec(input_shape=(224, 224), encoder="resnet18"), n_class=3, encoder=built
    )
    after = {name: p.detach() for name, p in built.named_parameters() if p.dim() == 4}
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert net.head.weight.abs().sum() > 0


def test_an_encoder_whose_declaration_is_a_lie_is_refused():
    """The one check a classifier does rely on: the head is sized from `channels` before any
    tensor is seen, so a declaration that does not match reality builds a dense layer of the
    wrong width - and the failure would land in the first matrix multiply."""

    class Lying(UShapedEncoder):
        def __init__(self):
            super().__init__(2, 3, blocks=2, filters=8)
            self.channels = (8, 16, 999)

    with pytest.raises(ModelError, match="declares channels"):
        PooledClassifier(spec(), n_class=2, encoder=Lying())


def test_the_level_count_is_not_this_networks_business():
    """An encoder with one level and one with five both serve a classifier, which reads the
    deepest width and discards the rest. Asserting a count here would assert more than the
    consumer relies on - the opposite of what `#294` was."""
    deep = build_classifier(spec(blocks=5, input_shape=(64, 64)), n_class=2)
    assert len(deep.encoder.channels) == 6
    shallow = PooledClassifier(
        spec(input_shape=(224, 224), encoder="resnet34"),
        n_class=2,
        encoder=PretrainedEncoder("resnet34", in_channels=3, bottleneck_only=True),
    )
    assert len(shallow.encoder.channels) == 1
    for net, size in ((deep, 64), (shallow, 224)):
        assert net(torch.zeros(1, 3, size, size)).shape == (1, 2)
