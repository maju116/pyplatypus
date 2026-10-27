"""Pretrained encoders: the seam, the refusals, and what is actually pretrained.

Nothing here downloads weights, and until `#294` this file claimed that the one question
needing a network was exercised by a test that is skipped without one. **There was no such
test**, and the claim had split a question that is really two:

* whether ImageNet weights *arrive* genuinely needs a network, and is still not tested here;
* whether they **survive construction** needs nothing at all, and is
  `test_a_transferred_backbone_survives_construction` below.

The second is what `#294` was: `initialise` walked every convolution under the network and
overwrote the backbone immediately after timm had loaded it. The test below would have
failed the day that shipped, offline, in milliseconds.
"""

import builtins

import pytest
import torch

from pyplatypus.models.encoders import PretrainedEncoder, verify_encoder
from pyplatypus.models.layers import ModelError
from pyplatypus.models.unet import build_model
from pyplatypus.spec.models import SegmentationModel

timm = pytest.importorskip("timm", reason="the `encoders` extra is not installed")


def spec(**overrides):
    base = {"name": "m", "input_shape": (64, 64), "channels": 3, "blocks": 4, "filters": 16}
    return SegmentationModel(**{**base, **overrides})


def encoder(**overrides):
    base = {"in_channels": 3, "blocks": 4, "filters": 16}
    return PretrainedEncoder(overrides.pop("name", "resnet34"), **{**base, **overrides})


# --- the contract ------------------------------------------------------------------------


def test_a_backbone_satisfies_the_encoder_contract():
    """The check that catches a half-resolution encoder has to pass for the real thing."""
    built = encoder()
    verify_encoder(built, 2, 3, 5)


def test_level_zero_is_ours_and_at_full_resolution():
    """An ImageNet stem strides by 2, so the finest level cannot come from the backbone.
    It is one block of ours, and the resolution is what the decoder's head needs."""
    built = encoder()
    features = built(torch.zeros(1, 3, 64, 64))
    assert built.own_level_zero
    assert built.stem is not None
    assert features[0].shape[-2:] == (64, 64)
    assert [f.shape[-1] for f in features] == [64, 32, 16, 8, 4]


def test_a_backbone_that_offers_full_resolution_is_used_directly():
    """vgg16 reports a 1/1 level, so prepending one of ours would waste it."""
    built = encoder(name="vgg16")
    assert not built.own_level_zero
    assert built.stem is None
    assert built(torch.zeros(1, 3, 64, 64))[0].shape[-2:] == (64, 64)


def test_a_whole_model_builds_and_keeps_the_input_size():
    model = build_model(spec(), encoder=encoder(), n_class=2)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


@pytest.mark.parametrize("architecture", ["u_net", "u_net_plus_plus", "res_u_net", "linknet"])
def test_every_architecture_accepts_a_backbone(architecture):
    """The backbone replaces the contracting path; the decoder is still the spec's choice."""
    model = build_model(spec(architecture=architecture), encoder=encoder(), n_class=2)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_gradients_reach_our_stage_as_well_as_the_backbone():
    model = build_model(spec(), encoder=encoder(), n_class=2)
    model(torch.randn(1, 3, 64, 64)).sum().backward()
    unused = [name for name, p in model.named_parameters() if p.grad is None]
    assert unused == []


@pytest.mark.parametrize("blocks", [2, 3, 4, 5])
def test_blocks_selects_how_many_backbone_stages_to_use(blocks):
    built = encoder(blocks=blocks)
    assert len(built.channels) == blocks + 1
    verify_encoder(built, 2, 3, blocks + 1)


def test_a_transferred_backbone_survives_construction():
    """What arrived pretrained is still there after the model is built. `#294`.

    `pretrained=False` on purpose: the question is whether construction *keeps* whatever
    the backbone was given, and random weights answer it exactly as well as ImageNet ones
    while needing no network. Measured when this was written: without the fix, 15 of 15
    convolutions in `resnet18` moved and one tensor's norm went 2.418 to 11.618.

    The second half is the control, and it is the one that makes this a test rather than an
    assertion that `initialise` does nothing: the full-resolution stage inside
    `PretrainedEncoder` is **ours**, starts random, and must still be initialised.
    """
    built = encoder(name="resnet18")
    before = {name: p.detach().clone() for name, p in built.named_parameters() if p.dim() == 4}
    assert any(name.startswith("backbone") for name in before), "no backbone convolutions found"
    assert any(name.startswith("stem") for name in before), "no stage of our own found"

    build_model(spec(encoder="resnet18"), n_class=2, encoder=built)

    after = {name: p.detach() for name, p in built.named_parameters() if p.dim() == 4}
    moved = {name for name in before if not torch.equal(before[name], after[name])}
    theirs = {name for name in moved if name.startswith("backbone")}
    assert not theirs, f"construction overwrote {len(theirs)} transferred convolutions"
    assert {name for name in moved if name.startswith("stem")}, (
        "our own stage was not initialised, so this test would pass on a model that "
        "initialises nothing at all"
    )


def test_what_the_trainer_protects_is_exactly_what_construction_left_alone():
    """One answer to *what arrived pretrained*, paired rather than asserted twice.

    `Trainer` reads `model.encoder.transferred` for the freeze and for
    `encoder_learning_rate`; `UShapedNet` reads the same property to decide what not to
    initialise. The two could drift if either side grew its own idea - a name pattern, a
    list of module types - so this compares the weights construction did not touch against
    the parameters that property offers, by name.
    """
    built = encoder()
    before = {name: p.detach().clone() for name, p in built.named_parameters() if p.dim() == 4}
    model = build_model(spec(encoder="resnet34"), n_class=2, encoder=built)
    after = {name: p.detach() for name, p in built.named_parameters() if p.dim() == 4}

    untouched = {name for name in before if torch.equal(before[name], after[name])}
    protected = {
        f"backbone.{name}"
        for name, p in model.encoder.transferred.named_parameters()
        if p.dim() == 4
    }
    assert protected, "the property offered nothing; the comparison would be two empty sets"
    assert untouched == protected


# --- the refusals -----------------------------------------------------------------------


def test_a_volume_is_refused_by_the_spec_before_anything_is_built():
    """Caught while the configuration is read, so a 3D run fails before a download."""
    with pytest.raises(ValueError, match="ImageNet is images"):
        spec(input_shape=(64, 64, 32), encoder="resnet34")


def test_the_encoder_itself_also_refuses_rank_three():
    """Belt and braces: the spec is the usual route in, but the class is public."""
    with pytest.raises(ModelError, match="2D only"):
        encoder(rank=3)


def test_pretrained_without_an_encoder_is_refused():
    """Otherwise the flag reads as transfer learning and does nothing at all."""
    with pytest.raises(ValueError, match="needs `encoder`"):
        spec(pretrained=True)


def test_a_patch_based_backbone_is_refused_by_name_and_reason():
    """convnext and swin start at 1/4. The missing 1/2 level is where a U-shaped decoder
    recovers fine detail, and inventing it would carry no pretrained information at all."""
    with pytest.raises(ModelError) as raised:
        encoder(name="convnext_tiny")
    message = str(raised.value)
    assert "convnext_tiny" in message
    assert "1/2" in message
    assert "resnet" in message  # and it says what to use instead


def test_asking_for_more_blocks_than_the_backbone_has_is_refused():
    with pytest.raises(ModelError, match=r"needs \[2, 4, 8, 16, 32, 64\]"):
        encoder(blocks=6)


def test_an_unknown_backbone_name_is_refused_with_somewhere_to_look():
    with pytest.raises(ModelError) as raised:
        encoder(name="not_a_real_backbone_xyz")
    assert "timm.list_models" in str(raised.value)


# --- channels ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_channel_counts_other_than_three(channels):
    """Grayscale CT has one, BraTS has four; ImageNet has three and timm adapts the stem."""
    built = PretrainedEncoder("resnet34", in_channels=channels, blocks=4, filters=16)
    verify_encoder(built, 2, channels, 5)


# --- normalisation ----------------------------------------------------------------------


def test_input_statistics_are_applied_only_with_the_weights_that_expect_them():
    """ImageNet statistics on a from-scratch run are meaningless numbers."""
    assert not encoder().normalises


def test_the_absence_of_timm_names_the_command_that_fixes_it():
    """The extra being optional is a promise that its absence is survivable and says so."""
    real_import = builtins.__import__

    def without_timm(name, *args, **kwargs):
        if name == "timm" or name.startswith("timm."):
            raise ImportError("hidden for this test")
        return real_import(name, *args, **kwargs)

    original = builtins.__import__
    builtins.__import__ = without_timm
    try:
        with pytest.raises(ModelError) as raised:
            encoder()
    finally:
        builtins.__import__ = original

    message = str(raised.value)
    assert "pyplatypus[encoders]" in message
    # And it names the route that needs nothing extra and works at both ranks.
    assert "built-in encoder" in message
