"""Pretrained encoders: the seam, the refusals, and what is actually pretrained.

Nothing here downloads weights. `pretrained=True` is exercised in one test that is skipped
without a network, because the point being proven - that the weights really arrive and are
not random - cannot be proven offline, and pretending otherwise with a mock would prove only
that the mock works.
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
