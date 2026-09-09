"""Rank handling. These tests are the guarantee behind PLAN.md rule 1."""

import pytest

from pyplatypus import ConfigError, from_dict


def test_2d_shape_gives_rank_2(config):
    assert from_dict(config).models[0].rank == 2


def test_3d_shape_validates_without_any_code_change(config):
    """v0.1 ships 2D only, but the spec must already accept a volume. If this test ever
    needs new code to pass, the rank-agnostic design has been broken."""
    config["models"][0]["input_shape"] = [64, 128, 128]
    spec = from_dict(config)
    assert spec.models[0].rank == 3
    assert spec.rank == 3


def test_rank_4_is_refused(config):
    config["models"][0]["input_shape"] = [8, 16, 16, 16]
    with pytest.raises(ConfigError):
        from_dict(config)


def test_models_may_not_mix_ranks(config, model_block):
    volume = dict(model_block, name="unet3d", input_shape=[64, 64, 64])
    config["models"] = [model_block, volume]
    with pytest.raises(ConfigError, match="same spatial rank"):
        from_dict(config)


def test_shape_must_survive_the_pooling(config):
    """255 is not divisible by 2**4, which would blow up in the decoder."""
    config["models"][0]["input_shape"] = [255, 255]
    with pytest.raises(ConfigError) as caught:
        from_dict(config)
    assert "divisible by 16" in str(caught.value)


def test_fewer_blocks_allow_smaller_shapes(config):
    config["models"][0]["input_shape"] = [24, 24]
    config["models"][0]["blocks"] = 3
    assert from_dict(config).models[0].blocks == 3
