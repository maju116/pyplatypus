"""Tiling: cut a big image up instead of shrinking it.

The old package had this as `h_splits`/`w_splits` and it worked one way only - images
went in, tiles came out, and nothing put the predicted tiles back together. Here the
spec side is rank-aware; the stitching is step 5's job and there is a test waiting for it.
"""

import pytest

from pyplatypus import ConfigError, from_dict


def test_no_tiling_by_default(config):
    model = from_dict(config).models[0]
    assert model.splits is None
    assert model.tiles_per_image == 1
    assert model.load_shape == (256, 256)


def test_tiling_sets_the_load_size(config):
    """A 2x3 grid of 256x256 tiles reads the source at 512x768."""
    config["models"][0]["splits"] = [2, 3]
    model = from_dict(config).models[0]
    assert model.load_shape == (512, 768)
    assert model.tiles_per_image == 6


def test_tiling_works_in_3d_untouched(config):
    """Same field, same code, one more dimension - PLAN.md rule 1 again."""
    config["models"][0]["input_shape"] = [64, 64, 64]
    config["models"][0]["splits"] = [2, 2, 2]
    model = from_dict(config).models[0]
    assert model.rank == 3
    assert model.load_shape == (128, 128, 128)
    assert model.tiles_per_image == 8


def test_splits_must_match_the_rank(config):
    config["models"][0]["splits"] = [2, 2, 2]  # but input_shape is 2D
    with pytest.raises(ConfigError, match="must match"):
        from_dict(config)


def test_all_ones_is_refused_as_a_mistake(config):
    """It would silently do nothing, which is exactly the class of bug extra='forbid'
    exists to prevent."""
    config["models"][0]["splits"] = [1, 1]
    with pytest.raises(ConfigError, match="does nothing"):
        from_dict(config)


def test_zero_splits_refused(config):
    config["models"][0]["splits"] = [0, 2]
    with pytest.raises(ConfigError):
        from_dict(config)


def test_tiles_keep_the_pooling_constraint(config):
    """input_shape is the tile size, so it is the tile that must survive the pooling."""
    config["models"][0]["input_shape"] = [250, 250]
    config["models"][0]["splits"] = [2, 2]
    with pytest.raises(ConfigError, match="divisible by 16"):
        from_dict(config)
